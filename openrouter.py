from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request

from dcf import DCFError, DCFResult, run_dcf
from parser import ParseError, ScenarioOverrides, ValuationInput, parse_input


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROMPT_TEMPLATE = Path("templates") / "dcf_research_prompt.md"
DEFAULT_REPORTS_DIR = Path("reports")
MAX_RETRIES = 3
RETRYABLE_HTTP_STATUS_CODES = {429}


@dataclass(frozen=True)
class TierConfig:
    tier: str
    model: str
    reasoning_effort: str
    timeout_seconds: int
    min_interval_seconds: float
    max_tokens: int
    input_cost_per_million: float
    output_cost_per_million: float


@dataclass(frozen=True)
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class ApiResult:
    payload: Mapping[str, Any]
    usage: UsageStats
    estimated_cost_usd: float


@dataclass
class RunStats:
    done: int = 0
    failed: int = 0
    skipped: int = 0
    total_cost_usd: float = 0.0
    total_flags: int = 0


@dataclass(frozen=True)
class TickerRunResult:
    ticker: str
    done: int
    failed: int
    skipped: int
    total_cost_usd: float
    total_flags: int
    output: str


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter request or response cannot be used."""


TIER_CONFIGS: dict[str, TierConfig] = {
    "cheap": TierConfig(
        tier="cheap",
        model="openai/gpt-5.4-nano:online",
        reasoning_effort="high",
        timeout_seconds=120,
        min_interval_seconds=6.0,
        max_tokens=32000,
        input_cost_per_million=0.20,
        output_cost_per_million=1.25,
    ),
    "full": TierConfig(
        tier="full",
        model="openai/gpt-5.4:online",
        reasoning_effort="xhigh",
        timeout_seconds=300,
        min_interval_seconds=12.0,
        max_tokens=64000,
        input_cost_per_million=2.50,
        output_cost_per_million=15.00,
    ),
}


class RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()
        self._next_available_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            ready_at = max(now, self._next_available_at)
            delay = ready_at - now
            self._next_available_at = ready_at + self._min_interval_seconds
        if delay > 0:
            time.sleep(delay)


def main() -> int:
    args = _parse_args()
    _load_dotenv(Path(".env"))
    _load_dotenv(Path(".env.local"))

    if args.compare and (args.tickers or args.ticker_file or args.dry_run):
        print("Error: --compare cannot be combined with tickers, --file, or --dry-run.", file=sys.stderr)
        return 1

    if args.compare:
        try:
            _print_comparison(DEFAULT_REPORTS_DIR, args.compare)
        except (OpenRouterError, OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    tickers = _resolve_tickers(args)
    if not tickers:
        print("Error: provide at least one ticker via CLI arguments or --file.", file=sys.stderr)
        return 1

    try:
        prompt_template = _load_prompt_template(DEFAULT_PROMPT_TEMPLATE)
    except (OSError, OpenRouterError):
        print(f"Error: unable to read prompt template '{DEFAULT_PROMPT_TEMPLATE}'.", file=sys.stderr)
        return 1

    if args.dry_run:
        _print_dry_run(prompt_template, tickers, _tiers_for_cli(args.tier))
        return 0

    if args.parallel < 1:
        print("Error: --parallel must be at least 1.", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: set OPENROUTER_API_KEY before running this script.", file=sys.stderr)
        return 1

    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_tiers = _tiers_for_cli(args.tier)
    rate_limiters = {
        tier_name: RateLimiter(TIER_CONFIGS[tier_name].min_interval_seconds)
        for tier_name in selected_tiers
    }
    stats = RunStats()
    run_date = date.today().isoformat()

    for result in _run_tickers(
        tickers=tickers,
        selected_tiers=selected_tiers,
        prompt_template=prompt_template,
        api_key=api_key,
        rate_limiters=rate_limiters,
        run_date=run_date,
        parallel=args.parallel,
    ):
        if result.output:
            print(result.output)
        stats.done += result.done
        stats.failed += result.failed
        stats.skipped += result.skipped
        stats.total_cost_usd += result.total_cost_usd
        stats.total_flags += result.total_flags
        _print_running_totals(stats)

    _print_summary(stats)
    return 1 if stats.failed else 0


def _parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Generate OpenRouter-backed equity research JSON and save validated reports."
    )
    cli.add_argument("tickers", nargs="*", help="Ticker symbols to research.")
    cli.add_argument("--file", dest="ticker_file", help="Path to a text file containing ticker symbols.")
    cli.add_argument("--tier", choices=("cheap", "full", "both"), default="cheap", help="Model tier to run.")
    cli.add_argument("--parallel", type=int, default=1, help="Number of tickers to process concurrently.")
    cli.add_argument("--dry-run", action="store_true", help="Print the rendered prompt without calling the API.")
    cli.add_argument(
        "--compare",
        metavar="TICKER",
        help="Load the latest cheap/full reports for a ticker and print a side-by-side summary.",
    )
    return cli.parse_args()


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.compare and (args.tickers or args.ticker_file):
        raise SystemExit("Error: --compare cannot be combined with tickers or --file.")

    raw_values: list[str] = []
    if args.ticker_file:
        raw_values.extend(_read_tickers_file(Path(args.ticker_file)))
    raw_values.extend(args.tickers)

    tickers: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for token in re.split(r"[\s,]+", raw_value.strip()):
            if not token:
                continue
            ticker = token.upper()
            if ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def _read_tickers_file(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"Error: unable to read ticker file '{path}'.") from exc

    values: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.append(stripped)
    return values


def _tiers_for_cli(selected_tier: str) -> list[str]:
    if selected_tier == "both":
        return ["cheap", "full"]
    return [selected_tier]


def _load_prompt_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8")
    if "{TICKER}" not in template:
        raise OpenRouterError(f"Prompt template '{path}' must include the {{TICKER}} placeholder.")
    return template


def _render_prompt(template: str, ticker: str) -> str:
    return template.replace("{TICKER}", ticker)


def _print_dry_run(prompt_template: str, tickers: list[str], tiers: list[str]) -> None:
    for index, ticker in enumerate(tickers, start=1):
        if index > 1:
            print()
        print(f"Ticker: {ticker}")
        for tier_name in tiers:
            config = TIER_CONFIGS[tier_name]
            print(
                "Tier: "
                f"{tier_name} | model={config.model} | reasoning={config.reasoning_effort} "
                f"| max_tokens={config.max_tokens} | timeout={config.timeout_seconds}s"
            )
        print()
        print(_render_prompt(prompt_template, ticker))


def _run_tickers(
    *,
    tickers: list[str],
    selected_tiers: list[str],
    prompt_template: str,
    api_key: str,
    rate_limiters: dict[str, RateLimiter],
    run_date: str,
    parallel: int,
):
    if parallel == 1:
        for ticker in tickers:
            yield _process_ticker(
                ticker=ticker,
                selected_tiers=selected_tiers,
                prompt_template=prompt_template,
                api_key=api_key,
                rate_limiters=rate_limiters,
                run_date=run_date,
            )
        return

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(
                _process_ticker,
                ticker=ticker,
                selected_tiers=selected_tiers,
                prompt_template=prompt_template,
                api_key=api_key,
                rate_limiters=rate_limiters,
                run_date=run_date,
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                yield future.result()
            except Exception as exc:
                yield TickerRunResult(
                    ticker=ticker,
                    done=0,
                    failed=1,
                    skipped=0,
                    total_cost_usd=0.0,
                    total_flags=0,
                    output=f"{ticker}\n{ticker} | failed: unexpected worker error: {exc}",
                )


def _process_ticker(
    *,
    ticker: str,
    selected_tiers: list[str],
    prompt_template: str,
    api_key: str,
    rate_limiters: dict[str, RateLimiter],
    run_date: str,
) -> TickerRunResult:
    lines = [ticker]
    prompt = _render_prompt(prompt_template, ticker)
    done = 0
    failed = 0
    skipped = 0
    total_cost_usd = 0.0
    total_flags = 0

    for tier_name in selected_tiers:
        config = TIER_CONFIGS[tier_name]
        report_path = _report_path(DEFAULT_REPORTS_DIR, ticker, run_date, tier_name)
        if report_path.exists():
            skipped += 1
            lines.append(f"Skip {ticker} {tier_name}: report already exists at {report_path}")
            continue

        api_result: ApiResult | None = None
        extraction_warnings: list[str] = []
        request_logs: list[str] = []
        try:
            api_result = _request_openrouter(
                prompt=prompt,
                config=config,
                api_key=api_key,
                rate_limiter=rate_limiters[tier_name],
                log_lines=request_logs,
            )
            total_cost_usd += api_result.estimated_cost_usd

            raw_report, extraction_warnings = _extract_report_payload(api_result.payload, config)
            valuation_input = parse_input(raw_report)
            valuation_result = run_dcf(valuation_input)
            quality_flags = _validate_quality(valuation_input, raw_report)
            saved_report = _build_saved_report(
                valuation_input=valuation_input,
                valuation_result=valuation_result,
                config=config,
                usage=api_result.usage,
                estimated_cost_usd=api_result.estimated_cost_usd,
                quality_flags=quality_flags,
            )
            _write_json(report_path, saved_report)
        except (OpenRouterError, ParseError, DCFError, OSError, ValueError) as exc:
            failed += 1
            lines.extend(request_logs)
            lines.append(
                _format_request_log(
                    ticker=ticker,
                    config=config,
                    usage=api_result.usage if api_result else None,
                    cost_usd=api_result.estimated_cost_usd if api_result else None,
                    quality_flags=[],
                    outcome=f"failed: {exc}",
                )
            )
            continue

        done += 1
        total_flags += len(quality_flags)
        lines.extend(request_logs)
        for warning in extraction_warnings:
            lines.append(f"Warning: {ticker} {tier_name}: {warning}")
        lines.append(
            _format_request_log(
                ticker=ticker,
                config=config,
                usage=api_result.usage,
                cost_usd=api_result.estimated_cost_usd,
                quality_flags=quality_flags,
                outcome=f"saved={report_path}",
            )
        )

    return TickerRunResult(
        ticker=ticker,
        done=done,
        failed=failed,
        skipped=skipped,
        total_cost_usd=total_cost_usd,
        total_flags=total_flags,
        output="\n".join(lines),
    )


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OpenRouterError(f"Unable to read dotenv file '{path}'.") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise OpenRouterError(f"Invalid dotenv line in '{path}' at {line_number}.")

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise OpenRouterError(f"Invalid dotenv key in '{path}' at {line_number}.")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _request_openrouter(
    *,
    prompt: str,
    config: TierConfig,
    api_key: str,
    rate_limiter: RateLimiter,
    log_lines: list[str],
) -> ApiResult:
    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": config.reasoning_effort},
        "max_tokens": config.max_tokens,
        "stream": False,
        "provider": {"require_parameters": True},
        # Response healing only works on non-streaming requests and helps salvage malformed JSON.
        "plugins": [{"id": "response-healing"}],
    }
    if config.tier != "cheap":
        body["response_format"] = {"type": "json_object"}
    encoded_body = json.dumps(body).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if referer := os.getenv("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_TITLE"):
        headers["X-Title"] = title

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        rate_limiter.wait()
        try:
            req = request.Request(
                OPENROUTER_URL,
                data=encoded_body,
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
            payload = _load_json_mapping(response_body, "OpenRouter response")
            usage = _extract_usage(payload)
            return ApiResult(
                payload=payload,
                usage=usage,
                estimated_cost_usd=_estimate_cost(config, usage),
            )
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            if _is_retryable_status(exc.code) and attempt <= MAX_RETRIES:
                delay = _retry_delay_seconds(attempt, exc.headers.get("Retry-After"))
                log_lines.append(
                    f"Retry {attempt}/{MAX_RETRIES} for {config.model} after HTTP {exc.code}; "
                    f"sleeping {delay:.1f}s"
                )
                time.sleep(delay)
                last_error = exc
                continue
            raise OpenRouterError(
                f"OpenRouter HTTP {exc.code}: {_truncate(response_body)}"
            ) from exc
        except (error.URLError, socket.timeout, TimeoutError) as exc:
            if attempt <= MAX_RETRIES:
                delay = _retry_delay_seconds(attempt, None)
                log_lines.append(
                    f"Retry {attempt}/{MAX_RETRIES} for {config.model} after network error; "
                    f"sleeping {delay:.1f}s"
                )
                time.sleep(delay)
                last_error = exc
                continue
            raise OpenRouterError(f"Network error while calling OpenRouter: {exc}") from exc

    raise OpenRouterError(f"OpenRouter request failed after retries: {last_error}")


def _extract_report_payload(
    payload: Mapping[str, Any],
    config: TierConfig,
) -> tuple[Mapping[str, Any], list[str]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("OpenRouter response did not include choices[0].")

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise OpenRouterError("OpenRouter response choice was not an object.")

    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise OpenRouterError("OpenRouter response did not include a message object.")

    content = message.get("content")
    if config.tier == "cheap":
        return _extract_json_object_with_healing(content)
    return _extract_json_object(content), []


def _extract_json_object(content: Any) -> Mapping[str, Any]:
    text = _content_to_text(content).strip()
    if not text:
        raise OpenRouterError("Model returned empty message content.")

    candidate_texts = [text]
    if text.startswith("```"):
        candidate_texts.append(re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL))

    decoder = json.JSONDecoder()
    for candidate in candidate_texts:
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return parsed

        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                parsed, end = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            trailing = stripped[index + end :].strip()
            if isinstance(parsed, Mapping) and not trailing:
                return parsed

    raise OpenRouterError("Unable to parse a JSON object from choices[0].message.content.")


def _extract_json_object_with_healing(content: Any) -> tuple[Mapping[str, Any], list[str]]:
    text = _content_to_text(content).strip()
    if not text:
        raise OpenRouterError("Model returned empty message content.")

    warnings: list[str] = []
    stripped = _strip_markdown_code_fences(text)
    if stripped != text:
        warnings.append("healed JSON by stripping markdown code fences before parsing.")

    parsed = _extract_first_json_object(stripped)
    if parsed is not None:
        _, start, end = parsed
        if start != 0 or end != len(stripped):
            warnings.append("healed JSON by extracting the first JSON object from surrounding text.")
        return parsed[0], warnings

    raise OpenRouterError("Unable to parse healed JSON from the cheap-tier response.")


def _extract_first_json_object(text: str) -> tuple[Mapping[str, Any], int, int] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed, index, index + end
    return None


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    return re.sub(r"```(?:json)?|```", "", stripped, flags=re.IGNORECASE).strip()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, Mapping):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("value"), str):
            return content["value"]
        for key in ("content", "parts"):
            nested = content.get(key)
            if isinstance(nested, (str, list, Mapping)) or nested is None:
                return _content_to_text(nested)
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            if isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            if item.get("type") == "text" and isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "".join(parts)
    raise OpenRouterError("Unsupported message.content format in OpenRouter response.")


def _extract_usage(payload: Mapping[str, Any]) -> UsageStats:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return UsageStats(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0)

    completion_details = usage.get("completion_tokens_details")
    reasoning_tokens = 0
    if isinstance(completion_details, Mapping):
        reasoning_tokens = _coerce_int(completion_details.get("reasoning_tokens"))
    if reasoning_tokens == 0:
        reasoning_tokens = _coerce_int(usage.get("reasoning_tokens"))

    return UsageStats(
        prompt_tokens=_coerce_int(usage.get("prompt_tokens")),
        completion_tokens=_coerce_int(usage.get("completion_tokens")),
        reasoning_tokens=reasoning_tokens,
    )


def _estimate_cost(config: TierConfig, usage: UsageStats) -> float:
    prompt_cost = (usage.prompt_tokens / 1_000_000) * config.input_cost_per_million
    output_tokens = usage.completion_tokens + usage.reasoning_tokens
    output_cost = (output_tokens / 1_000_000) * config.output_cost_per_million
    return round(prompt_cost + output_cost, 6)


def _validate_quality(
    valuation_input: ValuationInput,
    raw_report: Mapping[str, Any],
) -> list[str]:
    flags: list[str] = []

    current_price = valuation_input.assumptions.current_price
    if current_price in (None, 0):
        flags.append("stale_price")

    if not 0.04 <= valuation_input.assumptions.wacc <= 0.20:
        flags.append("wacc_outlier")

    if not 0.00 <= valuation_input.assumptions.terminal_growth <= 0.05:
        flags.append("terminal_growth_outlier")

    if _has_margin_reversal(valuation_input):
        flags.append("margin_reversal")

    if any(
        growth > 0.80 or growth < -0.50
        for growth in valuation_input.forecast.revenue_growth.values()
    ):
        flags.append("revenue_explosion")

    if len(valuation_input.comps) < 3:
        flags.append("missing_comps")

    if any(_scenario_is_empty(valuation_input.scenarios.get(name, ScenarioOverrides())) for name in ("bull", "bear")):
        flags.append("missing_scenarios")

    if _has_internal_inconsistency(valuation_input, raw_report):
        flags.append("internal_inconsistency")

    if _is_suspiciously_round(valuation_input):
        flags.append("suspiciously_round")

    return flags


def _has_margin_reversal(valuation_input: ValuationInput) -> bool:
    last_year = valuation_input.last_historical_year
    last_revenue = valuation_input.historical.revenue[last_year]
    if last_revenue == 0:
        return False

    last_margin = valuation_input.historical.ebit[last_year] / last_revenue
    if last_margin > 0:
        return any(value < 0 for value in valuation_input.forecast.ebit_margin.values())
    if last_margin < 0:
        return any(value > 0 for value in valuation_input.forecast.ebit_margin.values())
    return False


def _scenario_is_empty(scenario: ScenarioOverrides) -> bool:
    if scenario.description:
        return False
    forecast = scenario.forecast
    if any(
        getattr(forecast, field_name)
        for field_name in (
            "revenue_growth",
            "ebit_margin",
            "da_pct_sales",
            "capex_pct_sales",
            "nwc_pct_sales",
        )
    ):
        return False
    assumptions = scenario.assumptions
    return all(getattr(assumptions, field.name) is None for field in fields(assumptions))


def _has_internal_inconsistency(
    valuation_input: ValuationInput,
    raw_report: Mapping[str, Any],
) -> bool:
    support_block = raw_report.get("validation_support")
    if not isinstance(support_block, Mapping):
        return False

    forecast_revenue = support_block.get("forecast_revenue")
    if not isinstance(forecast_revenue, Mapping):
        return False

    first_forecast_year = valuation_input.forecast.years[0]
    reported_revenue = _coerce_float(forecast_revenue.get(str(first_forecast_year)))
    if reported_revenue is None:
        reported_revenue = _coerce_float(forecast_revenue.get(first_forecast_year))
    if reported_revenue is None:
        return False

    last_year = valuation_input.last_historical_year
    historical_revenue = valuation_input.historical.revenue[last_year]
    expected_revenue = historical_revenue * (
        1.0 + valuation_input.forecast.revenue_growth[first_forecast_year]
    )
    tolerance = max(abs(expected_revenue), abs(reported_revenue), 1.0) * 0.05
    return abs(expected_revenue - reported_revenue) > tolerance


def _is_suspiciously_round(valuation_input: ValuationInput) -> bool:
    forecast_values: list[float] = []
    forecast = valuation_input.forecast
    for field_name in (
        "revenue_growth",
        "ebit_margin",
        "da_pct_sales",
        "capex_pct_sales",
        "nwc_pct_sales",
    ):
        forecast_values.extend(getattr(forecast, field_name).values())

    if not forecast_values:
        return False

    rounded_values = sum(1 for value in forecast_values if _is_multiple_of(value, 0.05))
    return rounded_values > len(forecast_values) / 2


def _is_multiple_of(value: float, step: float) -> bool:
    quotient = value / step
    return abs(quotient - round(quotient)) <= 1e-9


def _build_saved_report(
    *,
    valuation_input: ValuationInput,
    valuation_result: DCFResult,
    config: TierConfig,
    usage: UsageStats,
    estimated_cost_usd: float,
    quality_flags: list[str],
) -> dict[str, Any]:
    report = asdict(valuation_input)
    parser_quality_flags = report.pop("quality_flags", [])
    meta_quality_flags = list(dict.fromkeys([*parser_quality_flags, *quality_flags]))
    report["_valuation"] = asdict(valuation_result)
    report["_meta"] = {
        "model": config.model,
        "tier": config.tier,
        "reasoning_effort": config.reasoning_effort,
        "generated_at": _utc_timestamp(),
        "cost_usd": estimated_cost_usd,
        "tokens": {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "reasoning": usage.reasoning_tokens,
        },
        "quality_flags": meta_quality_flags,
    }
    return report


def _report_path(reports_dir: Path, ticker: str, report_date: str, tier: str) -> Path:
    safe_ticker = re.sub(r"[^A-Za-z0-9._-]+", "-", ticker.upper()).strip("-") or "UNKNOWN"
    return reports_dir / f"{safe_ticker}_{report_date}_{tier}.json"


def _print_comparison(reports_dir: Path, ticker: str) -> None:
    normalized_ticker = ticker.upper()
    cheap_path = _find_latest_report(reports_dir, normalized_ticker, "cheap")
    full_path = _find_latest_report(reports_dir, normalized_ticker, "full")
    if cheap_path is None or full_path is None:
        raise OpenRouterError(
            f"Missing comparison reports for {normalized_ticker}. Expected both cheap and full reports in reports/."
        )

    cheap_report = _load_json_mapping(cheap_path.read_text(encoding="utf-8"), str(cheap_path))
    full_report = _load_json_mapping(full_path.read_text(encoding="utf-8"), str(full_path))

    cheap_summary = _comparison_summary(cheap_report)
    full_summary = _comparison_summary(full_report)

    rows = [
        ("Report", cheap_path.name, full_path.name),
        ("Intrinsic value", _format_currency(cheap_summary["intrinsic_value"]), _format_currency(full_summary["intrinsic_value"])),
        ("Current price", _format_currency(cheap_summary["current_price"]), _format_currency(full_summary["current_price"])),
        ("Upside", _format_pct(cheap_summary["upside"]), _format_pct(full_summary["upside"])),
        ("Quality flags", _format_flags(cheap_summary["quality_flags"]), _format_flags(full_summary["quality_flags"])),
    ]

    widths = [
        max(len(str(row[index])) for row in [("Metric", "cheap", "full"), *rows])
        for index in range(3)
    ]

    print(f"Ticker: {normalized_ticker}")
    print(f"{'Metric'.ljust(widths[0])}  {'cheap'.ljust(widths[1])}  {'full'.ljust(widths[2])}")
    print(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}")
    for row in rows:
        print(f"{str(row[0]).ljust(widths[0])}  {str(row[1]).ljust(widths[1])}  {str(row[2]).ljust(widths[2])}")


def _find_latest_report(reports_dir: Path, ticker: str, tier: str) -> Path | None:
    if not reports_dir.exists():
        return None

    safe_ticker = re.sub(r"[^A-Za-z0-9._-]+", "-", ticker.upper()).strip("-") or "UNKNOWN"
    pattern = re.compile(rf"^{re.escape(safe_ticker)}_(\d{{4}}-\d{{2}}-\d{{2}})_{re.escape(tier)}\.json$")
    candidates: list[tuple[str, float, Path]] = []

    for path in reports_dir.glob(f"{safe_ticker}_*_{tier}.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((match.group(1), mtime, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _comparison_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    valuation = report.get("_valuation")
    if not isinstance(valuation, Mapping):
        raise OpenRouterError("Report is missing _valuation.")

    scenarios = valuation.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise OpenRouterError("Report valuation is missing scenarios.")

    base = scenarios.get("base")
    if not isinstance(base, Mapping):
        raise OpenRouterError("Report valuation is missing the base scenario.")

    meta = report.get("_meta")
    if not isinstance(meta, Mapping):
        raise OpenRouterError("Report is missing _meta.")

    quality_flags = meta.get("quality_flags")
    if not isinstance(quality_flags, list):
        quality_flags = []

    return {
        "intrinsic_value": _coerce_float(base.get("per_share_value")),
        "current_price": _coerce_float(base.get("current_price")),
        "upside": _coerce_float(base.get("upside_downside_pct")),
        "quality_flags": [str(flag) for flag in quality_flags],
    }


def _format_request_log(
    *,
    ticker: str,
    config: TierConfig,
    usage: UsageStats | None,
    cost_usd: float | None,
    quality_flags: list[str],
    outcome: str,
) -> str:
    tokens = "tokens=n/a"
    if usage is not None:
        tokens = (
            "tokens="
            f"prompt:{usage.prompt_tokens},completion:{usage.completion_tokens},"
            f"reasoning:{usage.reasoning_tokens}"
        )

    cost = "cost=n/a"
    if cost_usd is not None:
        cost = f"cost=${cost_usd:.6f}"

    flags = _format_flags(quality_flags)
    return (
        f"{ticker} | tier={config.tier} | model={config.model} | "
        f"{tokens} | {cost} | flags={flags} | {outcome}"
    )


def _print_running_totals(stats: RunStats) -> None:
    print(
        "Totals: "
        f"done={stats.done} "
        f"failed={stats.failed} "
        f"skipped={stats.skipped} "
        f"total_cost=${stats.total_cost_usd:.6f} "
        f"total_flags={stats.total_flags}"
    )


def _print_summary(stats: RunStats) -> None:
    print(
        "Summary: "
        f"done={stats.done}, "
        f"failed={stats.failed}, "
        f"skipped={stats.skipped}, "
        f"total_cost=${stats.total_cost_usd:.6f}, "
        f"total_flags={stats.total_flags}"
    )


def _format_currency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_flags(flags: list[str]) -> str:
    return ",".join(flags) if flags else "none"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json_mapping(text: str, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"{label} did not contain valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise OpenRouterError(f"{label} must be a top-level JSON object.")
    return payload


def _is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_HTTP_STATUS_CODES or 500 <= status_code < 600


def _retry_delay_seconds(attempt: int, retry_after: str | None) -> float:
    parsed_retry_after = _parse_retry_after_seconds(retry_after)
    if parsed_retry_after is not None:
        return max(parsed_retry_after, 0.0)
    return min(30.0, float(2**attempt))


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, IndexError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return (retry_at - datetime.now(timezone.utc)).total_seconds()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _truncate(value: str, limit: int = 400) -> str:
    stripped = " ".join(value.split())
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3]}..."


if __name__ == "__main__":
    raise SystemExit(main())
