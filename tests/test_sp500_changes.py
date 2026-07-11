import json

from sp500_changes import build_sp500_changes, load_sp500_changes, write_sp500_changes

HTML = """
<html>
  <body>
    <table id="changes">
      <tr>
        <th rowspan="2">Effective Date</th>
        <th colspan="2">Added</th>
        <th colspan="2">Removed</th>
        <th rowspan="2">Reason</th>
      </tr>
      <tr>
        <th>Ticker</th>
        <th>Security</th>
        <th>Ticker</th>
        <th>Security</th>
      </tr>
      <tr>
        <td>May 7, 2026</td>
        <td>VEEV</td>
        <td>Veeva Systems</td>
        <td>CTRA</td>
        <td>Coterra Energy</td>
        <td>S&amp;P 500 constituent Devon Energy Corp. is acquiring Coterra Energy.</td>
      </tr>
      <tr>
        <td>November 3, 2025</td>
        <td>Q</td>
        <td>Qnity Electronics</td>
        <td></td>
        <td></td>
        <td>Dupont de Nemours spun off Qnity Electronics.</td>
      </tr>
    </table>
  </body>
</html>
"""


def test_build_sp500_changes_parses_selected_changes_table(tmp_path):
    source = tmp_path / "sp500.html"
    source.write_text(HTML, encoding="utf-8")

    frame = build_sp500_changes(source, source_url="https://example.test/source")

    assert frame["effective_date"].tolist() == ["2025-11-03", "2026-05-07"]
    latest = frame.iloc[-1]
    assert latest["added_ticker"] == "VEEV"
    assert latest["removed_ticker"] == "CTRA"
    assert bool(latest["approximate_membership_history"]) is True


def test_write_sp500_changes_outputs_summary_and_loadable_csv(tmp_path):
    source = tmp_path / "sp500.html"
    source.write_text(HTML, encoding="utf-8")
    output = tmp_path / "changes.csv"
    summary_path = tmp_path / "summary.json"
    metadata_path = tmp_path / "metadata.json"

    summary = write_sp500_changes(
        source_html=source,
        output_path=output,
        summary_path=summary_path,
        metadata_path=metadata_path,
        source_url="https://example.test/source",
    )

    loaded = load_sp500_changes(output)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert saved_summary["point_in_time_membership"] is False
    assert loaded["added_ticker"].tolist() == ["Q", "VEEV"]
    assert metadata_path.exists()
