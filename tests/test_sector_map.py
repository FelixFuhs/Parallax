import json

from sector_map import build_sector_map, load_sector_map, write_sector_map

HTML = """
<html>
  <body>
    <table id="constituents">
      <tr>
        <th>Symbol</th>
        <th>Security</th>
        <th>GICS Sector</th>
        <th>GICS Sub-Industry</th>
        <th>Headquarters Location</th>
        <th>Date added</th>
        <th>CIK</th>
        <th>Founded</th>
      </tr>
      <tr>
        <td>BRK.B</td>
        <td>Berkshire Hathaway</td>
        <td>Financials</td>
        <td>Multi-Sector Holdings</td>
        <td>Omaha, Nebraska</td>
        <td>2010-02-16</td>
        <td>1067983</td>
        <td>1839</td>
      </tr>
      <tr>
        <td>AAPL</td>
        <td>Apple Inc.</td>
        <td>Information Technology</td>
        <td>Technology Hardware</td>
        <td>Cupertino, California</td>
        <td>1982-11-30</td>
        <td>320193</td>
        <td>1977</td>
      </tr>
    </table>
  </body>
</html>
"""


def test_build_sector_map_parses_wikipedia_constituent_table(tmp_path):
    source = tmp_path / "sp500.html"
    source.write_text(HTML, encoding="utf-8")

    frame = build_sector_map(source, source_url="https://example.test/source")

    assert frame["ticker"].tolist() == ["AAPL", "BRK-B"]
    aapl = frame[frame["ticker"] == "AAPL"].iloc[0]
    assert aapl["sector"] == "Information Technology"
    assert aapl["sub_industry"] == "Technology Hardware"
    assert aapl["source_url"] == "https://example.test/source"


def test_write_sector_map_outputs_summary_and_loadable_csv(tmp_path):
    source = tmp_path / "sp500.html"
    source.write_text(HTML, encoding="utf-8")
    output = tmp_path / "sector_map.csv"
    summary_path = tmp_path / "summary.json"
    metadata_path = tmp_path / "metadata.json"

    summary = write_sector_map(
        source_html=source,
        output_path=output,
        summary_path=summary_path,
        metadata_path=metadata_path,
        source_url="https://example.test/source",
    )

    loaded = load_sector_map(output)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert saved_summary["current_snapshot_only"] is True
    assert loaded["BRK-B"]["sector"] == "Financials"
    assert metadata_path.exists()
