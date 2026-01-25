# filename: tests/unit/test_digest_arxiv.py
from pathlib import Path

import scripts.digest_arxiv as da


class DummyResp:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


def test_write_run_artifact_and_schema(tmp_path, monkeypatch):
    feed = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>Clock <math display=\"inline\">x</math> proposal</title>
        <summary>Abstract text.</summary>
        <updated>2025-01-02T00:00:00Z</updated>
        <published>2025-01-01T00:00:00Z</published>
        <author><name>A. Author</name></author>
        <link rel="alternate" href="https://arxiv.org/abs/2401.00001" />
        <category term="hep-ph" />
      </entry>
    </feed>
    """

    def fake_get(*args, **kwargs):
        return DummyResp(feed, 200)

    monkeypatch.setattr(da.requests, "get", fake_get)

    papers = da._fetch_arxiv_papers(["hep-ph"], "", 5)
    run_date = "2025-01-03"
    out_path = da._write_run_artifact(
        tmp_path / "artifacts" / "run",
        run_date,
        Path("config/config.yaml"),
        Path("profiles/group_profile.json"),
        "cat:hep-ph",
        papers,
    )

    assert out_path.name == "2025-01-03.json"
    data = da._load_json(out_path)
    assert data["run_date"] == "2025-01-03"
    assert data["source"] == "arxiv"
    assert data["query"] == "cat:hep-ph"
    assert len(data["papers"]) == 1
    paper = data["papers"][0]
    assert paper["arxiv_id"] == "2401.00001v1"
    assert paper["title"] == "Clock proposal"
    assert paper["abstract"] == "Abstract text."
