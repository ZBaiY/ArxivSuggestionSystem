# filename: tests/unit/test_digest_arxiv_gemini.py
import json
import sys
from pathlib import Path

import yaml

import scripts.digest_arxiv as da


class DummyResp:
    def __init__(self, text: str, status_code: int = 200, json_data=None):
        self.text = text
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


def test_digest_creates_ranked_outputs(tmp_path, monkeypatch):
    feed = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>Clock <math display=\"inline\">x</math> proposal</title>
        <summary>Abstract text one.</summary>
        <updated>2025-01-02T00:00:00Z</updated>
        <published>2025-01-01T00:00:00Z</published>
        <author><name>A. Author</name></author>
        <link rel="alternate" href="https://arxiv.org/abs/2401.00001" />
        <category term="hep-ph" />
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00002v1</id>
        <title>Another title</title>
        <summary>Abstract text two.</summary>
        <updated>2025-01-02T00:00:00Z</updated>
        <published>2025-01-01T00:00:00Z</published>
        <author><name>B. Author</name></author>
        <link rel="alternate" href="https://arxiv.org/abs/2401.00002" />
        <category term="hep-th" />
      </entry>
    </feed>
    """

    def fake_get(*args, **kwargs):
        return DummyResp(feed, 200)

    relevant_json = {
        "highly_relevant": [
            {
                "arxiv_id": "2401.00001v1",
                "relevance_score": 92,
                "why": "Strong match.",
                "must_read": True,
                "tags": ["clock"],
            }
        ],
        "rejected": [
            {"arxiv_id": "2401.00002v1", "reason": "Not aligned"}
        ],
    }

    inspiring_json = {
        "inspiring": [
            {
                "arxiv_id": "2401.00002v1",
                "readability": 70,
                "distance": 80,
                "inspiration": 85,
                "actionability": 60,
                "inspiring_score": 61,
                "why": "Transferable idea.",
            }
        ],
        "not_inspiring": [],
    }

    call_count = {"n": 0}

    def fake_post(url, json=None, timeout=60):
        call_count["n"] += 1
        if call_count["n"] == 1:
            payload = {"candidates": [{"content": {"parts": [{"text": json_module(relevant_json)}]}}]}
        else:
            payload = {"candidates": [{"content": {"parts": [{"text": json_module(inspiring_json)}]}}]}
        return DummyResp("", 200, payload)

    def json_module(obj):
        return json_lib.dumps(obj)

    import json as json_lib

    monkeypatch.setattr(da.requests, "get", fake_get)
    monkeypatch.setattr(da.requests, "post", fake_post)
    monkeypatch.setenv("GEMINI_API_KEY", "test")

    prompts_dir = tmp_path / "prompts"
    (prompts_dir / "relevant").mkdir(parents=True)
    (prompts_dir / "inspiring").mkdir(parents=True)
    (prompts_dir / "relevant" / "system.txt").write_text("system", encoding="utf-8")
    (prompts_dir / "relevant" / "user.txt").write_text("{{GROUP_PROFILE_JSON}}\n{{PAPERS_JSON}}", encoding="utf-8")
    (prompts_dir / "inspiring" / "system.txt").write_text("system", encoding="utf-8")
    (prompts_dir / "inspiring" / "user.txt").write_text("{{GROUP_PROFILE_JSON}}\n{{PAPERS_JSON}}", encoding="utf-8")

    config = {
        "arxiv": {"categories": ["hep-ph", "hep-th"], "extra_query": "", "days_back": 2, "max_results": 10},
        "profile": {"group_name": "Group", "profile_path": "profiles/group_profile.json"},
        "gemini": {"model": "gemini-2.5-flash", "max_output_tokens": 2000},
        "output": {
            "abstract_snippet_chars": 200,
            "max_candidates_to_send_to_gemini": 10,
            "n_highly_relevant": 10,
            "n_inspiring": 10,
        },
    }

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(da.sys, "argv", ["digest_arxiv.py", "--config", str(cfg_path)])

    assert da.main() == 0

    run_date = da._run_date_utc()
    raw_path = tmp_path / "artifacts" / "run" / f"{run_date}.json"
    rel_path = tmp_path / "artifacts" / "run" / f"{run_date}.highly_relevant.json"
    insp_path = tmp_path / "artifacts" / "run" / f"{run_date}.inspiring.json"

    assert raw_path.exists()
    assert rel_path.exists()
    assert insp_path.exists()

    rel_data = json.loads(rel_path.read_text())
    insp_data = json.loads(insp_path.read_text())

    assert "papers" in rel_data and len(rel_data["papers"]) == 1
    assert "papers" in insp_data and len(insp_data["papers"]) == 1
