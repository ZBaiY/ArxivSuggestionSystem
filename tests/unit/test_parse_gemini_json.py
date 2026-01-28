# filename: tests/unit/test_parse_gemini_json.py
import pytest

import scripts.digest_arxiv as da


def test_parse_gemini_json_truncated():
    text = '{"highly_relevant": [{"arxiv_id": "123", "why": "oops"}'
    with pytest.raises(RuntimeError) as exc:
        da._parse_gemini_json(text)
    assert "Likely truncated" in str(exc.value)


def test_parse_gemini_json_fenced():
    text = """```json
{"highly_relevant": []}
```"""
    data = da._parse_gemini_json(text)
    assert data == {"highly_relevant": []}
