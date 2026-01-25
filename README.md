# filename: README.md
ArxivSuggestionSystem v1

Install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
Set API key: `export GEMINI_API_KEY='YOUR_KEY'`
Build profile: `python scripts/build_profile.py --config config/config.yaml`
Run digest: `python scripts/digest_arxiv.py --config config/config.yaml`
Output digest: `cache/last_digest.md`
Profile JSON: `profiles/group_profile.json`
Preferences can be set under `profile.preferences` or via `profile.preferences_path` JSON.
Profile keywords are phrase-first and recency-weighted to emphasize recent group-specific themes.
Digest ranking writes artifacts under `artifacts/run/` (raw, highly_relevant, inspiring).
