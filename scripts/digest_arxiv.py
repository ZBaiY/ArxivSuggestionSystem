# filename: scripts/digest_arxiv.py
#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import requests
import yaml

ARXIV_API_URL = "http://export.arxiv.org/api/query"

FULLWIDTH_MAP = str.maketrans(
    {
        "：": ":",
        "，": ",",
        "。": ".",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "；": ";",
    }
)

PROMPT_VERSION = "v2-short"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    summary: str
    link: str
    updated: datetime
    published: datetime
    categories: List[str]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}")


def _normalize_generated_text(s: str) -> str:
    if not s:
        return ""
    return s.translate(FULLWIDTH_MAP)


def _strip_markup(s: str) -> str:
    if not s:
        return ""
    text = re.sub(r"<math[^>]*>.*?</math>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_for_extraction(s: str) -> str:
    if not s:
        return ""
    text = _strip_markup(s)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\(mathrm|mathbf|text)\{.*?\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\(left|right|displaystyle)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_time(value: str) -> datetime:
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _extract_arxiv_id(entry_id: str) -> str:
    if not entry_id:
        return ""
    parts = entry_id.rstrip("/").split("/")
    return parts[-1]


def _build_search_query(categories: List[str], extra_query: str) -> str:
    cat_query = " OR ".join([f"cat:{c}" for c in categories]) if categories else ""
    if extra_query:
        if cat_query:
            return f"({cat_query}) AND ({extra_query})"
        return extra_query
    return cat_query


def _fetch_arxiv_papers(categories: List[str], extra_query: str, max_results: int) -> List[Paper]:
    search_query = _build_search_query(categories, extra_query)
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
    }
    try:
        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch arXiv feed: {e}")
    if resp.status_code != 200:
        raise RuntimeError(f"arXiv API error: HTTP {resp.status_code}")

    feed = feedparser.parse(resp.text)
    papers: List[Paper] = []
    for entry in feed.entries:
        arxiv_id = _extract_arxiv_id(getattr(entry, "id", ""))
        title = _strip_markup(" ".join(getattr(entry, "title", "").split()))
        summary = " ".join(getattr(entry, "summary", "").split())
        link = ""
        for l in entry.get("links", []):
            if l.get("rel") == "alternate":
                link = l.get("href", "")
                break
        if not link:
            link = getattr(entry, "link", "")
        authors = [a.name for a in entry.get("authors", []) if getattr(a, "name", None)]
        updated = _parse_time(getattr(entry, "updated", ""))
        published = _parse_time(getattr(entry, "published", ""))
        categories = [t["term"] for t in entry.get("tags", []) if "term" in t]

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                summary=summary,
                link=link,
                updated=updated,
                published=published,
                categories=categories,
            )
        )
    return papers


def _subtract_workdays(start: datetime, days: int) -> datetime:
    current = start
    remaining = days
    while remaining > 0:
        current = current - timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _filter_by_days_back(papers: List[Paper], days_back: int) -> List[Paper]:
    if days_back <= 0:
        return papers
    cutoff = _subtract_workdays(datetime.now(timezone.utc), days_back)
    filtered: List[Paper] = []
    for p in papers:
        if p.updated >= cutoff or p.published >= cutoff:
            filtered.append(p)
    return filtered


def _load_profile(config_profile: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    profile_path = config_profile.get("profile_path", "profiles/group_profile.json")
    path = Path(profile_path)
    if path.exists():
        data = _load_json(path)
        if isinstance(data, dict):
            return data, path
    return {}, path


def _run_date_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _write_run_artifact(
    out_dir: Path,
    run_date: str,
    config_path: Path,
    profile_path: Path,
    query: str,
    papers: List[Paper],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_date}.json"
    payload = {
        "run_date": run_date,
        "source": "arxiv",
        "config_path": str(config_path),
        "profile_path": str(profile_path),
        "query": query,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "papers": [
            {
                "arxiv_id": p.arxiv_id,
                "title": _strip_markup(p.title),
                "authors": p.authors,
                "abstract": p.summary,
                "link": p.link,
                "published": p.published.isoformat(),
                "updated": p.updated.isoformat(),
                "categories": p.categories,
            }
            for p in papers
        ],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _compact_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    prefs = profile.get("preferences", {}) if isinstance(profile, dict) else {}

    def _cap(lst: List[str], n: int) -> List[str]:
        return lst[:n]

    return {
        "group_name": profile.get("group_name", "Group"),
        "keywords": _cap(profile.get("keywords", []) or [], 30),
        "bigrams": _cap(profile.get("bigrams", []) or [], 30),
        "trigrams": _cap(profile.get("trigrams", []) or [], 30),
        "negative_keywords": _cap(profile.get("negative_keywords", []) or [], 20),
        "preferences": {
            "favorite_authors": _cap(prefs.get("favorite_authors", []) or [], 20),
            "avoid_authors": _cap(prefs.get("avoid_authors", []) or [], 20),
            "favorite_categories": _cap(prefs.get("favorite_categories", []) or [], 20),
            "avoid_categories": _cap(prefs.get("avoid_categories", []) or [], 20),
            "interest_tags": _cap(prefs.get("interest_tags", []) or [], 20),
            "avoid_tags": _cap(prefs.get("avoid_tags", []) or [], 20),
        },
    }


def _paper_card(p: Paper, snippet_chars: int) -> Dict[str, Any]:
    return {
        "arxiv_id": p.arxiv_id,
        "title": _strip_markup(p.title),
        "categories": p.categories[:6],
        "abstract_snippet": _strip_markup(p.summary)[:snippet_chars],
    }


def _local_pre_score(p: Paper, profile: Dict[str, Any]) -> int:
    text = _clean_for_extraction(f"{p.title} {p.summary}").lower()
    categories = {c.lower() for c in p.categories}
    score = 0
    for kw in profile.get("keywords", []) or []:
        if kw and kw.lower() in text:
            score += 2
    for bg in profile.get("bigrams", []) or []:
        if bg and bg.lower() in text:
            score += 3
    for tg in profile.get("trigrams", []) or []:
        if tg and tg.lower() in text:
            score += 3
    for cat in profile.get("preferences", {}).get("favorite_categories", []) or []:
        if cat.lower() in categories:
            score += 1
    return score


def _select_candidates(papers: List[Paper], profile: Dict[str, Any], max_candidates: int) -> List[Paper]:
    if max_candidates <= 0 or len(papers) <= max_candidates:
        return papers
    scored = [(_local_pre_score(p, profile), p) for p in papers]
    scored.sort(key=lambda sp: (-sp[0], sp[1].updated))
    return [p for _, p in scored[:max_candidates]]


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_prompt(
    template: str,
    group_json: str,
    papers_json: str,
    formula: str,
    w_read: float,
    w_dist: float,
    w_insp: float,
    w_act: float,
    n_select: int,
) -> str:
    return (
        template.replace("{{GROUP_PROFILE_JSON}}", group_json)
        .replace("{{PAPERS_JSON}}", papers_json)
        .replace("{{INSPIRING_FORMULA}}", formula)
        .replace("{{W_READ}}", f"{w_read}")
        .replace("{{W_DIST}}", f"{w_dist}")
        .replace("{{W_INSP}}", f"{w_insp}")
        .replace("{{W_ACT}}", f"{w_act}")
        .replace("{{N_SELECT}}", str(n_select))
    )


def _gemini_generate(api_key: str, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_output_tokens},
    }
    try:
        resp = requests.post(url, json=body, timeout=180)
    except requests.RequestException as e:
        raise RuntimeError(f"Gemini request failed: {e}")
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error: HTTP {resp.status_code} - {resp.text[:400]}")
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini response missing candidates")
    finish_reason = candidates[0].get("finishReason", "")
    if finish_reason == "MAX_TOKENS":
        print(f"WARNING: Gemini output truncated due to MAX_TOKENS. Consider increasing max_output_tokens or reducing input size.", file=sys.stderr)
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError("Gemini response missing content parts")
    return parts[0].get("text", "")


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_gemini_json(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if "```" in raw:
        segments = raw.split("```")
        if len(segments) >= 2:
            raw = segments[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
    candidate = raw if raw.startswith("{") else _extract_json_object(raw)
    if not candidate:
        candidate = _extract_json_object(text)
    if not candidate or not candidate.rstrip().endswith("}"):
        tail = text[-300:]
        raise RuntimeError(
            "Likely truncated Gemini output (missing closing brace). "
            "Reduce output size or increase max_output_tokens.\n" + tail
        )
    try:
        return json.loads(candidate)
    except Exception:
        tail = text[-300:]
        raise RuntimeError(f"Gemini returned invalid JSON. Tail:\n{tail}")


def _gemini_json_with_retry(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    max_attempts: int = 1,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        prompt = user_prompt
        if attempt > 1:
            prompt = (
                user_prompt
                + "\n\nRETRY: Output STRICT JSON only. Ensure all strings are closed and JSON is valid."
            )
        text = _gemini_generate(api_key, model, system_prompt, prompt, max_output_tokens)
        try:
            return _parse_gemini_json(text)
        except RuntimeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Gemini JSON parse failed")


def _validate_relevant(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise RuntimeError("Relevant JSON must be an object")
    if "highly_relevant" not in data:
        raise RuntimeError("Relevant JSON missing required key")


def _validate_inspiring(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise RuntimeError("Inspiring JSON must be an object")
    if "inspiring" not in data:
        raise RuntimeError("Inspiring JSON missing required key")


def _enrich_paper(p: Optional[Paper]) -> Dict[str, Any]:
    if not p:
        return {}
    return {
        "arxiv_id": p.arxiv_id,
        "title": _strip_markup(p.title),
        "authors": p.authors,
        "abstract": p.summary,
        "link": p.link,
        "published": p.published.isoformat(),
        "updated": p.updated.isoformat(),
        "categories": p.categories,
    }


def _write_ranked_output(
    out_path: Path,
    run_meta: Dict[str, Any],
    items: List[Dict[str, Any]],
) -> None:
    payload = dict(run_meta)
    payload["papers"] = items
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily arXiv digest")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    arxiv_cfg = config.get("arxiv", {})
    categories = arxiv_cfg.get("categories", []) or []
    extra_query = arxiv_cfg.get("extra_query", "") or ""
    days_back = int(arxiv_cfg.get("days_back", 2))
    max_results = int(arxiv_cfg.get("max_results", 120))

    profile_cfg = config.get("profile", {})
    profile, profile_path = _load_profile(profile_cfg)

    query = _build_search_query(categories, extra_query)

    try:
        papers = _fetch_arxiv_papers(categories, extra_query, max_results)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    papers = _filter_by_days_back(papers, days_back)

    run_date = _run_date_utc()
    output_cfg = config.get("output", {})
    write_raw = bool(output_cfg.get("write_raw_run_artifact", False))
    raw_out_path: Optional[Path] = None
    if write_raw:
        raw_out_path = _write_run_artifact(
            Path("artifacts/run"),
            run_date,
            config_path,
            profile_path,
            query,
            papers,
        )

    gemini_cfg = config.get("gemini", {})
    model = gemini_cfg.get("model", "gemini-2.5-flash")
    max_output_tokens = int(gemini_cfg.get("max_output_tokens", 1500))
    max_attempts = int(gemini_cfg.get("max_attempts", 1))
    snippet_chars = int(output_cfg.get("abstract_snippet_chars", 240))
    max_candidates = int(output_cfg.get("max_candidates_to_send_to_gemini", 40))
    n_highly = min(int(output_cfg.get("n_highly_relevant", 3)), 3)
    n_inspiring = min(int(output_cfg.get("n_inspiring", 3)), 3)

    weights = output_cfg.get("inspiring_weights", {}) or {}
    w_read = float(weights.get("readability", 0.25))
    w_dist = float(weights.get("distance", 0.20))
    w_insp = float(weights.get("inspiration", 0.45))
    w_act = float(weights.get("actionability", 0.20))
    w_nov = float(weights.get("novelty", 0.0))
    # NOTE: Prompt defines Q1..Q5; keep formula in that notation so Gemini computes scores consistently.
    formula = f"{w_insp}*Q3 + {w_read}*Q1 + {w_act}*Q4 - {w_dist}*Q2 + {w_nov}*Q5"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is required", file=sys.stderr)
        return 1

    compact_profile = _compact_profile(profile)
    candidates = _select_candidates(papers, compact_profile, max_candidates)

    paper_cards = [_paper_card(p, snippet_chars) for p in candidates]

    relevant_system = _load_prompt(Path("prompts/relevant/system.txt"))
    relevant_user = _load_prompt(Path("prompts/relevant/user.txt"))
    inspiring_system = _load_prompt(Path("prompts/inspiring/system.txt"))
    inspiring_user = _load_prompt(Path("prompts/inspiring/user.txt"))

    group_json = json.dumps(compact_profile, ensure_ascii=False)
    papers_json = json.dumps(paper_cards, ensure_ascii=False)

    relevant_prompt = _render_prompt(
        relevant_user, group_json, papers_json, formula, w_read, w_dist, w_insp, w_act, n_highly
    )
    relevant_data = _gemini_json_with_retry(
        api_key, model, relevant_system, relevant_prompt, max_output_tokens, max_attempts
    )
    _validate_relevant(relevant_data)

    highly = (relevant_data.get("highly_relevant", []) or [])[:n_highly]
    selected_ids = {item.get("arxiv_id") for item in highly if item.get("arxiv_id")}

    remaining_cards = [c for c in paper_cards if c.get("arxiv_id") not in selected_ids]
    inspiring_prompt = _render_prompt(
        inspiring_user,
        group_json,
        json.dumps(remaining_cards, ensure_ascii=False),
        formula,
        w_read,
        w_dist,
        w_insp,
        w_act,
        n_inspiring,
    )
    inspiring_data = _gemini_json_with_retry(
        api_key, model, inspiring_system, inspiring_prompt, max_output_tokens, max_attempts
    )
    _validate_inspiring(inspiring_data)

    inspiring = (inspiring_data.get("inspiring", []) or [])[:n_inspiring]

    papers_by_id = {p.arxiv_id: p for p in candidates}

    def _attach_relevant(item: Dict[str, Any]) -> Dict[str, Any]:
        arxiv_id = item.get("arxiv_id", "")
        p = papers_by_id.get(arxiv_id)
        base = _enrich_paper(p) if p else {"arxiv_id": arxiv_id}
        out: Dict[str, Any] = {
            **base,
            "relevance_score": item.get("relevance_score"),
            "why": _normalize_generated_text(item.get("why", "")),
        }

        must_read = bool(item.get("must_read", False))
        if must_read:
            out["must_read"] = True
            tags = item.get("tags")
            if isinstance(tags, list) and tags:
                out["tags"] = tags[:4]

        return out

    def _attach_inspiring(item: Dict[str, Any]) -> Dict[str, Any]:
        arxiv_id = item.get("arxiv_id", "")
        p = papers_by_id.get(arxiv_id)
        base = _enrich_paper(p) if p else {"arxiv_id": arxiv_id}
        return {
            **base,
            "readability": item.get("readability"),
            "distance": item.get("distance"),
            "inspiration": item.get("inspiration"),
            "actionability": item.get("actionability"),
            "novelty": item.get("novelty"),
            "inspiring_score": item.get("inspiring_score"),
            "why": _normalize_generated_text(item.get("why", "")),
        }

    highly_out = [_attach_relevant(item) for item in highly]
    inspiring_out = [_attach_inspiring(item) for item in inspiring]

    run_meta = {
        "run_date": run_date,
        "source": "arxiv",
        "config_path": str(config_path),
        "profile_path": str(profile_path),
        "query": query,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "selected_count": len(highly_out),
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }

    base_dir = Path("artifacts/run")
    relevant_path = base_dir / f"{run_date}_highly_relevant.json"
    inspiring_path = base_dir / f"{run_date}_inspiring.json"

    _write_ranked_output(relevant_path, run_meta, highly_out)
    _write_ranked_output(inspiring_path, run_meta, inspiring_out)

    inspiring_meta = dict(run_meta)
    inspiring_meta["selected_count"] = len(inspiring_out)

    _write_ranked_output(inspiring_path, inspiring_meta, inspiring_out)

    wrote_raw = str(raw_out_path) if raw_out_path else "(raw disabled)"
    print(
        f"Fetched {len(papers)} papers; candidates {len(candidates)}. "
        f"Wrote {wrote_raw}, {relevant_path}, {inspiring_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
