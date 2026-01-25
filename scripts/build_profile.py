# filename: scripts/build_profile.py
#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import yaml

INSPIRE_API_URL = "https://inspirehep.net/api/literature"

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "show",
    "also",
    "study",
    "case",
    "data",
    "model",
    "models",
    "results",
    "analysis",
    "numerical",
    "calculation",
    "calculations",
    "current",
    "contribution",
    "contributions",
    "sensitivity",
    "experiment",
    "experiments",
    "theory",
    "field",
    "interaction",
    "interactions",
    "energy",
    "scale",
    "large",
    "light",
    "bounds",
    "limit",
    "limits",
    "constraint",
    "constraints",
    "display",
    "inline",
    "math",
    "coll",
    "template",
    "method",
    "methods",
    "search",
    "searches",
    "framework",
    "will",
    "strong",
    "future",
    "background",
    "generation",
    "particles",
    "breaking",
    "decay",
    "decays",
    "resonance",
    "signal",
    "signals",
}

PHRASE_BLACKLIST = {
    "display inline",
    "math display",
    "display math",
    "inline math",
}

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


def _strip_markup(s: str) -> str:
    if not s:
        return ""
    text = re.sub(r"<math[^>]*>.*?</math>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_punct(s: str) -> str:
    if not s:
        return ""
    return s.translate(FULLWIDTH_MAP)


def _strip_punct(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^\w\s]", "", s)


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


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


def _split_name(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        last = parts[0] if parts else None
        first = parts[1] if len(parts) > 1 else None
        return first, None, last
    tokens = _normalize_name(name).split()
    if not tokens:
        return None, None, None
    if len(tokens) == 1:
        return tokens[0], None, None
    first = tokens[0]
    last = tokens[-1]
    middle = " ".join(tokens[1:-1]) if len(tokens) > 2 else None
    return first, middle, last


def _name_variants(name: str) -> List[str]:
    first, middle, last = _split_name(name)
    variants = []
    if first and last:
        variants.append(f"{first} {last}")
        variants.append(f"{last}, {first}")
        variants.append(f"{first[0]}. {last}")
        if middle:
            variants.append(f"{first} {middle[0]}. {last}")
    elif first:
        variants.append(first)
    return list(dict.fromkeys([_normalize_name(v) for v in variants if v]))


def _generate_author_queries(query: str) -> List[str]:
    q = query.strip()
    if " AND " in q or " OR " in q:
        return [q]
    m = re.fullmatch(r'(a|author)\s*:\s*"(.+)"', q)
    if m:
        name = m.group(2)
    else:
        m = re.fullmatch(r'a\s*"(.+)"', q)
        if m:
            name = m.group(1)
        else:
            m = re.fullmatch(r'"(.+)"', q)
            if m:
                name = m.group(1)
            else:
                if ":" in q:
                    return [q]
                name = q
    variants = _name_variants(name)
    if not variants:
        return [q]
    expanded = [q]
    for v in variants:
        expanded.append(f'a "{v}"')
        expanded.append(f'author:"{v}"')
    return list(dict.fromkeys(expanded))


def _request_with_retry(params: Dict[str, Any], max_attempts: int = 3) -> Dict[str, Any]:
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(INSPIRE_API_URL, params=params, timeout=30)
        except requests.RequestException as e:
            if attempt == max_attempts:
                raise RuntimeError(f"INSPIRE request failed: {e}")
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 200:
            return resp.json()
        if attempt == max_attempts:
            raise RuntimeError(f"INSPIRE API error: HTTP {resp.status_code} - {resp.text[:400]}")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("INSPIRE request failed after retries")


def _extract_year(meta: Dict[str, Any]) -> Optional[int]:
    years: List[int] = []
    for imprint in meta.get("imprints", []) or []:
        date = imprint.get("date")
        if date:
            for match in YEAR_RE.findall(str(date)):
                years.append(int(match))
    for pub in meta.get("publication_info", []) or []:
        year = pub.get("year")
        if year:
            try:
                years.append(int(year))
            except ValueError:
                pass
    preprint_date = meta.get("preprint_date")
    if preprint_date:
        for match in YEAR_RE.findall(str(preprint_date)):
            years.append(int(match))
    if not years:
        return None
    return min(years)


def _extract_title(meta: Dict[str, Any]) -> str:
    pref = meta.get("preferred_titles") or []
    if pref:
        title = pref[0].get("title")
        if title:
            return title
    titles = meta.get("titles") or []
    if titles:
        title = titles[0].get("title")
        if title:
            return title
    return ""


def _extract_abstract(meta: Dict[str, Any]) -> str:
    abstracts = meta.get("abstracts") or []
    if abstracts:
        val = abstracts[0].get("value")
        if val:
            return val
    return ""


def _extract_keywords(meta: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for kw in meta.get("keywords", []) or []:
        val = kw.get("value")
        if val:
            out.append(val)
    return out


def _extract_arxiv(meta: Dict[str, Any]) -> Optional[str]:
    eprints = meta.get("arxiv_eprints", []) or []
    if eprints:
        val = eprints[0].get("value")
        if val:
            return val
    return None


def _tokenize(text: str) -> List[str]:
    cleaned = _clean_for_extraction(text).lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+\-]{2,}", cleaned)
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]


def _extract_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    grams = []
    for i in range(len(tokens) - n + 1):
        grams.append(" ".join(tokens[i : i + n]))
    return grams


def _phrase_tokens(phrase: str) -> List[str]:
    return _tokenize(phrase)


def _is_generic_phrase(phrase: str) -> bool:
    if not phrase:
        return True
    if phrase in PHRASE_BLACKLIST:
        return True
    tokens = _phrase_tokens(phrase)
    if not tokens:
        return True
    if all(t in STOPWORDS for t in tokens):
        return True
    return False


def _doc_weight(year: Optional[int], now_year: int, tau_years: float) -> float:
    if not year:
        return 1.0
    age = max(0, now_year - year)
    return math.exp(-age / tau_years)


def _apply_year_filter(
    records: List[Dict[str, Any]],
    min_year: int,
    hard: bool,
) -> List[Dict[str, Any]]:
    if not hard:
        return records
    out = []
    for r in records:
        year = r.get("year")
        if year is None or year >= min_year:
            out.append(r)
    return out


def _count_unigrams(records: List[Dict[str, Any]], tau_years: float) -> Tuple[Counter, Counter, int]:
    tf: Counter = Counter()
    df: Counter = Counter()
    doc_count = 0
    now_year = datetime.now(timezone.utc).year
    for rec in records:
        w = _doc_weight(rec.get("year"), now_year, tau_years)
        doc_tokens: List[str] = []
        for kw in rec.get("keywords", []) or []:
            doc_tokens.extend(_tokenize(kw))
        text = f"{rec.get('title', '')} {rec.get('abstract', '')}"
        doc_tokens.extend(_tokenize(text))
        if not doc_tokens:
            continue
        doc_count += 1
        counts = Counter(doc_tokens)
        for token, c in counts.items():
            tf[token] += w * c
        for token in counts.keys():
            df[token] += 1
    return tf, df, doc_count


def _count_phrases_from_keywords(
    records: List[Dict[str, Any]],
    tau_years: float,
) -> Tuple[Counter, Counter]:
    tf: Counter = Counter()
    df: Counter = Counter()
    now_year = datetime.now(timezone.utc).year
    for rec in records:
        w = _doc_weight(rec.get("year"), now_year, tau_years)
        phrases = set()
        for kw in rec.get("keywords", []) or []:
            cleaned = _normalize_punct(_clean_for_extraction(kw)).lower().strip()
            cleaned = _strip_punct(cleaned).strip()
            if not cleaned or _is_generic_phrase(cleaned):
                continue
            phrases.add(cleaned)
        for ph in phrases:
            tf[ph] += w
            df[ph] += 1
    return tf, df


def _count_ngrams(
    records: List[Dict[str, Any]],
    n: int,
    tau_years: float,
    use_abstracts: bool,
) -> Tuple[Counter, Counter, int]:
    tf: Counter = Counter()
    df: Counter = Counter()
    doc_count = 0
    now_year = datetime.now(timezone.utc).year
    for rec in records:
        texts = [rec.get("title", "")]
        if use_abstracts and rec.get("abstract"):
            texts.append(rec.get("abstract", ""))
        grams_in_doc: Counter = Counter()
        for text in texts:
            tokens = _tokenize(text)
            grams = _extract_ngrams(tokens, n)
            for g in grams:
                if g in PHRASE_BLACKLIST:
                    continue
                grams_in_doc[g] += 1
        if not grams_in_doc:
            continue
        doc_count += 1
        w = _doc_weight(rec.get("year"), now_year, tau_years)
        for g, c in grams_in_doc.items():
            tf[g] += w * c
        for g in grams_in_doc.keys():
            df[g] += 1
    return tf, df, doc_count


def _score_keywords(
    tf: Counter,
    df: Counter,
    doc_count: int,
    df_ratio_max: float,
    df_min: int,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if doc_count <= 0:
        return scores
    for token, tf_val in tf.items():
        df_val = df.get(token, 0)
        if df_val < df_min:
            continue
        df_ratio = df_val / doc_count
        if df_ratio > df_ratio_max:
            continue
        score = tf_val * math.log((doc_count + 1) / (df_val + 1))
        scores[token] = score
    return scores


def _filter_ngrams(
    df: Counter,
    doc_count: int,
    min_count: int,
    df_ratio_max: float,
) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for g, df_val in df.items():
        if df_val < min_count:
            continue
        if doc_count > 0 and (df_val / doc_count) > df_ratio_max:
            continue
        if g in PHRASE_BLACKLIST:
            continue
        out.append((g, df_val))
    return out


def _rank_phrases_by_df(items: List[Tuple[str, int]]) -> List[str]:
    items.sort(key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    return [k for k, _ in items]


def _contrast_scores(
    tf_group: Counter,
    tf_bg: Counter,
    alpha: float,
) -> Dict[str, float]:
    vocab = set(tf_group.keys()) | set(tf_bg.keys())
    V = max(1, len(vocab))
    tf_total_group = sum(tf_group.values())
    tf_total_bg = sum(tf_bg.values())
    scores: Dict[str, float] = {}
    for term in vocab:
        g = tf_group.get(term, 0.0)
        b = tf_bg.get(term, 0.0)
        score = math.log((g + alpha) / (tf_total_group + alpha * V)) - math.log(
            (b + alpha) / (tf_total_bg + alpha * V)
        )
        scores[term] = score
    return scores


def _top_by_score(scores: Dict[str, float], top_k: int) -> List[str]:
    items = list(scores.items())
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items[:top_k]]


def _dedup_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _compose_keywords(
    structured_phrases: List[str],
    trigrams: List[str],
    bigrams: List[str],
    unigrams: List[str],
    top_k: int,
) -> List[str]:
    combined = _dedup_preserve(structured_phrases + trigrams + bigrams + unigrams)
    return combined[:top_k]


def _normalize_list(items: List[str], strip_punct: bool) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _normalize_punct(item)
        if strip_punct:
            text = _strip_punct(text)
        text = " ".join(text.split())
        if text:
            out.append(text)
    return out


def _prompt_yes_no(message: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        resp = input(message).strip().lower()
    except EOFError:
        return False
    return resp in {"y", "yes"}


def _fetch_inspire_records(
    query: str,
    sort: str,
    pages: int,
    page_size: Optional[int],
    max_records: int,
    allow_fuzzy: bool = True,
    min_year: Optional[int] = None,
    years_back_hard: bool = True,
) -> Tuple[List[Dict[str, Any]], str, int]:
    candidates = _generate_author_queries(query) if allow_fuzzy else [query]

    def _fetch_for_query(q: str) -> Tuple[List[Dict[str, Any]], int]:
        records: List[Dict[str, Any]] = []
        raw_hits_fetched = 0
        for page in range(1, pages + 1):
            params: Dict[str, Any] = {"q": q, "sort": sort, "page": page}
            if page_size:
                params["size"] = page_size
            data = _request_with_retry(params)
            hits = data.get("hits", {}).get("hits", []) or []
            if not hits:
                break
            raw_hits_fetched += len(hits)
            batch = _parse_records(hits)
            if min_year is not None:
                batch = _apply_year_filter(batch, min_year, years_back_hard)
            records.extend(batch)
            if len(records) >= max_records:
                return records[:max_records], raw_hits_fetched
        return records, raw_hits_fetched

    first_query = candidates[0] if candidates else query
    first_records, first_raw = _fetch_for_query(first_query)
    if first_records:
        return first_records, first_query, first_raw
    if allow_fuzzy and len(candidates) > 1:
        if not _prompt_yes_no(
            f'No results for query "{first_query}". Try fuzzy author variants? [y/N] '
        ):
            return [], first_query, first_raw
    total_raw = first_raw
    for q in candidates[1:]:
        records, raw = _fetch_for_query(q)
        total_raw += raw
        if records:
            return records, q, total_raw
    return [], first_query, total_raw


def _parse_records(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for hit in hits:
        meta = hit.get("metadata", {}) or {}
        title = _strip_markup(_extract_title(meta))
        abstract = _strip_markup(_extract_abstract(meta))
        keywords = [_strip_markup(k) for k in _extract_keywords(meta)]
        arxiv = _extract_arxiv(meta)
        year = _extract_year(meta)
        inspire_id = hit.get("id") or meta.get("control_number")
        records.append(
            {
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
                "arxiv": arxiv,
                "year": year,
                "inspire_id": inspire_id,
            }
        )
    return records


def _build_profile(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    profile_cfg = config.get("profile", {})
    inspire_cfg = config.get("inspire", {})

    group_name = profile_cfg.get("group_name", "Group")
    profile_path = Path(profile_cfg.get("profile_path", "profiles/group_profile.json"))
    top_k_keywords = int(profile_cfg.get("top_k_keywords", 40))
    top_k_bigrams = int(profile_cfg.get("top_k_bigrams", 25))
    min_bigram_count = int(profile_cfg.get("min_bigram_count", 3))
    keyword_df_ratio_max = float(profile_cfg.get("keyword_df_ratio_max", 0.10))
    keyword_df_min = int(profile_cfg.get("keyword_df_min", 2))
    bigram_df_ratio_max = float(profile_cfg.get("bigram_df_ratio_max", 0.25))
    min_trigram_count = int(profile_cfg.get("min_trigram_count", 2))
    trigram_df_ratio_max = float(profile_cfg.get("trigram_df_ratio_max", 0.20))
    trigram_use_abstracts = bool(profile_cfg.get("trigram_use_abstracts", False))
    recency_tau_years = float(profile_cfg.get("recency_tau_years", 4.0))
    years_back = int(profile_cfg.get("years_back", 10))
    years_back_hard = bool(profile_cfg.get("years_back_hard", True))
    negative_keywords = profile_cfg.get("negative_keywords", []) or []

    enable_contrast = bool(profile_cfg.get("enable_background_contrast", False))
    background_cfg = profile_cfg.get("background", {}) or {}
    contrast_alpha = float(profile_cfg.get("background_contrast_alpha", 0.1))

    query = inspire_cfg.get("query", "")
    sort = inspire_cfg.get("sort", "mostrecent")
    pages = int(inspire_cfg.get("pages", 10))
    page_size = inspire_cfg.get("page_size")
    if page_size is not None:
        page_size = int(page_size)
    max_records = int(inspire_cfg.get("max_records", 200))

    if not query:
        raise RuntimeError("Missing inspire.query in config")

    now_year = datetime.now(timezone.utc).year
    min_year = now_year - years_back

    records, used_query, raw_hits_fetched = _fetch_inspire_records(
        query,
        sort,
        pages,
        page_size,
        max_records,
        allow_fuzzy=True,
        min_year=min_year,
        years_back_hard=years_back_hard,
    )

    tf_uni, df_uni, doc_count = _count_unigrams(records, recency_tau_years)
    scores_uni = _score_keywords(tf_uni, df_uni, doc_count, keyword_df_ratio_max, keyword_df_min)

    tf_struct, df_struct = _count_phrases_from_keywords(records, recency_tau_years)

    tf_bi, df_bi, title_doc_count = _count_ngrams(
        records, 2, recency_tau_years, use_abstracts=False
    )
    tf_tri, df_tri, tri_doc_count = _count_ngrams(
        records, 3, recency_tau_years, use_abstracts=trigram_use_abstracts
    )

    bigram_items = _filter_ngrams(df_bi, title_doc_count, min_bigram_count, bigram_df_ratio_max)
    trigram_items = _filter_ngrams(df_tri, tri_doc_count, min_trigram_count, trigram_df_ratio_max)

    bigrams_ranked = _rank_phrases_by_df(bigram_items)[:top_k_bigrams]
    trigrams_ranked = _rank_phrases_by_df(trigram_items)

    structured_phrases_ranked = _rank_phrases_by_df(list(df_struct.items()))

    if enable_contrast and background_cfg.get("query"):
        bg_query = background_cfg.get("query")
        bg_sort = background_cfg.get("sort", "mostrecent")
        bg_pages = int(background_cfg.get("pages", 12))
        bg_max = int(background_cfg.get("max_records", 300))
        bg_years_back = int(background_cfg.get("years_back", years_back))
        bg_min_year = now_year - bg_years_back
        bg_records, _, _ = _fetch_inspire_records(
            bg_query,
            bg_sort,
            bg_pages,
            page_size,
            bg_max,
            allow_fuzzy=False,
            min_year=bg_min_year,
            years_back_hard=years_back_hard,
        )

        tf_uni_bg, _, _ = _count_unigrams(bg_records, recency_tau_years)
        tf_struct_bg, _ = _count_phrases_from_keywords(bg_records, recency_tau_years)
        tf_bi_bg, _, _ = _count_ngrams(bg_records, 2, recency_tau_years, use_abstracts=False)
        tf_tri_bg, _, _ = _count_ngrams(
            bg_records, 3, recency_tau_years, use_abstracts=trigram_use_abstracts
        )

        scores_uni_contrast = _contrast_scores(tf_uni, tf_uni_bg, contrast_alpha)
        scores_struct_contrast = _contrast_scores(tf_struct, tf_struct_bg, contrast_alpha)
        scores_bi_contrast = _contrast_scores(tf_bi, tf_bi_bg, contrast_alpha)
        scores_tri_contrast = _contrast_scores(tf_tri, tf_tri_bg, contrast_alpha)

        unigrams_ranked = _top_by_score(scores_uni_contrast, top_k_keywords)
        structured_phrases_ranked = _top_by_score(scores_struct_contrast, top_k_keywords)
        trigrams_ranked = _top_by_score(scores_tri_contrast, top_k_keywords)
        bigrams_ranked = _top_by_score(scores_bi_contrast, top_k_bigrams)
        scoring_mode = "contrast"
    else:
        unigrams_ranked = _top_by_score(scores_uni, top_k_keywords)
        scoring_mode = "tfidf"

    structured_phrases_ranked = _normalize_list(structured_phrases_ranked, strip_punct=True)
    trigrams_ranked = _normalize_list(trigrams_ranked, strip_punct=True)
    bigrams_ranked = _normalize_list(bigrams_ranked, strip_punct=True)
    unigrams_ranked = _normalize_list(unigrams_ranked, strip_punct=True)
    keywords = _compose_keywords(
        structured_phrases_ranked,
        trigrams_ranked,
        bigrams_ranked,
        unigrams_ranked,
        top_k_keywords,
    )

    seed_papers = []
    for rec in records[:10]:
        seed_papers.append(
            {
                "title": _strip_markup(rec.get("title", "")),
                "year": rec.get("year"),
                "arxiv": rec.get("arxiv"),
                "inspire_id": rec.get("inspire_id"),
            }
        )

    profile = {
        "group_name": group_name,
        "source": {
            "kind": "inspirehep",
            "query": used_query,
            "fetched_records": len(records),
            "raw_hits_fetched": raw_hits_fetched,
            "min_year": min_year,
            "years_back": years_back,
            "years_back_hard": years_back_hard,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "keywords": keywords,
        "bigrams": bigrams_ranked[:top_k_bigrams],
        "trigrams": trigrams_ranked[: int(profile_cfg.get("top_k_trigrams", 25))],
        "structured_phrases": structured_phrases_ranked[: int(profile_cfg.get("top_k_structured_phrases", 25))],
        "negative_keywords": _normalize_list(negative_keywords, strip_punct=False),
        "seed_papers": seed_papers,
        "scoring": {
            "mode": scoring_mode,
            "recency_tau_years": recency_tau_years,
            "contrast_enabled": enable_contrast,
        },
    }

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    return profile, records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build group profile from INSPIRE-HEP")
    parser.add_argument("--config", required=True, default="config/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    try:
        config = _load_yaml(Path(args.config))
        profile, _ = _build_profile(config)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    source = profile.get("source", {})
    print(f"Raw hits fetched: {source.get('raw_hits_fetched', 0)}")
    print(f"Kept records after year filter: {source.get('fetched_records', 0)}")
    print(f"Years back: {source.get('years_back', 'n/a')}")
    print(f"Min year: {source.get('min_year', 'n/a')}")
    print("Top keywords:")
    print(", ".join(profile.get("keywords", [])[:20]))
    print("Top bigrams:")
    print(", ".join(profile.get("bigrams", [])[:10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
