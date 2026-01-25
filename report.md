# System Workflow Report (v1)

  ## 1. System Overview

  This system is a script-only pipeline that produces a daily arXiv digest
  ranked for a research group. It has two primary workflows: building a group
  profile from INSPIRE-HEP and generating a daily digest from arXiv ranked by
  Gemini. It runs offline as command-line scripts and writes artifacts to local
  files. It does not provide a service, UI, database, scheduler, or delivery
  mechanism (e.g., email/Telegram). It is explicitly non-product: no persistence
  beyond local JSON/Markdown, no authentication flows, and no monitoring.

  ## 2. Data Sources and Trust Boundaries

  The system ingests data from three external APIs and uses local files as
  state:

  - INSPIRE-HEP API: Trusted metadata source for literature records. Used to
    construct group profile features (keywords/bigrams). Data is read-only and
    not modified.
  - arXiv API: Trusted metadata source for daily submissions. Used to assemble
    candidate papers for ranking.
  - Gemini API: External ranking engine. Treated as untrusted output; strict
    JSON parsing is enforced.
  - Local cache files:
      - cache/seen.json tracks processed arXiv IDs and is the only persistent
        state for deduplication.
      - cache/last_digest.md stores the latest digest output.
  - Environment variables: GEMINI_API_KEY is required to call Gemini. The
    scripts do not store or log the key.

  Trust boundary summary: INSPIRE and arXiv are trusted data providers, Gemini
  is a probabilistic ranking system with output validation, and local cache
  files are treated as authoritative state when present.

  ## 3. Workflow A: Profile Construction (build_profile.py)

  Inputs:

  - config/config.yaml
  - inspire.query string (author/collaboration query)
  - inspire.pages, inspire.max_records, optional inspire.page_size

  Collection:
  The script paginates the INSPIRE literature endpoint using page= and sort=.
  Each page is fetched with exponential backoff (up to 3 attempts). Pagination
  stops when either the max_records limit is reached or the API returns no hits.

  Record Parsing (best-effort, missing fields tolerated):

  - Title: preferred_titles[0].title preferred; fallback titles[0].title
  - Abstract: abstracts[0].value
  - Keywords: keywords[].value
  - arXiv eprint: arxiv_eprints[0].value
  - Year: derived from imprints/publication/preprint date fields, using earliest
    year when multiple are found

  Deterministic extraction logic:

  - Tokenization: regex [a-zA-Z][a-zA-Z0-9_+-]{2,}
  - Lowercasing, stopword removal, digit-only removal
  - Token counts are aggregated from both keywords and title+abstract text
  - Very frequent tokens are downweighted by a heuristic: tokens with frequency
    ≥ 35% of the max frequency are dropped
  - Keywords list: top top_k_keywords tokens by frequency
  - Bigrams: adjacent token pairs from titles only, filtered by minimum
    frequency min_bigram_count, ranked by frequency, capped at top_k_bigrams

  Output artifact:
  profiles/group_profile.json with schema:

  {
    "group_name": "...",
    "source": { "kind": "inspirehep", "query": "...", "fetched_records": N,
  "fetched_at": "ISO-8601" },
    "keywords": [...],
    "bigrams": [...],
    "negative_keywords": [...],
    "seed_papers": [ { "title": "...", "year": 2024, "arxiv": "2401.12345",
  "inspire_id": "..." } ]
  }

  Intentionally discarded:

  - Full author lists
  - Journal venue details
  - All fields except those required to derive keywords and sanity-check seed
    papers
  - Full record bodies (only the JSON summary is retained)

  ## 4. Workflow B: Daily Digest (digest_arxiv.py)

  Inputs:

  - config/config.yaml
  - Group profile: profiles/group_profile.json if present; otherwise use
    keywords from YAML

  arXiv query construction:

  - categories are mapped to cat:... and joined with OR
  - Optional extra_query is AND’d with categories
  - Query runs against http://export.arxiv.org/api/query sorted by
    lastUpdatedDate desc

  Time filtering semantics:
  days_back is applied against both updated and published. A paper is included
  if either timestamp is within the cutoff (UTC now minus days_back).

  Deduplication:
  cache/seen.json stores processed arXiv IDs. By default, papers already present
  are excluded. include_seen=true bypasses this filter.

  Gemini call conditions:

  - If no papers remain after filtering/deduplication, Gemini is not called.
  - If GEMINI_API_KEY is missing, the script exits nonzero without calling
    Gemini.

  Gemini prompt contract:
  The model receives the group profile and the list of candidate papers (title,
  abstract, categories, authors, link). Output must be strict JSON with fields:

  {
    "core":[{"id":"...", "score":0-100, "why":"<=2 sentences", "tags":[...]}],
    "adjacent":[...],
    "skipped":[{"id":"...", "reason":"..."}]
  }

  The script enforces JSON parsing and truncates core/adjacent lists to n_core
  and n_adjacent. If the response is malformed, execution fails with a clear
  error.

  Core vs Adjacent semantics:

  - core: papers highly aligned with group profile
  - adjacent: papers tangentially related but potentially informative
    The classification is determined solely by Gemini output; the script does
    not impose additional semantic scoring.

  Markdown digest construction:
  The digest includes a UTC date header, two sections (Core and Adjacent), and
  for each item: title/link, arXiv ID, updated/published dates, up to six
  authors, tags, score, and short rationale. The digest is printed to stdout and
  written to cache/last_digest.md.

  ## 5. Caching and Reproducibility

  Cached artifacts:

  - cache/seen.json: set of arXiv IDs seen across runs; persists dedup state
  - cache/last_digest.md: last generated digest

  Not cached:

  - Gemini responses (no audit trail)
  - Raw arXiv feeds or INSPIRE responses

  Determinism limits:

  - arXiv feed content changes over time
  - INSPIRE records update and the pagination order can drift
  - Gemini output is nondeterministic, though temperature is low
  - Results depend on the current time (days_back filter) and the contents of
    seen.json

  ## 6. Failure Modes and Expected Behavior

  - Missing GEMINI_API_KEY: digest script exits with nonzero status and an error
    message; no API call is made.
  - arXiv network failure: digest script exits nonzero with an error; no cache
    updates occur.
  - INSPIRE API partial failure: build script retries per page; after repeated
    failures it exits nonzero, producing no profile output.
  - Gemini malformed output: digest script fails with a JSON error; prints
    response head for debugging and exits nonzero.
  - Empty daily results: digest script prints a short message and exits 0
    without calling Gemini.
  - I/O errors: propagate as runtime errors; no explicit recovery beyond error
    reporting.

  Exit status is 0 on successful completion or empty result; nonzero on any
  error.

  ## 7. Extensibility Boundaries

  Explicitly out of scope for v1: notification channels (email/Telegram),
  databases, UIs, schedulers/cron, and long-term storage. The script boundary is
  intentional to keep the system local, auditable, and easy to reason about.

  Clean extension points:

  - Output sinks: add new consumers of the generated markdown without altering
    arXiv or Gemini logic.
  - Profile enrichers: replace or augment keyword extraction with additional
    sources, provided the output schema is preserved.
  - Ranking hooks: swap Gemini with another ranking service by honoring the same
    JSON contract.

  The separation into two scripts is preserved to decouple slow, infrequent
  profile construction from daily digest generation.

  ## 8. Minimal ASCII Architecture Diagram

                  +------------------------+
                  |   config/config.yaml   |
                  +-----------+------------+
                              |
                              v
    +------------------+   INSPIRE API   +----------------------+
    | build_profile.py | <-------------> | inspirehep.net (JSON) |
    +---------+--------+                  +----------------------+
              |
              v
    profiles/group_profile.json
              |
              | (optional override)
              v
    +------------------+   arXiv API     +-----------------------+
    | digest_arxiv.py  | <-------------> | export.arxiv.org (Atom)|
    +----+---------+---+                  +-----------------------+
         |         |
         |         +----------------------+
         |                                |
         v                                v
  cache/seen.json                  Gemini API (JSON)
         |                                |
         v                                v
  cache/last_digest.md               stdout digest

  This describes the current v1 data flow and boundaries without implying
  persistence or service orchestration.