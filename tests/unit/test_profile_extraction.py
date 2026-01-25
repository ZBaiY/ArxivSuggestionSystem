# filename: tests/unit/test_profile_extraction.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import build_profile as bp  # noqa: E402


def test_clean_text_strips_math_and_latex():
    text = (
        "We study $E=mc^2$ and \\begin{equation}x\\end{equation} "
        "plus \\mathrm{SU}(2) with \\(a+b\\) and \\[c+d\\]. "
        "<div>junk</div>"
    )
    cleaned = bp._clean_text(text)
    assert "$" not in cleaned
    assert "begin" not in cleaned
    assert "mathrm" not in cleaned
    assert "<" not in cleaned and ">" not in cleaned


def test_keywords_phrase_first_contains_spaces_when_available():
    keywords = bp._compose_keywords(
        structured_phrases=["dark matter"],
        trigrams=["cosmic microwave background"],
        bigrams=["axion physics"],
        unigrams=["axion", "neutrino"],
        top_k=5,
    )
    assert any(" " in k for k in keywords)


def test_blacklist_removes_display_inline_bigram():
    records = [
        {"title": "Display inline math in title", "abstract": "", "keywords": [], "year": 2024},
        {"title": "Another display inline example", "abstract": "", "keywords": [], "year": 2023},
    ]
    tf_bi, df_bi, doc_count = bp._count_ngrams(records, 2, tau_years=4.0, use_abstracts=False)
    filtered = bp._filter_ngrams(df_bi, doc_count, min_count=1, df_ratio_max=1.0)
    bigrams = [b for b, _ in filtered]
    assert "display inline" not in bigrams


def test_apply_year_filter_hard():
    records = [
        {"title": "old", "year": 2010},
        {"title": "new", "year": 2022},
        {"title": "missing", "year": None},
    ]
    kept = bp._apply_year_filter(records, min_year=2015, hard=True)
    titles = [r["title"] for r in kept]
    assert "old" not in titles
    assert "new" in titles
    assert "missing" in titles


def test_normalize_punct_fullwidth():
    text = "dark matter：a test，ok（x）"
    assert bp._normalize_punct(text) == "dark matter: a test, ok(x)"


def test_strip_markup_removes_mathml_in_title():
    inp = 'Clock <math display="inline"><mrow><mi>Th</mi><mn>229</mn></mrow></math> proposal'
    out = bp._strip_markup(inp)
    assert out == "Clock proposal"


def test_strip_markup_keeps_fullwidth_colon():
    inp = 'A：Title <math display="inline">x</math>'
    out = bp._strip_markup(inp)
    assert out.startswith("A：Title")
    assert "<math" not in out


def test_clean_for_extraction_removes_latex_and_tags():
    inp = "Test $x+y$ and \\\\mathrm{SU}(2) <b>tag</b>"
    out = bp._clean_for_extraction(inp)
    assert "$" not in out
    assert "mathrm" not in out
    assert "<" not in out and ">" not in out
