import pandas as pd
import re
import numpy as np

# ── Try importing rapidfuzz (Tier 1) ──────────────────────────────────────────
try:
    from rapidfuzz import process as _rf_process, fuzz as _rf_fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    print("⚠️  rapidfuzz not installed — Tier 1 fuzzy matching disabled, will use string fallback.")

# ── Try importing sentence-transformers (Tier 2) ──────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    print("⚠️  sentence-transformers not installed — Tier 2 semantic matching disabled.")

# ── Matching thresholds ────────────────────────────────────────────────────────
FUZZY_THRESHOLD       = 70   # WRatio score below this triggers Tier 2 (0–100)
SEMANTIC_THRESHOLD    = 0.35 # cosine similarity minimum to accept a semantic match

# ---------- Keyword rules ----------
HIGH_AUTO = [
    "compile", "collect", "gather", "retrieve", "search", "query",
    "analyze", "calculate", "compute", "process", "generate", "produce",
    "summarize", "report", "document", "record", "enter", "input",
    "monitor", "track", "review", "check", "verify", "sort", "classify"
]

LOW_AUTO = [
    "negotiate", "counsel", "advise", "mentor", "coach", "supervise",
    "coordinate", "collaborate", "communicate", "present", "persuade",
    "lead", "manage", "motivate", "inspire", "empathize",
    "design", "create", "innovate", "strategize", "judge", "decide"
]

# Defensive cleanup — upstream vars come from Block 2
_clean_occupations = clean_occupations.copy()
_clean_tasks = clean_tasks.copy()
_clean_wages = clean_wages.copy()

_clean_occupations["occupation_title"] = _clean_occupations["occupation_title"].astype(str).str.strip()
_clean_occupations["onet_code"] = _clean_occupations["onet_code"].astype(str).str.strip()

_clean_tasks["task_description"] = _clean_tasks["task_description"].astype(str).str.strip()
_clean_tasks["onet_code"] = _clean_tasks["onet_code"].astype(str).str.strip()

_clean_wages["noc_label"] = _clean_wages["noc_label"].astype(str).str.strip()
_clean_wages["avg_hourly_wage"] = pd.to_numeric(_clean_wages["avg_hourly_wage"], errors="coerce")

# ── Pre-compute occupation title embeddings (Tier 2 cache) ────────────────────
# Computed once at block load time; stored as module-level variables so
# downstream blocks (and the Streamlit script) can access them via variable().
_occ_title_list = _clean_occupations["occupation_title"].tolist()

if _ST_AVAILABLE:
    print("🔄  Computing occupation title embeddings with all-MiniLM-L6-v2 …")
    _embedding_model = _ST("all-MiniLM-L6-v2")
    occ_title_embeddings = _embedding_model.encode(_occ_title_list, normalize_embeddings=True, show_progress_bar=False)
    print(f"✅  Embeddings ready — shape: {occ_title_embeddings.shape}")
else:
    _embedding_model = None
    occ_title_embeddings = None
    print("ℹ️   Semantic embeddings skipped (sentence-transformers unavailable).")


# ── Wage title list + embeddings for Tier 2 wage matching ─────────────────────
_wage_title_list = _clean_wages["noc_label"].tolist()

if _ST_AVAILABLE:
    wage_title_embeddings = _embedding_model.encode(_wage_title_list, normalize_embeddings=True, show_progress_bar=False)
else:
    wage_title_embeddings = None


# ---------- Helpers ----------
def score_task(task: str):
    desc = str(task).lower()
    score = 0.5
    matched = []

    for kw in HIGH_AUTO:
        if kw in desc:
            score += 0.08
            matched.append(f"+{kw}")

    for kw in LOW_AUTO:
        if kw in desc:
            score -= 0.10
            matched.append(f"-{kw}")

    score = round(min(max(score, 0), 1), 2)
    return score, ", ".join(matched[:3])


def risk_level(avg_score):
    if avg_score is None:
        return "Unknown"
    if avg_score >= 0.70:
        return "High Risk"
    if avg_score >= 0.45:
        return "Medium Risk"
    return "Low Risk"


def find_occupation(job_title: str, fuzzy_threshold: int = FUZZY_THRESHOLD):
    """Two-tier fuzzy matching for occupation lookup.

    Returns (row_or_None, confidence_pct: int, match_method: str)

    Tier 1 — rapidfuzz WRatio: fast string similarity on all occupation titles.
              If best score >= fuzzy_threshold → accept match.
    Tier 2 — sentence-transformers cosine similarity: semantic search on
              pre-computed all-MiniLM-L6-v2 embeddings.
              Falls back to simple string contains if both tiers unavailable.
    """
    query = str(job_title).strip()
    if not query:
        return None, 0, "none"

    # ── Tier 0: exact / substring match (always attempted first, instant) ──
    mask = _clean_occupations["occupation_title"].str.contains(query, case=False, na=False, regex=False)
    exact_match = _clean_occupations[mask]
    if not exact_match.empty:
        return exact_match.iloc[0], 100, "exact"

    # ── Tier 1: rapidfuzz WRatio ───────────────────────────────────────────
    rf_score = 0
    if _RAPIDFUZZ_AVAILABLE:
        best = _rf_process.extractOne(query, _occ_title_list, scorer=_rf_fuzz.WRatio)
        if best is not None:
            rf_match_title, rf_score, rf_idx = best
            if rf_score >= fuzzy_threshold:
                row = _clean_occupations.iloc[rf_idx]
                return row, int(rf_score), "rapidfuzz"

    # ── Tier 2: sentence-transformers semantic search ──────────────────────
    if _ST_AVAILABLE and occ_title_embeddings is not None:
        query_vec = _embedding_model.encode([query], normalize_embeddings=True)
        sims = _cosine_sim(query_vec, occ_title_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= SEMANTIC_THRESHOLD:
            row = _clean_occupations.iloc[best_idx]
            confidence = int(round(best_sim * 100))
            return row, confidence, "semantic"

    # ── String word fallback (last resort) ────────────────────────────────
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
    for w in words:
        mask = _clean_occupations["occupation_title"].str.contains(w, case=False, na=False, regex=False)
        match = _clean_occupations[mask]
        if not match.empty:
            return match.iloc[0], 50, "word_fallback"

    return None, 0, "none"


def find_wage(occ_title: str, fuzzy_threshold: int = FUZZY_THRESHOLD):
    """Two-tier fuzzy matching for wage lookup.

    Returns (wage_or_None, label_or_None, confidence_pct: int, match_method: str)
    """
    lower = str(occ_title).lower().strip()
    if not lower:
        return None, None, 0, "none"

    # ── Tier 0: substring match ────────────────────────────────────────────
    mask = _clean_wages["noc_label"].str.contains(lower, case=False, na=False, regex=False)
    exact_match = _clean_wages[mask]
    if not exact_match.empty:
        row = exact_match.iloc[0]
        wage = pd.to_numeric(row["avg_hourly_wage"], errors="coerce")
        wage = None if pd.isna(wage) else float(wage)
        return wage, row["noc_label"], 100, "exact"

    # ── Tier 1: rapidfuzz WRatio ───────────────────────────────────────────
    rf_score = 0
    if _RAPIDFUZZ_AVAILABLE:
        best = _rf_process.extractOne(lower, _wage_title_list, scorer=_rf_fuzz.WRatio)
        if best is not None:
            _, rf_score, rf_idx = best
            if rf_score >= fuzzy_threshold:
                row = _clean_wages.iloc[rf_idx]
                wage = pd.to_numeric(row["avg_hourly_wage"], errors="coerce")
                wage = None if pd.isna(wage) else float(wage)
                return wage, row["noc_label"], int(rf_score), "rapidfuzz"

    # ── Tier 2: sentence-transformers semantic search ──────────────────────
    if _ST_AVAILABLE and wage_title_embeddings is not None:
        query_vec = _embedding_model.encode([lower], normalize_embeddings=True)
        sims = _cosine_sim(query_vec, wage_title_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= SEMANTIC_THRESHOLD:
            row = _clean_wages.iloc[best_idx]
            wage = pd.to_numeric(row["avg_hourly_wage"], errors="coerce")
            wage = None if pd.isna(wage) else float(wage)
            return wage, row["noc_label"], int(round(best_sim * 100)), "semantic"

    # ── Word fallback ──────────────────────────────────────────────────────
    words = [w for w in re.findall(r"\w+", lower) if len(w) > 3]
    for w in words:
        match = _clean_wages[_clean_wages["noc_label"].str.contains(w, case=False, na=False, regex=False)]
        if not match.empty:
            row = match.iloc[0]
            wage = pd.to_numeric(row["avg_hourly_wage"], errors="coerce")
            wage = None if pd.isna(wage) else float(wage)
            return wage, row["noc_label"], 50, "word_fallback"

    return None, None, 0, "none"


# ---------- Main analysis ----------
def analyze(job_title: str):
    query = str(job_title).strip()
    empty = pd.DataFrame(columns=["task", "score", "reason"])

    if not query:
        return {
            "found": False, "query": query,
            "occupation": None, "onet_code": None,
            "occ_confidence": 0, "occ_match_method": "none",
            "avg_score": None, "level": "Unknown",
            "hourly_cad": None, "wage_matched": None,
            "wage_confidence": 0, "wage_match_method": "none",
            "monthly_saving": None, "task_count": 0,
            "top_tasks": empty, "low_tasks": empty,
            "message": "Please enter a job title."
        }

    occ, occ_confidence, occ_match_method = find_occupation(query)

    if occ is None:
        return {
            "found": False, "query": query,
            "occupation": None, "onet_code": None,
            "occ_confidence": 0, "occ_match_method": "none",
            "avg_score": None, "level": "Unknown",
            "hourly_cad": None, "wage_matched": None,
            "wage_confidence": 0, "wage_match_method": "none",
            "monthly_saving": None, "task_count": 0,
            "top_tasks": empty, "low_tasks": empty,
            "message": f"No matching occupation found for '{query}'."
        }

    occ_title = occ["occupation_title"]
    onet_code = occ["onet_code"]

    # Pull tasks for this occupation
    job_tasks = _clean_tasks.loc[
        _clean_tasks["onet_code"] == onet_code,
        "task_description"
    ].dropna().tolist()

    scored = []
    for t in job_tasks:
        s, r = score_task(t)
        scored.append({"task": t, "score": s, "reason": r if r else "baseline"})
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    if len(scored) == 0:
        avg = None
        top_df = empty.copy()
        low_df = empty.copy()
    else:
        avg = round(sum(x["score"] for x in scored) / len(scored), 2)
        top_df = pd.DataFrame(scored[:5])
        low_df = pd.DataFrame(scored[-3:]).sort_values("score", ascending=True).reset_index(drop=True)

    # Wage + ROI (now returns 4 values)
    hourly, wage_label, wage_confidence, wage_match_method = find_wage(occ_title)

    monthly_saving = None
    if avg is not None and hourly is not None:
        monthly_saving = round(160 * avg * 0.4 * hourly)

    return {
        "found": True, "query": query,
        "occupation": occ_title, "onet_code": onet_code,
        "occ_confidence": occ_confidence, "occ_match_method": occ_match_method,
        "avg_score": avg, "level": risk_level(avg),
        "hourly_cad": hourly, "wage_matched": wage_label,
        "wage_confidence": wage_confidence, "wage_match_method": wage_match_method,
        "monthly_saving": monthly_saving, "task_count": len(scored),
        "top_tasks": top_df, "low_tasks": low_df,
        "message": None
    }


# Optional helper for UI dropdown/search suggestions
occupation_titles = sorted(_clean_occupations["occupation_title"].dropna().unique().tolist())

# ── Smoke tests ───────────────────────────────────────────────────────────────
_tests = ["nurse", "coder", "software developer", "accountant"]
print("\nBlock 3 ready ✓")
print(f"Occupations available : {len(occupation_titles):,}")
print(f"rapidfuzz available   : {_RAPIDFUZZ_AVAILABLE}")
print(f"sentence-transformers : {_ST_AVAILABLE}")
print(f"\n{'Query':<22} {'Matched Occupation':<40} {'Conf':>5} {'Method':<15}")
print("-" * 90)
for _q in _tests:
    _r = analyze(_q)
    _occ_str = _r['occupation'] or "No match"
    print(f"{_q:<22} {_occ_str:<40} {_r['occ_confidence']:>4}% {_r['occ_match_method']:<15}")
