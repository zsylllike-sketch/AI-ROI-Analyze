"""
Automation Risk API  —  FastAPI deployment script
Entrypoint: app/main.py
Run cmd: uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

import re
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from zerve import variable

# ── Load data from canvas blocks (Block3 -- Scoring Functions) ─────────────────
# Using a startup event to avoid 503 errors on cold starts; data is loaded
# lazily and cached at module level after first successful load.

_clean_occupations: Optional[pd.DataFrame] = None
_clean_tasks: Optional[pd.DataFrame] = None
_clean_wages: Optional[pd.DataFrame] = None
_occupation_titles: Optional[list] = None
_occ_title_list: Optional[list] = None
_wage_title_list: Optional[list] = None
_data_loaded: bool = False
_load_error: Optional[str] = None


def _load_data():
    """Load canvas variables once. Sets module-level _data_loaded flag."""
    global _clean_occupations, _clean_tasks, _clean_wages
    global _occupation_titles, _occ_title_list, _wage_title_list
    global _data_loaded, _load_error

    if _data_loaded:
        return

    clean_occupations: pd.DataFrame = variable("Block3 -- Scoring Functions", "clean_occupations")
    clean_tasks:       pd.DataFrame = variable("Block3 -- Scoring Functions", "clean_tasks")
    clean_wages:       pd.DataFrame = variable("Block3 -- Scoring Functions", "clean_wages")
    occupation_titles: list         = variable("Block3 -- Scoring Functions", "occupation_titles")

    # Ensure correct dtypes
    clean_occupations["occupation_title"] = clean_occupations["occupation_title"].astype(str).str.strip()
    clean_occupations["onet_code"]        = clean_occupations["onet_code"].astype(str).str.strip()
    clean_tasks["task_description"]       = clean_tasks["task_description"].astype(str).str.strip()
    clean_tasks["onet_code"]              = clean_tasks["onet_code"].astype(str).str.strip()
    clean_wages["noc_label"]              = clean_wages["noc_label"].astype(str).str.strip()
    clean_wages["avg_hourly_wage"]        = pd.to_numeric(clean_wages["avg_hourly_wage"], errors="coerce")

    _clean_occupations = clean_occupations
    _clean_tasks       = clean_tasks
    _clean_wages       = clean_wages
    _occupation_titles = occupation_titles
    _occ_title_list    = clean_occupations["occupation_title"].tolist()
    _wage_title_list   = clean_wages["noc_label"].tolist()
    _data_loaded       = True


# Load data at import time so it's ready before the first request
_load_data()

# ── Optional: rapidfuzz (graceful fallback if not installed) ──────────────────
try:
    from rapidfuzz import process as _rf_process, fuzz as _rf_fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

# ── Matching constants ────────────────────────────────────────────────────────
FUZZY_THRESHOLD    = 70
SEMANTIC_THRESHOLD = 0.35

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


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _score_task(task: str):
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


def _risk_level(avg_score: Optional[float]) -> str:
    if avg_score is None:
        return "Unknown"
    if avg_score >= 0.70:
        return "High Risk"
    if avg_score >= 0.45:
        return "Medium Risk"
    return "Low Risk"


def _find_occupation(job_title: str):
    """Returns (row_or_None, confidence_pct, match_method)"""
    query = str(job_title).strip()
    if not query:
        return None, 0, "none"

    # Tier 0: exact / substring match (always fast)
    mask = _clean_occupations["occupation_title"].str.contains(query, case=False, na=False, regex=False)
    exact = _clean_occupations[mask]
    if not exact.empty:
        return exact.iloc[0], 100, "exact"

    # Tier 1: rapidfuzz WRatio (if available)
    if _RAPIDFUZZ_AVAILABLE:
        best = _rf_process.extractOne(query, _occ_title_list, scorer=_rf_fuzz.WRatio)
        if best is not None:
            _, rf_score, rf_idx = best
            if rf_score >= FUZZY_THRESHOLD:
                return _clean_occupations.iloc[rf_idx], int(rf_score), "rapidfuzz"

    # Tier 2: word-based fallback
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
    for w in words:
        mask = _clean_occupations["occupation_title"].str.contains(w, case=False, na=False, regex=False)
        match = _clean_occupations[mask]
        if not match.empty:
            return match.iloc[0], 50, "word_fallback"

    return None, 0, "none"


def _find_wage(occ_title: str):
    """Returns (wage_or_None, label_or_None, confidence_pct, match_method)"""
    lower = str(occ_title).lower().strip()
    if not lower:
        return None, None, 0, "none"

    # Tier 0: substring match
    mask = _clean_wages["noc_label"].str.contains(lower, case=False, na=False, regex=False)
    exact = _clean_wages[mask]
    if not exact.empty:
        row = exact.iloc[0]
        wage = pd.to_numeric(row["avg_hourly_wage"], errors="coerce")
        wage = None if pd.isna(wage) else float(wage)
        return wage, row["noc_label"], 100, "exact"

    # Tier 1: rapidfuzz
    if _RAPIDFUZZ_AVAILABLE:
        best = _rf_process.extractOne(lower, _wage_title_list, scorer=_rf_fuzz.WRatio)
        if best is not None:
            _, rf_score, rf_idx = best
            if rf_score >= FUZZY_THRESHOLD:
                row = _clean_wages.iloc[rf_idx]
                wage = pd.to_numeric(row["avg_hourly_wage"], errors="coerce")
                wage = None if pd.isna(wage) else float(wage)
                return wage, row["noc_label"], int(rf_score), "rapidfuzz"

    # Word fallback
    words = [w for w in re.findall(r"\w+", lower) if len(w) > 3]
    for w in words:
        match = _clean_wages[_clean_wages["noc_label"].str.contains(w, case=False, na=False, regex=False)]
        if not match.empty:
            row = match.iloc[0]
            wage = pd.to_numeric(row["avg_hourly_wage"], errors="coerce")
            wage = None if pd.isna(wage) else float(wage)
            return wage, row["noc_label"], 50, "word_fallback"

    return None, None, 0, "none"


def _run_analyze(job_title: str) -> dict:
    """Full scoring pipeline — returns a JSON-serializable dict."""
    query = str(job_title).strip()
    if not query:
        return {"found": False, "query": query, "message": "Please enter a job title."}

    occ_row, occ_confidence, occ_method = _find_occupation(query)
    if occ_row is None:
        return {
            "found": False, "query": query,
            "message": f"No matching occupation found for '{query}'."
        }

    occ_title = occ_row["occupation_title"]
    onet_code = occ_row["onet_code"]

    job_tasks = _clean_tasks.loc[
        _clean_tasks["onet_code"] == onet_code, "task_description"
    ].dropna().tolist()

    scored = sorted(
        [{"task": t, "score": s, "reason": r or "baseline"}
         for t, (s, r) in ((t, _score_task(t)) for t in job_tasks)],
        key=lambda x: x["score"], reverse=True
    )

    if scored:
        avg = round(sum(x["score"] for x in scored) / len(scored), 2)
        top_tasks = [
            {"task": x["task"], "score": round(x["score"] * 100, 1), "reason": x["reason"]}
            for x in scored[:5]
        ]
        low_tasks = [
            {"task": x["task"], "score": round(x["score"] * 100, 1), "reason": x["reason"]}
            for x in sorted(scored[-3:], key=lambda x: x["score"])
        ]
    else:
        avg = None
        top_tasks = []
        low_tasks = []

    hourly, wage_label, wage_confidence, wage_method = _find_wage(occ_title)

    monthly_roi = None
    annual_roi  = None
    if avg is not None and hourly is not None:
        monthly_roi = round(160 * avg * 0.4 * hourly)
        annual_roi  = monthly_roi * 12

    return {
        "found": True,
        "query": query,
        "occupation": occ_title,
        "onet_code": onet_code,
        "occ_confidence": occ_confidence,
        "occ_match_method": occ_method,
        "risk_score": round(avg * 100) if avg is not None else None,
        "level": _risk_level(avg),
        "hourly_wage_cad": hourly,
        "wage_matched": wage_label,
        "wage_confidence": wage_confidence,
        "wage_match_method": wage_method,
        "monthly_roi": monthly_roi,
        "annual_roi": annual_roi,
        "task_count": len(scored),
        "top_tasks": top_tasks,
        "low_tasks": low_tasks,
        "message": None,
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Automation Risk API",
    description=(
        "Analyses automation risk for any job title using O*NET task data and "
        "StatCan wage data. Returns a 0–100 risk score, wage estimate, ROI, "
        "and task-level breakdown."
    ),
    version="1.0.0",
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    job_title: str


class TaskItem(BaseModel):
    task: str
    score: float
    reason: str


class AnalyzeResponse(BaseModel):
    found: bool
    query: str
    occupation: Optional[str] = None
    onet_code: Optional[str] = None
    occ_confidence: Optional[int] = None
    occ_match_method: Optional[str] = None
    risk_score: Optional[int] = None
    level: Optional[str] = None
    hourly_wage_cad: Optional[float] = None
    wage_matched: Optional[str] = None
    wage_confidence: Optional[int] = None
    wage_match_method: Optional[str] = None
    monthly_roi: Optional[int] = None
    annual_roi: Optional[int] = None
    task_count: Optional[int] = None
    top_tasks: Optional[List[TaskItem]] = None
    low_tasks: Optional[List[TaskItem]] = None
    message: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check", tags=["System"])
def health():
    """Returns 200 OK with basic service status when the API is ready."""
    occ_count = len(_occ_title_list) if _occ_title_list else 0
    return {
        "status": "ok",
        "occupations_loaded": occ_count,
        "data_loaded": _data_loaded,
    }


@app.get("/occupations", summary="List all occupation titles", tags=["Data"])
def list_occupations():
    """Returns the full list of occupation titles available for analysis."""
    titles = _occupation_titles if _occupation_titles else []
    return {"count": len(titles), "occupations": titles}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse automation risk for a job title",
    tags=["Analysis"],
)
def analyze_endpoint(req: AnalyzeRequest):
    """
    POST `{"job_title": "Marketing Manager"}` to receive:

    - **risk_score**: 0–100 automation risk score
    - **level**: Low Risk / Medium Risk / High Risk
    - **hourly_wage_cad**: matched StatCan hourly wage (CAD)
    - **monthly_roi** / **annual_roi**: estimated savings from automation
    - **top_tasks**: top 5 highest-risk tasks for this occupation
    - **low_tasks**: bottom 3 lowest-risk tasks for this occupation
    """
    if not _data_loaded:
        raise HTTPException(status_code=503, detail="Data not yet loaded. Please try again shortly.")

    result = _run_analyze(req.job_title)
    if not result.get("found") and result.get("message"):
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Automation Risk API is running. Visit /docs for the Swagger UI."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)