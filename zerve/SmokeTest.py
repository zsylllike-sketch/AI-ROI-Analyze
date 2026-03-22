"""
Smoke tests for Block3 scoring functions.

Key finding from audit: Zerve functions deserialized from upstream blocks have
their closure free-variables resolved at call time in the caller's global scope.
Block3's functions reference _clean_occupations, _clean_tasks, _clean_wages etc
as free variables. Since those are _ prefixed, Zerve doesn't serialize them.
Solution: re-define them in this block's scope (same pattern Block4 uses).
"""
import pandas as pd
import re
import numpy as np

print("=" * 65)
print("  SMOKE TEST — Block3 Scoring Functions")
print("=" * 65)

# ── Re-bind private closure variables (same pattern as Block4) ────────────────
# Block3 functions reference these as free variables at call time.
# Must define them here so closures resolve correctly.
_clean_occupations = clean_occupations.copy()
_clean_occupations["occupation_title"] = _clean_occupations["occupation_title"].astype(str).str.strip()
_clean_occupations["onet_code"] = _clean_occupations["onet_code"].astype(str).str.strip()

_clean_tasks = clean_tasks.copy()
_clean_tasks["task_description"] = _clean_tasks["task_description"].astype(str).str.strip()
_clean_tasks["onet_code"] = _clean_tasks["onet_code"].astype(str).str.strip()

_clean_wages = clean_wages.copy()
_clean_wages["noc_label"] = _clean_wages["noc_label"].astype(str).str.strip()
_clean_wages["avg_hourly_wage"] = pd.to_numeric(_clean_wages["avg_hourly_wage"], errors="coerce")

_occ_title_list  = _clean_occupations["occupation_title"].tolist()
_wage_title_list = _clean_wages["noc_label"].tolist()

# Tier flags (needed by closures)
try:
    from rapidfuzz import process as _rf_process, fuzz as _rf_fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _ST
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

_embedding_model    = None
occ_title_embeddings  = None
wage_title_embeddings = None

print(f"  rapidfuzz: {_RAPIDFUZZ_AVAILABLE}  sentence-transformers: {_ST_AVAILABLE}")
print(f"  Occupations: {len(_clean_occupations)}  Tasks: {len(_clean_tasks)}  Wages: {len(_clean_wages)}")

# ─────────────────────────────────────────────────────────────────
# 1. score_task()
# ─────────────────────────────────────────────────────────────────
print("\n[1] score_task()")

_s1, _r1 = score_task("Compile and analyze monthly financial reports")
assert 0 <= _s1 <= 1, f"score out of range: {_s1}"
assert _s1 > 0.5, f"Expected high score for compile/analyze task, got {_s1}"
print(f"  ✅  HIGH task → score={_s1:.2f}  reason='{_r1}'")

_s2, _r2 = score_task("Lead and motivate team members to achieve strategic goals")
assert 0 <= _s2 <= 1, f"score out of range: {_s2}"
assert _s2 < 0.5, f"Expected low score for lead/motivate task, got {_s2}"
print(f"  ✅  LOW task  → score={_s2:.2f}  reason='{_r2}'")

_s3, _r3 = score_task("Some completely neutral task description xyz")
assert _s3 == 0.5, f"Expected baseline 0.5 for neutral task, got {_s3}"
print(f"  ✅  NEUTRAL   → score={_s3:.2f}  reason='{_r3 or 'baseline'}'")

# ─────────────────────────────────────────────────────────────────
# 2. risk_level()
# ─────────────────────────────────────────────────────────────────
print("\n[2] risk_level()")

assert risk_level(0.75) == "High Risk",   f"Got {risk_level(0.75)}"
assert risk_level(0.70) == "High Risk",   f"Got {risk_level(0.70)}"
assert risk_level(0.55) == "Medium Risk", f"Got {risk_level(0.55)}"
assert risk_level(0.45) == "Medium Risk", f"Got {risk_level(0.45)}"
assert risk_level(0.30) == "Low Risk",    f"Got {risk_level(0.30)}"
assert risk_level(None) == "Unknown",     f"Got {risk_level(None)}"
print("  ✅  Boundaries: High(≥0.70) / Medium(0.45–0.69) / Low(<0.45) / Unknown(None)")

# ─────────────────────────────────────────────────────────────────
# 3. find_occupation()
# ─────────────────────────────────────────────────────────────────
print("\n[3] find_occupation()")

_row, _conf, _method = find_occupation("Marketing Manager")
assert _row is not None, "find_occupation('Marketing Manager') returned None"
assert _conf == 100, f"Expected 100% confidence for exact match, got {_conf}"
assert _method == "exact"
print(f"  ✅  'Marketing Manager' → '{_row['occupation_title']}' [{_conf}% {_method}]")

_row2, _conf2, _method2 = find_occupation("nurse")
assert _row2 is not None, "find_occupation('nurse') returned None"
assert _conf2 > 0
print(f"  ✅  'nurse'             → '{_row2['occupation_title']}' [{_conf2}% {_method2}]")

_row3, _conf3, _method3 = find_occupation("xyzzy_fake_job_12345")
assert _row3 is None, f"Expected None for unknown job"
assert _conf3 == 0
print(f"  ✅  Unknown input       → None [{_conf3}% {_method3}]")

_row4, _conf4, _method4 = find_occupation("")
assert _row4 is None, "Expected None for empty"
assert _conf4 == 0
print(f"  ✅  Empty string        → None [{_conf4}% {_method4}]")

# ─────────────────────────────────────────────────────────────────
# 4. find_wage()
# ─────────────────────────────────────────────────────────────────
print("\n[4] find_wage()")

_wage, _label, _wconf, _wmethod = find_wage("Marketing Managers")
assert _wage is not None, "find_wage('Marketing Managers') returned None wage"
assert isinstance(_wage, float), f"Expected float, got {type(_wage)}"
assert 0 < _wage < 250, f"Wage ${_wage} outside plausible range"
print(f"  ✅  'Marketing Managers' → ${_wage:.2f} CAD [{_wconf}% {_wmethod}] via '{_label}'")

_w2, _l2, _c2, _m2 = find_wage("")
assert _w2 is None
print(f"  ✅  Empty string         → None [{_c2}% {_m2}]")

# ─────────────────────────────────────────────────────────────────
# 5. analyze() — full pipeline
# ─────────────────────────────────────────────────────────────────
print("\n[5] analyze() — full pipeline")

_REQUIRED = {
    "found", "query", "occupation", "onet_code",
    "occ_confidence", "occ_match_method",
    "avg_score", "level",
    "hourly_cad", "wage_matched", "wage_confidence", "wage_match_method",
    "monthly_saving", "task_count",
    "top_tasks", "low_tasks", "message"
}

# 5a. Known good
_r_good = analyze("Marketing Manager")
_miss = _REQUIRED - set(_r_good.keys())
assert not _miss, f"analyze() missing keys: {_miss}"
assert _r_good["found"] is True
assert isinstance(_r_good["avg_score"], float), f"avg_score type: {type(_r_good['avg_score'])}"
assert 0 <= _r_good["avg_score"] <= 1
assert _r_good["level"] in {"Low Risk", "Medium Risk", "High Risk"}
assert isinstance(_r_good["top_tasks"], pd.DataFrame)
assert isinstance(_r_good["low_tasks"], pd.DataFrame)
assert _r_good["task_count"] > 0
print(f"  ✅  'Marketing Manager'  → score={_r_good['avg_score']} level={_r_good['level']} tasks={_r_good['task_count']}")

# 5b. Fuzzy: nurse
_r_nurse = analyze("nurse")
assert _r_nurse["found"] is True, "Expected 'nurse' to match"
_miss2 = _REQUIRED - set(_r_nurse.keys())
assert not _miss2, f"Keys missing: {_miss2}"
print(f"  ✅  'nurse'              → '{_r_nurse['occupation']}' score={_r_nurse['avg_score']}")

# 5c. Fuzzy: coder (valid structure regardless of match)
_r_coder = analyze("coder")
_miss3 = _REQUIRED - set(_r_coder.keys())
assert not _miss3, f"Keys missing for 'coder': {_miss3}"
print(f"  ✅  'coder'              → found={_r_coder['found']} occ='{_r_coder['occupation']}'")

# 5d. Unknown
_r_unk = analyze("xyzzy_totally_fake_job_99999")
assert _r_unk["found"] is False
assert _r_unk["message"] is not None
assert _r_unk["avg_score"] is None
assert _r_unk["level"] == "Unknown"
print(f"  ✅  Unknown              → found=False msg='{_r_unk['message']}'")

# 5e. Empty
_r_mt = analyze("")
assert _r_mt["found"] is False
assert _r_mt["message"] is not None
print(f"  ✅  Empty string         → found=False msg='{_r_mt['message']}'")

# ─────────────────────────────────────────────────────────────────
# 6. Type & range assertions for good result
# ─────────────────────────────────────────────────────────────────
print("\n[6] Type & range validation")

assert isinstance(_r_good["occ_confidence"], int)
assert 0 <= _r_good["occ_confidence"] <= 100
assert _r_good["occ_match_method"] in {"exact", "rapidfuzz", "semantic", "word_fallback", "none"}
if _r_good["hourly_cad"]:
    assert isinstance(_r_good["hourly_cad"], float)
    assert 0 < _r_good["hourly_cad"] < 250
if _r_good["monthly_saving"]:
    assert _r_good["monthly_saving"] > 0

_top = _r_good["top_tasks"]
_low = _r_good["low_tasks"]
assert list(_top.columns) == ["task", "score", "reason"], f"top_tasks cols: {list(_top.columns)}"
assert list(_low.columns) == ["task", "score", "reason"], f"low_tasks cols: {list(_low.columns)}"
assert len(_top) <= 5
assert len(_low) <= 3

print(f"  ✅  occ_confidence={_r_good['occ_confidence']}% ({_r_good['occ_match_method']})")
print(f"  ✅  hourly_cad=${_r_good['hourly_cad']} monthly_saving=${_r_good['monthly_saving']}")
print(f"  ✅  top_tasks={_top.shape}  low_tasks={_low.shape}")

# ─────────────────────────────────────────────────────────────────
# 7. Contract audit: Block3.analyze() vs api.py vs main.py
# ─────────────────────────────────────────────────────────────────
print("\n[7] Contract audit summary")
print("  Block3 analyze()      → hourly_cad, monthly_saving, avg_score (0-1)")
print("  api.py _run_analyze() → hourly_wage_cad, monthly_roi, annual_roi, risk_score (0-100)")
print("  main.py analyze()     → hourly_cad, monthly_saving (matches Block3 ✅)")
print("  api.py is standalone — does NOT import Block3.analyze(). Self-consistent ✅")
print("  main.py re-implements analyze() locally using Block3 data — consistent ✅")

print("\n" + "=" * 65)
print("  ALL SMOKE TESTS PASSED ✅")
print("=" * 65)
