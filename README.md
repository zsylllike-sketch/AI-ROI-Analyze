# AI Automation Risk Analyzer

An end-to-end data pipeline and deployment system that analyses automation risk for any job title using O*NET task data and Statistics Canada wage data. Returns a 0–100 risk score, wage estimate, ROI projection, and task-level breakdown.

---

## Architecture Overview

```
Canvas Data Pipeline (Zerve Blocks)
────────────────────────────────────────────
Block1 → Block2 → Block3 → Block4 → Run test_api.py
                        ↓
              Deployment Scripts
         ┌──────────────────────────────┐
         │  FastAPI (backend API)       │  FastAPI  · risk-api.hub.zerve.cloud
         │  Streamlit (interactive UI)  │  Streamlit · risk-api-streamlit.hub.zerve.cloud
         │  Streamlit (direct canvas)   │  Streamlit · ai-roi.hub.zerve.cloud
         └──────────────────────────────┘
```

---

## Canvas Pipeline

All three deployment scripts depend on variables produced by the canvas notebook pipeline.
**The pipeline blocks must be run in order before any deployment script will work.**

| Block | Name | Purpose |
|-------|------|---------|
| Block1 | `Block1 -- load raw data` | Loads O*NET occupations, task statements, and StatCan wage CSV. Extracts latest Canadian average hourly wages. Produces: `occupations`, `tasks`, `statcan`, `wages` |
| Block2 | `Block2 -- clean / normalize` | Normalises and renames all three DataFrames. Drops survey metadata columns. Produces: `clean_occupations`, `clean_tasks`, `clean_wages` |
| Block3 | `Block3 -- analyze(job_title)` | Defines all scoring and matching logic. Pre-computes occupation title list. Produces: `clean_occupations`, `clean_tasks`, `clean_wages`, `occupation_titles`, `score_task`, `risk_level`, `find_occupation`, `find_wage`, `analyze` |
| Block4 | `Block4 -- static report generator` | End-to-end smoke test using "Marketing Manager". Generates and saves `automation_risk_report.png`. |
| Test | `Run test_api.py` | Runs `Test/test_api.py` in-process against canvas data. Verifies all API logic passes without a live server. |

### ⚠️ Critical: Run pipeline before deploying

Both the FastAPI script and one variant of the Streamlit app load canvas variables via
`from zerve import variable`. If `Block3 -- analyze(job_title)` has not been run to a
**success** state, the deployment container will fail at startup with a data loading
error that surfaces as a **503 Service Unavailable**.

**Correct startup order:**

```
1. Run Block1  →  success ✅
2. Run Block2  →  success ✅
3. Run Block3  →  success ✅   ← backend API and interactive frontend depend on this block
4. Deploy / restart the API and Streamlit scripts
```

---

## Deployment Scripts

### 1. Backend API (FastAPI)

| Property | Value |
|----------|-------|
| **Script name** | `Automation Risk API` |
| **Framework** | FastAPI |
| **Entrypoint** | `app/main.py` |
| **Run command** | `uvicorn app.main:app --host 0.0.0.0 --port 8080` |
| **DNS** | `https://risk-api.hub.zerve.cloud` |

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `status`, `data_loaded`, `occupations_loaded` |
| `GET` | `/occupations` | Full list of occupation titles (1,016 items) |
| `POST` | `/analyze` | Analyse a job title. Body: `{"job_title": "Software Developer"}` |
| `GET` | `/docs` | Swagger UI |

#### Canvas variable loading (import-time)

The API script loads canvas variables **at module import time**, not inside a startup
event. This is the pattern that resolved 503 errors on cold starts:

```python
from zerve import variable

# Loaded once when uvicorn imports app/main.py
_clean_occupations = variable("Block3 -- analyze(job_title)", "clean_occupations")
_clean_tasks       = variable("Block3 -- analyze(job_title)", "clean_tasks")
_clean_wages       = variable("Block3 -- analyze(job_title)", "clean_wages")
_occupation_titles = variable("Block3 -- analyze(job_title)", "occupation_titles")
_data_loaded = True
```

A `_data_loaded` flag guards the `/analyze` endpoint; if the import-time load ever
fails, the endpoint raises `HTTP 503` with a clear message rather than an unhandled
exception.

---

### 2. Interactive Frontend (Streamlit — API-backed, `risk-api-streamlit`)

| Property | Value |
|----------|-------|
| **Script name** | `Streamlit` (medium compute) |
| **Framework** | Streamlit |
| **Entrypoint** | `app/main.py` |
| **Run command** | `streamlit run app/main.py --server.port 8080 --server.address 0.0.0.0` |
| **DNS** | `https://risk-api-streamlit.hub.zerve.cloud` |

This variant does **not** import canvas variables directly. It calls the FastAPI backend
(`https://risk-api.hub.zerve.cloud`) for all analysis, making it fully decoupled from
the canvas pipeline at runtime.

#### `requests` → `urllib` fallback

The `requests` package may not be available in the Streamlit deployment environment.
The script handles this gracefully with a try/except import pattern:

```python
try:
    import requests as _requests

    def _http_get(url, timeout=30):
        r = _requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _http_post(url, json_data, timeout=60):
        r = _requests.post(url, json=json_data, timeout=timeout)
        return r.status_code, r.json()

except ImportError:
    import urllib.request, urllib.error, json as _json

    def _http_get(url, timeout=30):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode())

    def _http_post(url, json_data, timeout=60):
        data = _json.dumps(json_data).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, _json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, _json.loads(e.read().decode())
```

`urllib` is part of the Python standard library and is always available — no package
installation required.

---

### 3. Interactive Frontend (Streamlit — direct canvas, `ai-roi`)

| Property | Value |
|----------|-------|
| **Script name** | `Streamlit` (small compute) |
| **Framework** | Streamlit |
| **Entrypoint** | `app/main.py` |
| **Run command** | `streamlit run app/main.py --server.port 8080 --server.address 0.0.0.0` |
| **DNS** | `https://ai-roi.hub.zerve.cloud` |

This variant loads DataFrames **directly from the canvas block** using
`from zerve import variable`, then runs all scoring logic locally inside the script.
It does not call the FastAPI backend.

```python
from zerve import variable

@st.cache_resource(show_spinner="Loading occupation data…")
def _load_data():
    _occ    = variable("Block3 -- analyze(job_title)", "clean_occupations")
    _tasks  = variable("Block3 -- analyze(job_title)", "clean_tasks")
    _wages  = variable("Block3 -- analyze(job_title)", "clean_wages")
    _titles = variable("Block3 -- analyze(job_title)", "occupation_titles")
    return _occ, _tasks, _wages, _titles

clean_occupations, clean_tasks, clean_wages, occupation_titles = _load_data()
```

The `@st.cache_resource` decorator ensures DataFrames are loaded exactly once per
server lifecycle, eliminating repeated cold-start data fetches.

---

## Troubleshooting

### 503 Service Unavailable on API startup

**Root cause:** The canvas pipeline (Block1 → Block2 → Block3) had not been run
to success before the backend API was deployed, so `variable(...)` calls at import
time found no serialised data to load.

**Fix applied:**

1. **Run the full pipeline** in order before deploying or restarting the API script.
   Block3 must show a green ✅ status — that is the block the API reads from.

2. **Canvas variables are loaded at import time** (not in a `@app.on_event("startup")`
   callback). This ensures data is ready before uvicorn begins accepting requests,
   preventing the race condition that caused 503s on cold starts.

3. The `/health` endpoint now reflects actual load state:
   ```json
   { "status": "ok", "occupations_loaded": 1016, "data_loaded": true }
   ```
   A `data_loaded: false` response indicates the pipeline was not run first.

### Streamlit app shows "API request failed"

- Check that the **backend API** (`Automation Risk API`) is deployed and healthy:
  `GET https://risk-api.hub.zerve.cloud/health`
- If the API is healthy but the interactive frontend still fails, verify the `requests` →
  `urllib` fallback is present in `app/main.py`. The urllib fallback is the safe
  default when the `requests` package is not installed in the Streamlit environment.

### "No matching occupation found" for a valid job title

- Block3 may be stale. Re-run Block3 so the API picks up the latest
  `clean_occupations` DataFrame.
- Try the `/docs` Swagger UI to test the `/analyze` endpoint directly and inspect
  the raw response.

---

## Data Sources

| Dataset | File | Records |
|---------|------|---------|
| O*NET Occupations | `Data/Occupation Data.txt` | 1,016 occupations |
| O*NET Task Statements | `Data/Task Statements.txt` | 18,796 tasks |
| Statistics Canada wages | `Data/14100444.csv` | 5.7 M rows (filtered to 765 wage records) |

---

## Scoring Logic

### Automation risk score (0–100)

Each task is scored by keyword matching against two word lists:

- **HIGH_AUTO keywords** (25 words: compile, collect, analyze, compute, …): `+0.08` per match
- **LOW_AUTO keywords** (22 words: negotiate, counsel, design, lead, …): `−0.10` per match
- **Baseline score**: `0.50` (clipped to `[0, 1]`)

The occupation score is the average across all its O*NET task statements, scaled to 0–100.

### Risk levels

| Level | Score range | Meaning |
|-------|-------------|---------|
| 🟢 Low Risk | 0 – 44 | Creative, social, strategic tasks dominate |
| 🟡 Medium Risk | 45 – 69 | Mix of routine and complex tasks |
| 🔴 High Risk | 70 – 100 | Routine or repetitive tasks dominate |

### ROI estimate

```
monthly_roi = 160 hrs/mo × automation_score × 0.40 × hourly_wage_CAD
annual_roi  = monthly_roi × 12
```

Assumption: 40% of automatable hours are realistically saved per month.

### Occupation matching (tiered)

| Tier | Method | Fallback condition |
|------|--------|--------------------|
| 0 | Exact / substring match | Always attempted first |
| 1 | rapidfuzz WRatio (≥ 70 threshold) | If Tier 0 fails |
| 2 | sentence-transformers cosine sim (≥ 0.35) | If Tier 1 score < 70 or rapidfuzz unavailable |
| 3 | Word-token keyword fallback | If all tiers fail |

---

## Project Files

```
/
├── Data/
│   ├── Occupation Data.txt        # O*NET occupation titles & descriptions
│   ├── Task Statements.txt        # O*NET task statements per occupation
│   └── 14100444.csv               # Statistics Canada wage survey
├── Test/
│   └── test_api.py                # Full API test suite (run via "Run test_api.py" block)
├── api.py                         # Legacy standalone API entrypoint (reference only)
├── streamlit.py                   # Standalone Streamlit script (reference / local dev)
├── automation_risk_report.png     # Latest chart output from Block4 smoke test
└── README.md                      # This file
```

---

## Local Development

To run the API locally (requires pipeline data already serialised on canvas):

```bash
# Install dependencies
pip install fastapi uvicorn pandas numpy rapidfuzz

# Run the API
uvicorn app.main:app --reload --port 8080
```

To run the Streamlit app locally against the deployed API:

```bash
pip install streamlit pandas numpy matplotlib

# The app calls https://risk-api.hub.zerve.cloud — no local API needed
streamlit run streamlit.py
```

---

*Last updated: 2026-03-22 — reflects 503 fix: import-time canvas variable loading,
urllib fallback for requests, and Block1→Block2→Block3 pipeline execution requirement.*
