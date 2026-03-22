import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Try to import requests; if unavailable, use urllib instead ────────────────
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
    import urllib.request
    import urllib.error
    import json as _json

    def _http_get(url, timeout=30):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode())

    def _http_post(url, json_data, timeout=60):
        data = _json.dumps(json_data).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, _json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, _json.loads(e.read().decode())

API_BASE     = "https://risk-api.hub.zerve.cloud"
ANALYZE_URL  = f"{API_BASE}/analyze"
OCCUPATIONS_URL = f"{API_BASE}/occupations"

BG       = "#1D1D20"
FG       = "#fbfbff"
SUBTEXT  = "#909094"
YELLOW   = "#ffd400"
GREEN    = "#17b26a"
ORANGE   = "#FFB482"
CORAL    = "#FF9F9B"
BLUE     = "#A1C9F4"
RISK_COLORS = {"Low Risk": GREEN, "Medium Risk": ORANGE, "High Risk": CORAL}


@st.cache_data(ttl=3600, show_spinner=False)
def load_occupations():
    try:
        data = _http_get(OCCUPATIONS_URL, timeout=30)
        return data.get("occupations", [])
    except Exception:
        return []


occupation_titles = load_occupations()


def confidence_badge(confidence: int, method: str) -> str:
    if confidence >= 90:
        color = GREEN
    elif confidence >= 70:
        color = YELLOW
    elif confidence >= 50:
        color = ORANGE
    else:
        color = CORAL

    method_labels = {
        "exact": "exact match",
        "rapidfuzz": "fuzzy match",
        "semantic": "semantic match",
        "word_fallback": "keyword match",
        "none": "no match",
    }
    label = method_labels.get(method, method)

    return (
        f"<span style='background:{color}22; color:{color}; border:1px solid {color}55; "
        f"border-radius:12px; padding:2px 10px; font-size:0.78rem; font-weight:600;'>"
        f"{confidence}% confidence · {label}</span>"
    )


def normalize_result(api_result: dict) -> dict:
    top_tasks = pd.DataFrame(api_result.get("top_tasks", []))
    low_tasks = pd.DataFrame(api_result.get("low_tasks", []))

    return {
        "found": api_result.get("found", False),
        "query": api_result.get("query"),
        "occupation": api_result.get("occupation"),
        "onet_code": api_result.get("onet_code"),
        "occ_confidence": api_result.get("occ_confidence", 0),
        "occ_match_method": api_result.get("occ_match_method", "none"),
        "risk_score": api_result.get("risk_score"),
        "level": api_result.get("level", "Unknown"),
        "hourly_cad": api_result.get("hourly_wage_cad"),
        "wage_matched": api_result.get("wage_matched"),
        "wage_confidence": api_result.get("wage_confidence", 0),
        "wage_match_method": api_result.get("wage_match_method", "none"),
        "monthly_saving": api_result.get("monthly_roi"),
        "annual_roi": api_result.get("annual_roi"),
        "task_count": api_result.get("task_count", 0),
        "top_tasks": top_tasks,
        "low_tasks": low_tasks,
        "message": api_result.get("message"),
    }


def analyze_via_api(job_title: str) -> dict:
    query = str(job_title).strip()
    if not query:
        return {
            "found": False,
            "message": "Please enter a job title."
        }

    try:
        status_code, response_data = _http_post(
            ANALYZE_URL,
            json_data={"job_title": query},
            timeout=60
        )

        if status_code == 404:
            return {"found": False, "message": f"No matching occupation found for '{query}'."}

        if status_code == 422:
            return {"found": False, "message": "Please enter a valid job title."}

        if status_code >= 400:
            return {"found": False, "message": f"API error ({status_code}). Please try again."}

        return normalize_result(response_data)

    except Exception as e:
        return {
            "found": False,
            "message": f"API request failed: {e}"
        }


def build_chart(result: dict):
    occ         = result["occupation"]
    score_0_100 = result["risk_score"] if result["risk_score"] is not None else 0
    level       = result["level"]
    hourly      = result["hourly_cad"]
    monthly_roi = result["monthly_saving"]
    annual_roi  = result["annual_roi"]
    wage_label  = result["wage_matched"] or "N/A"
    top_tasks   = result["top_tasks"]
    low_tasks   = result["low_tasks"]
    rc          = RISK_COLORS.get(level, BLUE)

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.97, top=0.88, bottom=0.10,
        hspace=0.55, wspace=0.40
    )

    fig.text(0.5, 0.95, f"Automation Risk Report · {occ}",
             ha="center", va="center", fontsize=16, fontweight="bold", color=FG)
    fig.text(0.5, 0.915, "Frontend: Streamlit · Backend: FastAPI",
             ha="center", va="center", fontsize=10, color=SUBTEXT)

    # Gauge
    ax_gauge = fig.add_subplot(gs[0, 0])
    ax_gauge.set_facecolor(BG)
    ax_gauge.set_aspect("equal")

    for t_s, t_e, col in [
        (np.pi, np.pi * 2/3, GREEN),
        (np.pi * 2/3, np.pi / 3, ORANGE),
        (np.pi / 3, 0, CORAL)
    ]:
        t = np.linspace(t_s, t_e, 100)
        ax_gauge.plot(np.cos(t), np.sin(t), color=col, lw=12, solid_capstyle="butt", alpha=0.85)

    ang = np.pi * (1 - score_0_100 / 100)
    ax_gauge.annotate("", xy=(0.72*np.cos(ang), 0.72*np.sin(ang)), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=FG, lw=2, mutation_scale=14))
    ax_gauge.plot(0, 0, "o", color=FG, ms=6, zorder=5)
    ax_gauge.text(0, -0.22, f"{score_0_100}", ha="center", va="center",
                  fontsize=30, fontweight="bold", color=rc)
    ax_gauge.text(0, -0.46, "/ 100", ha="center", va="center", fontsize=11, color=SUBTEXT)
    ax_gauge.text(0, -0.68, level, ha="center", va="center",
                  fontsize=12, fontweight="bold", color=rc)
    ax_gauge.set_xlim(-1.3, 1.3)
    ax_gauge.set_ylim(-0.85, 1.15)
    ax_gauge.axis("off")
    ax_gauge.set_title("Automation Risk Score", color=FG, fontsize=11, pad=8)

    # ROI
    ax_roi = fig.add_subplot(gs[0, 1])
    ax_roi.set_facecolor(BG)
    ax_roi.axis("off")
    ax_roi.set_title("ROI Estimate", color=FG, fontsize=11, pad=8)

    cards = [
        ("Matched Occupation", occ, FG),
        ("Wage Data Match", wage_label, SUBTEXT),
        ("Avg Hourly Wage", f"${hourly:.2f} CAD" if hourly is not None else "N/A", YELLOW),
        ("Monthly Time Savings", f"${monthly_roi:,} CAD" if monthly_roi is not None else "N/A", GREEN),
        ("Annual ROI Estimate", f"${annual_roi:,} CAD" if annual_roi is not None else "N/A", GREEN),
        ("Tasks Scored", str(result["task_count"]), SUBTEXT),
    ]

    y_pos = 0.97
    for lbl, val, col in cards:
        ax_roi.text(0, y_pos, lbl, transform=ax_roi.transAxes, fontsize=8, color=SUBTEXT, va="top")
        y_pos -= 0.11
        ax_roi.text(0, y_pos, str(val), transform=ax_roi.transAxes, fontsize=10,
                    fontweight="bold", color=col, va="top", wrap=True)
        y_pos -= 0.08

    # Guide
    ax_ctx = fig.add_subplot(gs[0, 2])
    ax_ctx.set_facecolor(BG)
    ax_ctx.axis("off")
    ax_ctx.set_title("Risk Level Guide", color=FG, fontsize=11, pad=8)

    guide = [
        ("🟢 Low Risk", "0 – 44", "Creative, social, and strategic work dominates.", GREEN),
        ("🟡 Medium Risk", "45 – 69", "Mixed routine and higher-level work.", ORANGE),
        ("🔴 High Risk", "70 – 100", "Routine and repetitive work dominates.", CORAL),
    ]

    y_pos = 0.96
    for ttl, rng, dsc, col in guide:
        ax_ctx.text(0, y_pos, f"{ttl} ({rng})", transform=ax_ctx.transAxes,
                    fontsize=9.5, fontweight="bold", color=col, va="top")
        y_pos -= 0.10
        ax_ctx.text(0.03, y_pos, dsc, transform=ax_ctx.transAxes,
                    fontsize=8, color=SUBTEXT, va="top", wrap=True)
        y_pos -= 0.20

    # Tasks
    ax_tasks = fig.add_subplot(gs[1, :])
    ax_tasks.set_facecolor(BG)

    all_tasks = pd.concat([top_tasks, low_tasks], ignore_index=True).drop_duplicates(subset="task").copy()
    if not all_tasks.empty:
        all_tasks["score_pct"] = all_tasks["score"]
        all_tasks["short_task"] = all_tasks["task"].str[:72].str.strip()
        all_tasks = all_tasks.sort_values("score_pct", ascending=True)

        bar_colors = [
            CORAL if s >= 70 else ORANGE if s >= 45 else GREEN
            for s in all_tasks["score_pct"]
        ]

        bars = ax_tasks.barh(
            all_tasks["short_task"],
            all_tasks["score_pct"],
            color=bar_colors,
            height=0.6,
            edgecolor="none"
        )

        for b, v in zip(bars, all_tasks["score_pct"]):
            ax_tasks.text(
                b.get_width() + 0.8,
                b.get_y() + b.get_height()/2,
                f"{v:.0f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color=FG,
                fontweight="bold"
            )

        ax_tasks.axvline(score_0_100, color=YELLOW, lw=1.5, linestyle="--", alpha=0.8)

    ax_tasks.set_xlim(0, 115)
    ax_tasks.set_xlabel("Automation Risk Score (0–100)", color=SUBTEXT, fontsize=9)
    ax_tasks.set_title("Task-Level Automation Risk Breakdown", color=FG, fontsize=11, pad=8, loc="left")
    ax_tasks.tick_params(axis="y", colors=FG, labelsize=8)
    ax_tasks.tick_params(axis="x", colors=SUBTEXT, labelsize=8)

    for spine in ax_tasks.spines.values():
        spine.set_visible(False)

    ax_tasks.xaxis.grid(True, color=SUBTEXT, alpha=0.15, lw=0.6)
    ax_tasks.set_axisbelow(True)

    leg_patches = [
        mpatches.Patch(color=GREEN, label="Low Risk (<45)"),
        mpatches.Patch(color=ORANGE, label="Medium Risk (45–69)"),
        mpatches.Patch(color=CORAL, label="High Risk (≥70)")
    ]
    ax_tasks.legend(handles=leg_patches, loc="lower right", framealpha=0, labelcolor=FG, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# UI
st.set_page_config(
    page_title="AI Automation Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {FG}; }}
    .stSidebar {{ background-color: #141416; }}
    .stButton > button {{
        background-color: #ffd400;
        color: #1D1D20;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
    }}
    .stButton > button:hover {{ background-color: #e6bf00; }}
    h1, h2, h3 {{ color: {FG}; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<h2 style='color:{FG}'>🤖 Automation Risk</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT}'>Frontend powered by Streamlit, backend powered by FastAPI.</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"<p style='color:{SUBTEXT}; font-size:0.85rem;'>📊 {len(occupation_titles):,} occupations loaded</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT}; font-size:0.8rem;'>API: {API_BASE}</p>", unsafe_allow_html=True)

st.markdown(f"<h1 style='color:{FG}; margin-bottom:0;'>AI Automation Risk Analyzer</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{SUBTEXT}; margin-top:0.25rem;'>Enter any job title to analyze via the FastAPI backend.</p>", unsafe_allow_html=True)

col_input, col_btn = st.columns([4, 1])
with col_input:
    job_input = st.text_input(
        "Job Title",
        placeholder="e.g. Software Developer, Accountant, Nurse...",
        label_visibility="collapsed"
    )
with col_btn:
    run_btn = st.button("Analyze ▶", use_container_width=True)

if job_input and len(job_input) >= 2:
    suggestions = [t for t in occupation_titles if job_input.lower() in t.lower()][:5]
    if suggestions:
        st.caption("💡 Suggestions: " + " · ".join(suggestions))

st.divider()

if run_btn and job_input.strip():
    with st.spinner("Analyzing via API..."):
        result = analyze_via_api(job_input.strip())

    if not result.get("found", False):
        st.warning(result.get("message", "Analysis failed."))
    else:
        occ = result["occupation"]
        score_0_100 = result["risk_score"] if result["risk_score"] is not None else 0
        level = result["level"]
        hourly = result["hourly_cad"]
        annual_roi = result["annual_roi"]
        rc = RISK_COLORS.get(level, BLUE)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Occupation", occ)
        m2.metric("Risk Score", f"{score_0_100}/100", delta=level)
        m3.metric("Avg Hourly Wage", f"${hourly:.2f} CAD" if hourly is not None else "N/A")
        m4.metric("Est. Annual ROI", f"${annual_roi:,} CAD" if annual_roi is not None else "N/A")

        occ_badge = confidence_badge(result["occ_confidence"], result["occ_match_method"])
        wage_badge = confidence_badge(result["wage_confidence"], result["wage_match_method"])

        st.markdown(
            f"<div style='margin-top:0.25rem; margin-bottom:0.25rem;'>"
            f"<span style='color:{SUBTEXT}; font-size:0.85rem;'>Occupation match:</span> {occ_badge}"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='margin-bottom:0.75rem;'>"
            f"<span style='color:{SUBTEXT}; font-size:0.85rem;'>Wage match:</span> {wage_badge}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(f"<p style='color:{rc}; font-size:1.1rem; font-weight:bold;'>⚡ {level}</p>", unsafe_allow_html=True)

        fig = build_chart(result)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("📋 Task Details", expanded=False):
            tab_high, tab_low = st.tabs(["🔴 Highest Risk Tasks", "🟢 Lowest Risk Tasks"])
            with tab_high:
                if not result["top_tasks"].empty:
                    st.dataframe(result["top_tasks"], use_container_width=True, hide_index=True)
            with tab_low:
                if not result["low_tasks"].empty:
                    st.dataframe(result["low_tasks"], use_container_width=True, hide_index=True)

elif run_btn:
    st.info("Please enter a job title to analyze.")
else:
    st.markdown(f"""
    <div style='text-align:center; padding:3rem; color:{SUBTEXT};'>
        <p style='font-size:3rem;'>🤖</p>
        <p style='font-size:1.2rem; color:{FG};'>Enter a job title above and click <strong>Analyze</strong></p>
        <p>The Streamlit frontend will call your FastAPI backend and render the result.</p>
    </div>
    """, unsafe_allow_html=True)