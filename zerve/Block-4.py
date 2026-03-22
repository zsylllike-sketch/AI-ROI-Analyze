import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import re

# ── Config ──────────────────────────────────────────────────────────────────
JOB_TITLE = "Marketing Manager"          # ← change this to any job title

BG       = "#1D1D20"
FG       = "#fbfbff"
SUBTEXT  = "#909094"
YELLOW   = "#ffd400"
GREEN    = "#17b26a"
ORANGE   = "#FFB482"
CORAL    = "#FF9F9B"
BLUE     = "#A1C9F4"
LAVENDER = "#D0BBFF"

RISK_COLORS = {"Low Risk": GREEN, "Medium Risk": ORANGE, "High Risk": CORAL}

# ── Restore private closure variables from Block3 ────────────────────────────
# Block3 exposes clean_* as public variables but its helper functions reference
# the underscore-prefixed copies AND the rapidfuzz/sentence-transformers flags.
# Re-bind them here so the closures resolve correctly in this block's scope.
_clean_occupations = clean_occupations.copy()
_clean_tasks       = clean_tasks.copy()
_clean_wages       = clean_wages.copy()

# Tier availability flags (closures from Block3 reference these)
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

# Embeddings are None (not serialized), but closures handle that gracefully
_embedding_model    = None
occ_title_embeddings  = None
wage_title_embeddings = None

# Occupation title list used by find_occupation closure
_occ_title_list  = _clean_occupations["occupation_title"].tolist()
_wage_title_list = _clean_wages["noc_label"].tolist()

# ── Run analysis ─────────────────────────────────────────────────────────────
result = analyze(JOB_TITLE)

if isinstance(result, str):
    print(f"⚠️  {result}")
    raise SystemExit(result)

occ             = result["occupation"]
score_0_100     = round(result["avg_score"] * 100)
level           = result["level"]
hourly          = result["hourly_cad"]
monthly_roi     = result["monthly_saving"]
wage_label      = result["wage_matched"] or "N/A"
occ_confidence  = result["occ_confidence"]
occ_method      = result["occ_match_method"]
wage_confidence = result["wage_confidence"]
wage_method     = result["wage_match_method"]
top_tasks       = result["top_tasks"]
low_tasks       = result["low_tasks"]

print(f"✅  {occ}  |  Risk {score_0_100}/100  |  {level}")
print(f"    Occupation match: {occ_confidence}% confidence  [{occ_method}]")
if hourly:
    annual_roi = monthly_roi * 12 if monthly_roi else None
    print(f"    Hourly wage: ${hourly:.2f} CAD  |  Monthly ROI: ${monthly_roi:,}  |  Annual ROI: ${annual_roi:,}")
    print(f"    Wage match: {wage_confidence}% confidence  [{wage_method}]")

# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_color(lvl):
    return RISK_COLORS.get(lvl, BLUE)

rc = risk_color(level)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         left=0.06, right=0.97,
                         top=0.88, bottom=0.10,
                         hspace=0.55, wspace=0.40)

fig.text(0.5, 0.95, f"Automation Risk Report  ·  {occ}",
         ha="center", va="center", fontsize=16, fontweight="bold", color=FG)
fig.text(0.5, 0.915, "Based on O*NET task analysis & StatCan wage data",
         ha="center", va="center", fontsize=10, color=SUBTEXT)

# ── Panel 1 — Gauge ───────────────────────────────────────────────────────────
ax_gauge = fig.add_subplot(gs[0, 0])
ax_gauge.set_facecolor(BG)
ax_gauge.set_aspect("equal")

for t_start, t_end, col in [(np.pi, np.pi*2/3, GREEN),
                              (np.pi*2/3, np.pi/3, ORANGE),
                              (np.pi/3, 0, CORAL)]:
    t = np.linspace(t_start, t_end, 100)
    ax_gauge.plot(np.cos(t), np.sin(t), color=col, lw=12, solid_capstyle="butt", alpha=0.85)

angle = np.pi * (1 - score_0_100 / 100)
ax_gauge.annotate("", xy=(0.72*np.cos(angle), 0.72*np.sin(angle)),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color=FG, lw=2, mutation_scale=14))
ax_gauge.plot(0, 0, "o", color=FG, ms=6, zorder=5)
ax_gauge.text(0, -0.22, f"{score_0_100}", ha="center", va="center",
              fontsize=30, fontweight="bold", color=rc)
ax_gauge.text(0, -0.46, "/ 100", ha="center", va="center", fontsize=11, color=SUBTEXT)
ax_gauge.text(0, -0.68, level, ha="center", va="center",
              fontsize=12, fontweight="bold", color=rc)
ax_gauge.set_xlim(-1.3, 1.3); ax_gauge.set_ylim(-0.85, 1.15); ax_gauge.axis("off")
ax_gauge.set_title("Automation Risk Score", color=FG, fontsize=11, pad=8)

# ── Panel 2 — ROI cards (with confidence %) ───────────────────────────────────
ax_roi = fig.add_subplot(gs[0, 1])
ax_roi.set_facecolor(BG); ax_roi.axis("off")
ax_roi.set_title("ROI Estimate", color=FG, fontsize=11, pad=8)

cards = [
    ("Matched Occupation",   f"{occ}  ({occ_confidence}% match)",           FG),
    ("Wage Data Match",      f"{wage_label}  ({wage_confidence}% match)",    SUBTEXT),
    ("Avg Hourly Wage",      f"${hourly:.2f} CAD" if hourly else "N/A",      YELLOW),
    ("Monthly Time Savings", f"${monthly_roi:,} CAD" if monthly_roi else "N/A", GREEN),
    ("Annual ROI Estimate",  f"${monthly_roi*12:,} CAD" if monthly_roi else "N/A", GREEN),
    ("Automation Assumption","40% of 160 hrs/mo automatable",                SUBTEXT),
]

y = 0.97
for label, value, col in cards:
    ax_roi.text(0, y, label, transform=ax_roi.transAxes,
                fontsize=8, color=SUBTEXT, va="top")
    y -= 0.11
    ax_roi.text(0, y, str(value), transform=ax_roi.transAxes,
                fontsize=10, fontweight="bold", color=col, va="top", wrap=True)
    y -= 0.08

# ── Panel 3 — Risk level guide ────────────────────────────────────────────────
ax_ctx = fig.add_subplot(gs[0, 2])
ax_ctx.set_facecolor(BG); ax_ctx.axis("off")
ax_ctx.set_title("Risk Level Guide", color=FG, fontsize=11, pad=8)

guide = [
    ("🟢  Low Risk",    "0 – 44",  "Creative, social, strategic tasks dominate. "
                                    "Automation potential is limited.",  GREEN),
    ("🟡  Medium Risk", "45 – 69", "Mix of routine and complex tasks. "
                                    "Partial automation likely.", ORANGE),
    ("🔴  High Risk",   "70 – 100","Routine, data-entry, or repetitive tasks. "
                                    "High automation potential.", CORAL),
]
y = 0.96
for title, rng, desc, col in guide:
    ax_ctx.text(0, y, f"{title}  ({rng})", transform=ax_ctx.transAxes,
                fontsize=9.5, fontweight="bold", color=col, va="top")
    y -= 0.10
    ax_ctx.text(0.03, y, desc, transform=ax_ctx.transAxes,
                fontsize=8, color=SUBTEXT, va="top", wrap=True)
    y -= 0.20

# ── Panel 4 — Task breakdown ──────────────────────────────────────────────────
ax_tasks = fig.add_subplot(gs[1, :])
ax_tasks.set_facecolor(BG)

all_tasks = pd.concat([top_tasks, low_tasks], ignore_index=True)
all_tasks = all_tasks.drop_duplicates(subset="task").copy()
all_tasks["score_pct"]  = (all_tasks["score"] * 100).round(1)
all_tasks["short_task"] = all_tasks["task"].str[:72].str.strip()
all_tasks = all_tasks.sort_values("score_pct", ascending=True)

bar_colors = [CORAL if s >= 70 else ORANGE if s >= 45 else GREEN
              for s in all_tasks["score_pct"]]

bars = ax_tasks.barh(all_tasks["short_task"], all_tasks["score_pct"],
                      color=bar_colors, height=0.6, edgecolor="none")

for bar, val in zip(bars, all_tasks["score_pct"]):
    ax_tasks.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                  f"{val:.0f}", va="center", ha="left",
                  fontsize=8.5, color=FG, fontweight="bold")

ax_tasks.axvline(score_0_100, color=YELLOW, lw=1.5, linestyle="--", alpha=0.8)
ax_tasks.text(score_0_100 + 1, -0.7, f"Avg {score_0_100}", color=YELLOW, fontsize=8, va="top")
ax_tasks.set_xlim(0, 115)
ax_tasks.set_xlabel("Automation Risk Score  (0 – 100)", color=SUBTEXT, fontsize=9)
ax_tasks.set_title("Task-Level Automation Risk Breakdown", color=FG,
                   fontsize=11, pad=8, loc="left")
ax_tasks.tick_params(axis="y", colors=FG, labelsize=8)
ax_tasks.tick_params(axis="x", colors=SUBTEXT, labelsize=8)
for spine in ax_tasks.spines.values():
    spine.set_visible(False)
ax_tasks.xaxis.grid(True, color=SUBTEXT, alpha=0.15, lw=0.6)
ax_tasks.set_axisbelow(True)

leg_patches = [mpatches.Patch(color=GREEN,  label="Low Risk  (< 45)"),
               mpatches.Patch(color=ORANGE, label="Medium Risk  (45–69)"),
               mpatches.Patch(color=CORAL,  label="High Risk  (≥ 70)")]
ax_tasks.legend(handles=leg_patches, loc="lower right",
                framealpha=0, labelcolor=FG, fontsize=8)

plt.savefig("automation_risk_report.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("Chart saved → automation_risk_report.png")
