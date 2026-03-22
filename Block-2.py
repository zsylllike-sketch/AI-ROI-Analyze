import pandas as pd
import re

# ── 1. OCCUPATIONS ─────────────────────────────────────────────
# Keep only the SOC code and title — drop verbose description
clean_occupations = occupations[["O*NET-SOC Code", "Title"]].copy()
clean_occupations.columns = ["onet_code", "occupation_title"]
clean_occupations["onet_code"] = clean_occupations["onet_code"].str.strip()

# ── 2. TASKS ───────────────────────────────────────────────────
# Drop: Task ID (internal), Incumbents Responding (survey meta),
#       Date, Domain Source — keep the meaningful content only
clean_tasks = tasks[["O*NET-SOC Code", "Task", "Task Type"]].copy()
clean_tasks.columns = ["onet_code", "task_description", "task_type"]
clean_tasks["onet_code"] = clean_tasks["onet_code"].str.strip()

# ── 3. WAGES (already latest Canada-only from Block 1) ─────────
# Extract NOC code from the label, e.g. "... [1234]" → "1234"
clean_wages = wages.copy()
clean_wages.columns = ["noc_label", "avg_hourly_wage"]
clean_wages["noc_code"] = clean_wages["noc_label"].str.extract(r"\[(\d+)\]$")

# Drop rows with no NOC code (aggregate rows like "Total, all occupations")
clean_wages = clean_wages.dropna(subset=["noc_code"])
clean_wages = clean_wages[["noc_code", "noc_label", "avg_hourly_wage"]].reset_index(drop=True)

# Clean up the label — strip trailing code bracket
clean_wages["noc_label"] = clean_wages["noc_label"].str.replace(r"\s*\[[\w\s]+\]$", "", regex=True).str.strip()

# ── Summary ────────────────────────────────────────────────────
print("=== clean_occupations ===")
print(f"  Shape : {clean_occupations.shape}")
print(clean_occupations.head(3).to_string(index=False))

print("\n=== clean_tasks ===")
print(f"  Shape : {clean_tasks.shape}")
print(clean_tasks.head(3).to_string(index=False))

print("\n=== clean_wages ===")
print(f"  Shape : {clean_wages.shape}")
print(clean_wages.head(3).to_string(index=False))
print(f"\n  Wage range : ${clean_wages['avg_hourly_wage'].min():.2f} – ${clean_wages['avg_hourly_wage'].max():.2f} /hr")
print(f"  Null wages : {clean_wages['avg_hourly_wage'].isnull().sum()}")