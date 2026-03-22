import pandas as pd

# ── Load ───────────────────────────────────────────────────────
occupations = pd.read_csv("Data/Occupation Data.txt", sep="\t")
tasks       = pd.read_csv("Data/Task Statements.txt", sep="\t")
statcan     = pd.read_csv("Data/14100444.csv")

wages = statcan[
    (statcan["Statistics"] == "Average offered hourly wage") &
    (statcan["REF_DATE"] == statcan["REF_DATE"].max()) &
    (statcan["GEO"] == "Canada")
][["National Occupational Classification", "VALUE"]].dropna()
wages["VALUE"] = pd.to_numeric(wages["VALUE"], errors="coerce")

# ── Baseline stats ─────────────────────────────────────────────
for name, df in [("Occupations", occupations), ("Tasks", tasks), ("StatCan Wages", wages)]:
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Rows × Cols  : {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Column types : {df.dtypes.value_counts().to_dict()}")
    print(f"  Missing vals : {df.isnull().sum().sum():,} cells")
    for col in df.columns:
        print(f"    {col:<45} unique={df[col].nunique():<8} null={df[col].isnull().mean()*100:.1f}%")

print(f"\nOccupations: {len(occupations)}")
print(f"Tasks: {len(tasks)}")
print(f"Wage records: {len(wages)}")
print(occupations.head())
