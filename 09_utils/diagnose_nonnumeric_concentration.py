"""
Diagnostic: scan the curve files referenced by the master index and report
every UNIQUE non-numeric value found in the concentration column, with
which file(s) it came from. Doesn't compute anything - just surfaces what
string patterns we need to agree on a parsing rule for before re-running
nca_calc.py.

Usage:
    python3 diagnose_nonnumeric.py \
        --master pk_curve_level_master_index_filled.csv \
        --curve-root /Users/xxy/Documents/MML \
        --out nonnumeric_report.csv
"""
import argparse
import os
import re
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--curve-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    master = pd.read_csv(args.master)
    # only look at rows that previously errored / are worth re-checking
    target = master[master["nca_flag"].astype(str).str.startswith("error")] \
        if "nca_flag" in master.columns else master

    rows = []
    for _, row in target.iterrows():
        full_path = os.path.join(args.curve_root, row["sub_csv_path"])
        if not os.path.exists(full_path):
            continue
        d = pd.read_csv(full_path)
        conc_col = d.columns[1]
        for v in d[conc_col].dropna().unique():
            try:
                float(v)
            except (ValueError, TypeError):
                rows.append(dict(sub_csv_path=row["sub_csv_path"],
                                   raw_value=v))

    report = pd.DataFrame(rows)
    report.to_csv(args.out, index=False)

    print(f"Found {len(report)} non-numeric cell instances across "
          f"{report['sub_csv_path'].nunique() if len(report) else 0} files.")
    print("\nUnique value patterns and counts:")
    if len(report):
        print(report["raw_value"].value_counts().to_string())
    print(f"\nFull detail written to {args.out}")


if __name__ == "__main__":
    main()
