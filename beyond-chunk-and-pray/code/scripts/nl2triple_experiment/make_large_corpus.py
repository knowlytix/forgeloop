# SPDX-License-Identifier: Apache-2.0
"""Generate a larger, internally-consistent annual-report corpus for the stress
test: same table schemas the regex ingester already parses (so build_store works
unchanged), but ~4x the facts and one extra fiscal-year relation (has_fy2023).

6 divisions, 18 segments, an 8-line income statement across 3 years, an 8-line
balance sheet, and 8 corporate facts. Totals are summed so GeodeLoop's anchor and
ENM checks stay consistent. Writes data/annual_report_large.md.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO_ROOT, "data", "annual_report_large.md")

DIVISIONS = [
    ("Technology", "North America", "Dana Cole"),
    ("Operations", "Europe", "Sam Reyes"),
    ("Consumer", "North America", "Priya Nair"),
    ("Healthcare", "Asia Pacific", "Elena Ruiz"),
    ("Financial Services", "Europe", "Marcus Feld"),
    ("Energy", "Latin America", "Tomas Vega"),
]

# (segment, division, revenue, headcount) -- distinct values, no collisions.
SEGMENTS = [
    ("Cloud Platform", "Technology", 120.0, 340),
    ("Devices", "Technology", 80.0, 210),
    ("Data Services", "Technology", 65.0, 150),
    ("Logistics", "Operations", 95.0, 430),
    ("Retail", "Operations", 60.0, 520),
    ("Fulfillment", "Operations", 48.0, 275),
    ("Home Goods", "Consumer", 72.0, 390),
    ("Apparel", "Consumer", 54.0, 310),
    ("Grocery", "Consumer", 88.0, 610),
    ("Diagnostics", "Healthcare", 105.0, 260),
    ("Medical Devices", "Healthcare", 77.0, 185),
    ("Care Services", "Healthcare", 41.0, 205),
    ("Payments", "Financial Services", 133.0, 175),
    ("Lending", "Financial Services", 69.0, 140),
    ("Insurance", "Financial Services", 58.0, 165),
    ("Renewables", "Energy", 91.0, 230),
    ("Grid", "Energy", 64.0, 195),
    ("Storage", "Energy", 37.0, 110),
]

# Income statement, three years (FY2023..FY2025). Revenue FY2025 == segment total.
INCOME = [
    ("Revenue", None, None, None),        # filled from segment total below
    ("Cost of Revenue", 0.60, 0.62, 0.64),
    ("Gross Profit", 0.40, 0.38, 0.36),
    ("Operating Expenses", 0.20, 0.21, 0.22),
    ("Operating Income", 0.20, 0.17, 0.14),
    ("Interest Expense", 0.02, 0.02, 0.03),
    ("Pretax Income", 0.18, 0.15, 0.11),
    ("Net Income", 0.13, 0.11, 0.08),
]

BALANCE = [
    ("Total Assets", 1840.0),
    ("Current Assets", 720.0),
    ("Cash and Equivalents", 305.0),
    ("Total Liabilities", 910.0),
    ("Current Liabilities", 388.0),
    ("Long Term Debt", 402.0),
    ("Shareholders Equity", 930.0),
    ("Retained Earnings", 615.0),
]

CORPORATE = [
    ("CEO", "Morgan Lee"),
    ("CFO", "Jordan Pratt"),
    ("COO", "Alex Kim"),
    ("Chief Technology Officer", "Rowan Patel"),
    ("General Counsel", "Sofia Marin"),
    ("Auditor", "Pinecrest LLP"),
    ("Headquarters", "Seattle"),
    ("Fiscal Year End", "June 30"),
]


def main():
    rev_total = round(sum(s[2] for s in SEGMENTS), 1)
    hc_total = sum(s[3] for s in SEGMENTS)
    rev = {"FY2025": rev_total, "FY2024": round(rev_total * 0.90, 1),
           "FY2023": round(rev_total * 0.82, 1)}

    L = []
    L.append("# Northwind Industries — Annual Report FY2025\n")
    L.append("Northwind Industries is a diversified company spanning technology, "
             "operations, consumer, healthcare, financial services and energy. "
             "This report summarizes fiscal 2025 (ended June 30, 2025).\n")

    L.append("## 1. Segment Performance\n")
    L.append("Revenue in millions of US dollars; headcount in employees. The Total "
             "row is the sum of the reportable segments.\n")
    L.append("| Segment | Division | Revenue | Headcount |")
    L.append("| :--- | :--- | ---: | ---: |")
    for seg, div, r, hc in SEGMENTS:
        L.append(f"| {seg} | {div} | {r} | {hc} |")
    L.append(f"| Total | All | {rev_total} | {hc_total} |\n")

    L.append("## 2. Divisions\n")
    L.append("Each division operates in a primary region under a division head.\n")
    L.append("| Division | Region | Head |")
    L.append("| :--- | :--- | :--- |")
    for div, region, head in DIVISIONS:
        L.append(f"| {div} | {region} | {head} |")
    L.append("")

    L.append("## 3. Income Statement\n")
    L.append("In millions of US dollars.\n")
    L.append("| Line Item | FY2025 | FY2024 | FY2023 |")
    L.append("| :--- | ---: | ---: | ---: |")
    for name, f25, f24, f23 in INCOME:
        if name == "Revenue":
            L.append(f"| Revenue | {rev['FY2025']} | {rev['FY2024']} | {rev['FY2023']} |")
        else:
            v25 = round(rev["FY2025"] * f25, 1)
            v24 = round(rev["FY2024"] * f24, 1)
            v23 = round(rev["FY2023"] * f23, 1)
            L.append(f"| {name} | {v25} | {v24} | {v23} |")
    L.append("")

    L.append("## 4. Balance Sheet\n")
    L.append("In millions of US dollars, as of June 30, 2025.\n")
    L.append("| Line Item | Amount |")
    L.append("| :--- | ---: |")
    for name, amt in BALANCE:
        L.append(f"| {name} | {amt} |")
    L.append("")

    L.append("## 5. Corporate Facts\n")
    L.append("| Attribute | Value |")
    L.append("| :--- | :--- |")
    for attr, val in CORPORATE:
        L.append(f"| {attr} | {val} |")
    L.append("")

    L.append("## 6. Management Discussion and Analysis\n")
    L.append("Fiscal 2025 reflected broad-based growth across the segment portfolio, "
             "with Payments and Cloud Platform leading consolidated revenue and the "
             "Healthcare division improving operating leverage. The figures above are "
             "authoritative; this narrative is qualitative.\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    n_facts = (len(SEGMENTS) * 3 + 3 + len(DIVISIONS) * 2 + len(INCOME) * 3
               + len(BALANCE) + len(CORPORATE))
    print(f"wrote {OUT}")
    print(f"  {len(SEGMENTS)} segments, {len(DIVISIONS)} divisions, "
          f"~{n_facts} content facts, relations incl. has_fy2023")


if __name__ == "__main__":
    main()
