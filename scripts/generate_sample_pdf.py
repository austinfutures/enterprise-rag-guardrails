"""Regenerates data/corvanta_fy2025_annual_financial_summary.pdf (fictional company).

Usage:  python scripts/generate_sample_pdf.py      (requires: pip install reportlab)
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "corvanta_fy2025_annual_financial_summary.pdf"

PAGES = [
    (
        "CORVANTA INDUSTRIES FY2025 ANNUAL FINANCIAL SUMMARY",
        [
            "This summary covers Corvanta Industries' fiscal year 2025, which ended on December 31, 2025. All figures are in US dollars and are unaudited.",
            "FINANCIAL HIGHLIGHTS",
            "Total revenue for FY2025 was $4.82 billion, an increase of 11.3 percent from $4.33 billion in FY2024. Gross margin improved to 58.4 percent, compared with 56.1 percent in the prior year.",
            "Operating income was $912 million, representing an operating margin of 18.9 percent. Net income was $684 million and diluted earnings per share were $3.42.",
            "Revenue growth was driven primarily by expansion of the Cloud Services segment and higher renewal rates among enterprise customers.",
        ],
    ),
    (
        "SEGMENT AND REGIONAL PERFORMANCE",
        [
            "Cloud Services revenue was $2.31 billion, an increase of 19.7 percent year over year. Cloud Services is now the largest segment and accounts for 48 percent of total revenue.",
            "Enterprise Software revenue was $1.64 billion, an increase of 6.2 percent. Professional Services revenue was $0.87 billion, an increase of 1.1 percent.",
            "By geography, North America contributed 61 percent of revenue, EMEA contributed 27 percent and APAC contributed 12 percent.",
            "The number of customers with more than $1 million in annual recurring revenue grew to 412 at the end of FY2025.",
        ],
    ),
    (
        "CASH FLOW AND CAPITAL ALLOCATION",
        [
            "Operating cash flow was $1.21 billion and free cash flow was $968 million. Capital expenditures were $242 million, focused on data center capacity.",
            "Cash and cash equivalents were $2.06 billion at year end, and total debt was $1.15 billion.",
            "During FY2025 Corvanta repurchased $400 million of its common stock. The Board declared a quarterly dividend of $0.28 per share.",
            "FY2026 OUTLOOK",
            "For FY2026 Corvanta expects total revenue between $5.25 billion and $5.35 billion. The company expects an operating margin between 19.5 percent and 20.5 percent.",
            "Key risks to the outlook include foreign exchange volatility, increased competition from AI-native software vendors and slower enterprise IT spending.",
        ],
    ),
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        for p in paras:
            style = styles["Heading2"] if p.isupper() else styles["BodyText"]
            story += [Paragraph(p, style), Spacer(1, 8)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=LETTER, title="Corvanta FY2025 Annual Financial Summary").build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
