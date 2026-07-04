"""Render report/report.md to report/report.pdf using reportlab.

This is a focused Markdown-to-PDF converter that supports exactly the constructs
used in the report: headings, paragraphs, bold/italic/inline-code, bullet lists,
a pipe table, horizontal rules, and an appendix of the generated figures.
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "report.md"
REPORT_PDF = ROOT / "report" / "report.pdf"
FIGURES_DIR = ROOT / "reports" / "figures"

# Figures appended as an appendix, with human-readable captions.
FIGURE_CAPTIONS = [
    ("class_balance.png", "Class imbalance: fraud is 0.17% of all transactions."),
    ("amount_distribution.png", "Transaction Amount before and after a log1p transform."),
    ("fraud_rate_by_hour.png", "Transaction volume and fraud rate by hour of day."),
    ("correlation_with_target.png", "Top features by |Spearman correlation| with the fraud label."),
    ("pr_curves.png", "Precision-Recall curves (corrected pipeline)."),
    ("roc_curves.png", "ROC curves (corrected pipeline) - all look excellent."),
    ("confusion_rf.png", "Confusion matrix of the corrected Random Forest."),
    ("cost_curve.png", "Cost-sensitive threshold selection (FN = 100 x FP)."),
]


def make_styles():
    styles = getSampleStyleSheet()
    styles["Title"].fontSize = 18
    styles["Title"].spaceAfter = 14
    styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle("BodyBullet", parent=styles["Body"], leftIndent=14, bulletIndent=4, spaceAfter=2))
    styles.add(ParagraphStyle("Caption", parent=styles["Body"], fontSize=9, textColor=colors.grey, alignment=1, spaceBefore=2))
    styles.add(ParagraphStyle("Cell", parent=styles["Body"], fontSize=8, alignment=1, spaceAfter=0))
    return styles


def inline(text: str) -> str:
    """Convert inline Markdown to reportlab mini-markup, escaping XML first."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def build_table(rows, styles):
    data = [[Paragraph(inline(c), styles["Cell"]) for c in row] for row in rows]
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def parse_markdown(md_text, styles):
    flow = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line:
            i += 1
            continue

        # Horizontal rule.
        if re.fullmatch(r"-{3,}", line):
            flow.append(Spacer(1, 6))
            i += 1
            continue

        # Headings.
        if line.startswith("# "):
            flow.append(Paragraph(inline(line[2:]), styles["Title"]))
            i += 1
            continue
        if line.startswith("## "):
            flow.append(Paragraph(inline(line[3:]), styles["H1c"]))
            i += 1
            continue
        if line.startswith("### "):
            flow.append(Paragraph(inline(line[4:]), styles["H2c"]))
            i += 1
            continue

        # Pipe table: a header line, a separator line, then body rows.
        if line.startswith("|") and i + 1 < len(lines) and re.search(r"\|\s*-{2,}", lines[i + 1]):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = []
            for row_idx, raw in enumerate(block):
                if row_idx == 1:  # skip the |---|---| separator
                    continue
                cells = [c.strip() for c in raw.strip().strip("|").split("|")]
                rows.append(cells)
            flow.append(Spacer(1, 4))
            flow.append(build_table(rows, styles))
            flow.append(Spacer(1, 6))
            continue

        # Bullet list. A single item may wrap across several indented lines.
        if line.lstrip().startswith("- "):
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                item = [lines[i].lstrip()[2:]]
                i += 1
                # Absorb indented continuation lines that belong to this bullet.
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not lines[i].lstrip().startswith("- ")
                    and lines[i][0] in (" ", "\t")
                ):
                    item.append(lines[i].strip())
                    i += 1
                flow.append(Paragraph(inline(" ".join(item)), styles["BodyBullet"], bulletText="•"))
            continue

        # Paragraph (gather until blank line).
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||-{3,}|- )", lines[i].lstrip()):
            para.append(lines[i].rstrip())
            i += 1
        flow.append(Paragraph(inline(" ".join(para)), styles["Body"]))

    return flow


def append_figures(flow, styles):
    available = [(f, c) for f, c in FIGURE_CAPTIONS if (FIGURES_DIR / f).exists()]
    if not available:
        return
    flow.append(PageBreak())
    flow.append(Paragraph("Appendix: Figures", styles["H1c"]))
    for fname, caption in available:
        img = Image(str(FIGURES_DIR / fname))
        max_w = 15 * cm
        if img.imageWidth > max_w:
            scale = max_w / img.imageWidth
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
        img.hAlign = "CENTER"
        flow.append(Spacer(1, 8))
        flow.append(img)
        flow.append(Paragraph(caption, styles["Caption"]))


def main():
    if not REPORT_MD.exists():
        sys.exit(f"Missing {REPORT_MD}")
    styles = make_styles()
    flow = parse_markdown(REPORT_MD.read_text(), styles)
    append_figures(flow, styles)
    doc = SimpleDocTemplate(
        str(REPORT_PDF), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Critical Review of a Credit-Card Fraud Detection Tutorial",
    )
    doc.build(flow)
    print(f"Wrote {REPORT_PDF} ({REPORT_PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
