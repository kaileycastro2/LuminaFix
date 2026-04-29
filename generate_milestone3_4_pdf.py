from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)

OUTPUT_PATH = "/home/shreyank06/Desktop/projects/upwork_projects/style_transfer/Milestone_3_4_Todo.pdf"

PRIMARY = HexColor("#1a237e")
ACCENT = HexColor("#0d47a1")
TEXT_COLOR = HexColor("#212121")
SUBTLE = HexColor("#616161")
TABLE_HEADER_BG = HexColor("#1a237e")
TABLE_ALT_ROW = HexColor("#f5f5f5")
BORDER_COLOR = HexColor("#bbdefb")
CHECK_GREEN = HexColor("#2e7d32")
WARN_RED = HexColor("#c62828")
DONE_BG = HexColor("#e8f5e9")
M3_COLOR = HexColor("#1565c0")
M4_COLOR = HexColor("#6a1b9a")


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="DocTitle", fontSize=22, leading=28, textColor=PRIMARY, fontName="Helvetica-Bold", spaceAfter=4))
    styles.add(ParagraphStyle(name="DocSubtitle", fontSize=11, leading=16, textColor=SUBTLE, fontName="Helvetica", spaceAfter=16))
    styles.add(ParagraphStyle(name="SectionHead", fontSize=16, leading=22, textColor=PRIMARY, fontName="Helvetica-Bold", spaceBefore=20, spaceAfter=8))
    styles.add(ParagraphStyle(name="SubHead", fontSize=12, leading=17, textColor=ACCENT, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=15, textColor=TEXT_COLOR, fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=4))
    styles.add(ParagraphStyle(name="BulletC", fontSize=10, leading=15, textColor=TEXT_COLOR, fontName="Helvetica", leftIndent=16, spaceAfter=3, bulletIndent=4))
    styles.add(ParagraphStyle(name="CodeBlk", fontSize=9, leading=13, textColor=HexColor("#1b5e20"), fontName="Courier", backColor=HexColor("#f1f8e9"), borderColor=HexColor("#c5e1a5"), borderWidth=0.5, borderPadding=6, leftIndent=8, spaceAfter=8, spaceBefore=4))
    styles.add(ParagraphStyle(name="Highlight", fontSize=10, leading=15, textColor=PRIMARY, fontName="Helvetica-BoldOblique", backColor=HexColor("#e8eaf6"), borderColor=HexColor("#7986cb"), borderWidth=1, borderPadding=10, spaceBefore=8, spaceAfter=8))
    styles.add(ParagraphStyle(name="Warning", fontSize=10, leading=15, textColor=WARN_RED, fontName="Helvetica-Bold", backColor=HexColor("#ffebee"), borderColor=HexColor("#ef9a9a"), borderWidth=1, borderPadding=10, spaceBefore=8, spaceAfter=8))
    styles.add(ParagraphStyle(name="M3Head", fontSize=14, leading=20, textColor=M3_COLOR, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="M4Head", fontSize=14, leading=20, textColor=M4_COLOR, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Source", fontSize=8, leading=11, textColor=SUBTLE, fontName="Helvetica-Oblique", spaceBefore=6))
    return styles


def divider():
    return HRFlowable(width="100%", thickness=1, color=BORDER_COLOR, spaceBefore=4, spaceAfter=4)


def make_table(headers, rows, col_widths=None):
    hcells = [Paragraph(h, ParagraphStyle("h", fontSize=9.5, textColor=white, fontName="Helvetica-Bold", alignment=TA_CENTER, leading=13)) for h in headers]
    bcells = [[Paragraph(c, ParagraphStyle("c", fontSize=9, textColor=TEXT_COLOR, fontName="Helvetica", leading=13)) for c in r] for r in rows]
    data = [hcells] + bcells
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_ROW))
    t.setStyle(TableStyle(cmds))
    return t


def b(text, styles):
    return Paragraph(f"<bullet>&bull;</bullet> {text}", styles["BulletC"])


def checkbox(text, styles, checked=False):
    mark = "<font color='#2e7d32'>[x]</font>" if checked else "<font color='#0d47a1'>[ ]</font>"
    return Paragraph(f"{mark}  {text}", styles["Body"])


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(PRIMARY)
    canvas.setLineWidth(2)
    canvas.line(40, A4[1] - 36, A4[0] - 40, A4[1] - 36)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SUBTLE)
    canvas.drawString(40, A4[1] - 30, "LuminaFix — Milestone 3 & 4 Todo")
    canvas.setStrokeColor(BORDER_COLOR)
    canvas.setLineWidth(0.5)
    canvas.line(40, 40, A4[0] - 40, 40)
    canvas.drawString(40, 28, "LuminaFix Style Transfer Project")
    canvas.drawRightString(A4[0] - 40, 28, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf():
    s = build_styles()
    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=A4, leftMargin=45, rightMargin=45, topMargin=55, bottomMargin=55)
    story = []

    # ── TITLE ──
    story.append(Paragraph("Milestone 3 &amp; 4 — Todo List", s["DocTitle"]))
    story.append(Paragraph("Color pairing improvements &amp; future enhancements for NILUT style transfer", s["DocSubtitle"]))
    story.append(divider())

    # ══════════════════════════════════════════════════════════════
    # MILESTONE 3: Smart Color Pairing
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("MILESTONE 3: Smart Color Pairing", s["SectionHead"]))
    story.append(Paragraph(
        "Replace blind percentile-based color pairing with nearest-neighbor matching + distance threshold. "
        "This is the core quality improvement for style transfer accuracy.",
        s["Body"]
    ))
    story.append(Spacer(1, 6))

    # ── Current vs New (Side by Side) ──
    story.append(Paragraph("Current vs New Training Logic", s["SubHead"]))

    story.append(make_table(
        ["Step", "Current (Blind Percentile Sort)", "New (Nearest-Neighbor Match)"],
        [
            ["1", "Take 50K color samples from content", "Take 50K color samples from content"],
            ["2", "Take 50K color samples from reference", "Take 50K color samples from reference"],
            ["3", "Sort BOTH by percentile", "For each content color, find NEAREST reference color"],
            ["4", "Pair them 1:1 by sorted position", "If distance small &#8594; pair with reference color\nIf distance large &#8594; pair with ITSELF"],
            ["5", "Train: input=content &#8594; output=ref", "Train: input=content &#8594; output=paired_color"],
        ],
        col_widths=[35, 200, 255],
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        '<b>Key difference:</b> Steps 3-4 change. Instead of blind percentile sorting, use nearest-neighbor matching '
        'with a distance threshold. Colors without a close match in the reference learn to stay as-is.',
        s["Highlight"]
    ))

    # ── Why This Matters ──
    story.append(Paragraph("Why This Matters", s["SubHead"]))
    story.append(make_table(
        ["Problem with Current", "How New Fixes It"],
        [
            ["Sky blue forced to map to skin tone (wrong match)", "Sky blue finds nearest blue in reference, or stays as-is"],
            ["Every color MUST change, even if no good match exists", "Colors without a close match keep their original value"],
            ["Percentile position ≠ color similarity", "Actual color distance (Euclidean in Lab space) used"],
            ["Produces color artifacts in mismatched regions", "Preserves original colors where style doesn't apply"],
        ],
        col_widths=[245, 245],
    ))

    # ── Todo Items for M3 ──
    story.append(PageBreak())
    story.append(Paragraph("Milestone 3 — Todo Checklist", s["M3Head"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>3.1  Implement Nearest-Neighbor Pairing</b>", s["SubHead"]))
    story.append(checkbox("Replace histogram/percentile sort with KDTree nearest-neighbor lookup", s))
    story.append(checkbox("Use scipy.spatial.KDTree or sklearn.neighbors.BallTree for fast lookup", s))
    story.append(checkbox("Match in Lab color space (A,B channels) for perceptual accuracy", s))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Pseudocode:", s["Body"]))
    story.append(Paragraph(
        "from scipy.spatial import KDTree<br/><br/>"
        "content_colors = sample_50k(content_images)  # shape: (50000, 2)<br/>"
        "ref_colors = sample_50k(ref_images)           # shape: (50000, 2)<br/><br/>"
        "tree = KDTree(ref_colors)<br/>"
        "distances, indices = tree.query(content_colors)<br/><br/>"
        "threshold = 15.0  # Lab units — tune this<br/><br/>"
        "paired = []<br/>"
        "for i, (dist, idx) in enumerate(zip(distances, indices)):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;if dist &lt; threshold:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;paired.append((content_colors[i], ref_colors[idx]))<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;else:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;paired.append((content_colors[i], content_colors[i]))  # keep original",
        s["CodeBlk"]
    ))

    story.append(Paragraph("<b>3.2  Add Distance Threshold Parameter</b>", s["SubHead"]))
    story.append(checkbox("Add configurable threshold (default ~15 Lab units)", s))
    story.append(checkbox("Lower threshold = more conservative (fewer changes, more originals kept)", s))
    story.append(checkbox("Higher threshold = more aggressive (more colors remapped)", s))
    story.append(Spacer(1, 4))

    story.append(make_table(
        ["Threshold", "Behavior", "Use Case"],
        [
            ["5-10", "Very conservative — only near-exact matches", "Subtle color grading"],
            ["10-20", "Balanced — good matches only", "General style transfer (recommended)"],
            ["20-30", "Aggressive — loose matches allowed", "Strong style application"],
            ["999+", "Everything matches (same as current behavior)", "Backwards compatibility"],
        ],
        col_widths=[80, 210, 200],
    ))

    story.append(Paragraph("<b>3.3  Update train.py</b>", s["SubHead"]))
    story.append(checkbox("Modify the color pairing function in train.py", s))
    story.append(checkbox("Add threshold as a CLI argument or config parameter", s))
    story.append(checkbox("Log stats: how many pairs matched vs kept-as-is (for debugging)", s))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "Example log output:<br/>"
        "Style: moody_cinematic<br/>"
        "  Total pairs: 50,000<br/>"
        "  Matched (dist &lt; 15): 32,450 (64.9%)<br/>"
        "  Kept original (dist &gt;= 15): 17,550 (35.1%)<br/>"
        "  Avg match distance: 8.3 Lab units",
        s["CodeBlk"]
    ))

    story.append(Paragraph("<b>3.4  Retrain &amp; Validate</b>", s["SubHead"]))
    story.append(checkbox("Retrain all 5 per-style models + universal with new pairing", s))
    story.append(checkbox("Compare old vs new on validation images (side-by-side)", s))
    story.append(checkbox("Check: sky stays blue, skin stays natural, only style-relevant colors change", s))
    story.append(checkbox("Tune threshold if needed based on visual results", s))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        '<b>Success criteria:</b> Style transfer applies style colors where appropriate, '
        'but preserves original colors in regions where the reference has no good match. '
        'No more forced color artifacts.',
        s["Highlight"]
    ))

    # ══════════════════════════════════════════════════════════════
    # MILESTONE 4: Future Enhancements (if time permits)
    # ══════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("MILESTONE 4: Future Enhancements (If Time Permits)", s["SectionHead"]))
    story.append(Paragraph(
        "These are stretch goals that build on top of Milestone 3. Only tackle these after M3 is solid.",
        s["Body"]
    ))
    story.append(Spacer(1, 6))

    # 4.1
    story.append(Paragraph("4.1  Adaptive Threshold Per Style", s["M4Head"]))
    story.append(checkbox("Instead of one fixed threshold, compute per-style threshold automatically", s))
    story.append(checkbox("Based on the color distribution overlap between content and reference", s))
    story.append(b("Styles with narrow color palettes (e.g., moody cinematic) get tighter thresholds", s))
    story.append(b("Styles with broad palettes (e.g., warm vintage) get wider thresholds", s))
    story.append(Spacer(1, 6))

    # 4.2
    story.append(Paragraph("4.2  Weighted Pairing by Distance", s["M4Head"]))
    story.append(checkbox("Instead of binary (match / keep original), use soft blending", s))
    story.append(checkbox("Close match = mostly reference color, far match = mostly original", s))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "weight = max(0, 1 - distance / threshold)<br/>"
        "paired_color = weight * ref_color + (1 - weight) * content_color",
        s["CodeBlk"]
    ))
    story.append(b("Produces smoother transitions instead of hard cutoff", s))
    story.append(Spacer(1, 6))

    # 4.3
    story.append(Paragraph("4.3  Multi-Channel Matching (L + A + B)", s["M4Head"]))
    story.append(checkbox("Currently matching on A,B channels only (chrominance)", s))
    story.append(checkbox("Optionally include L channel (luminance) for brightness-aware matching", s))
    story.append(b("Prevents dark shadows from being paired with bright highlights of same hue", s))
    story.append(Spacer(1, 6))

    # 4.4
    story.append(Paragraph("4.4  Intensity/Strength Slider in Web UI", s["M4Head"]))
    story.append(checkbox("Add a slider (0-100%) that controls how much style is applied", s))
    story.append(checkbox("Maps to the distance threshold internally", s))
    story.append(b("0% = no change (threshold=0, everything keeps original)", s))
    story.append(b("100% = full style (threshold=999, same as current behavior)", s))
    story.append(b("50% = balanced (threshold=15, recommended default)", s))
    story.append(Spacer(1, 6))

    # 4.5
    story.append(Paragraph("4.5  Batch Processing &amp; Progress Feedback", s["M4Head"]))
    story.append(checkbox("Process multiple images in one upload", s))
    story.append(checkbox("Show progress bar during style transfer", s))
    story.append(checkbox("Download all results as ZIP", s))

    # ── Summary ──
    story.append(PageBreak())
    story.append(Paragraph("Summary", s["SectionHead"]))

    story.append(make_table(
        ["Milestone", "Focus", "Priority", "Status"],
        [
            ["M3", "Smart color pairing (nearest-neighbor + threshold)", "HIGH — Core quality fix", "TODO"],
            ["M4.1", "Adaptive threshold per style", "Medium", "Stretch"],
            ["M4.2", "Weighted/soft pairing by distance", "Medium", "Stretch"],
            ["M4.3", "Multi-channel matching (L+A+B)", "Low", "Stretch"],
            ["M4.4", "Intensity slider in web UI", "Medium", "Stretch"],
            ["M4.5", "Batch processing + progress bar", "Low", "Stretch"],
        ],
        col_widths=[45, 225, 120, 60],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        '<b>Bottom line:</b> Milestone 3 is the one change that matters most — replacing blind percentile sort '
        'with nearest-neighbor matching. Everything in M4 is a nice-to-have that builds on top of it.',
        s["Highlight"]
    ))

    story.append(Spacer(1, 8))
    story.append(divider())
    story.append(Paragraph(
        '<b>Quick reference — The only code change in M3:</b><br/><br/>'
        'OLD: Sort both color arrays by percentile &#8594; pair 1:1 by position<br/>'
        'NEW: KDTree(ref_colors).query(content_colors) &#8594; pair if distance &lt; threshold, else keep original',
        s["Highlight"]
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
