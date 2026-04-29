from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)

OUTPUT_PATH = "/home/shreyank06/Desktop/projects/upwork_projects/style_transfer/NILUT_Training_Todo.pdf"

PRIMARY = HexColor("#1a237e")
ACCENT = HexColor("#0d47a1")
TEXT_COLOR = HexColor("#212121")
SUBTLE = HexColor("#616161")
TABLE_HEADER_BG = HexColor("#1a237e")
TABLE_ALT_ROW = HexColor("#f5f5f5")
BORDER_COLOR = HexColor("#bbdefb")
CHECK_GREEN = HexColor("#2e7d32")
WARN_RED = HexColor("#c62828")
WARN_ORANGE = HexColor("#e65100")


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="DocTitle", fontSize=24, leading=30, textColor=PRIMARY, fontName="Helvetica-Bold", spaceAfter=4))
    styles.add(ParagraphStyle(name="DocSubtitle", fontSize=11, leading=16, textColor=SUBTLE, fontName="Helvetica", spaceAfter=16))
    styles.add(ParagraphStyle(name="SectionHead", fontSize=16, leading=22, textColor=PRIMARY, fontName="Helvetica-Bold", spaceBefore=20, spaceAfter=8))
    styles.add(ParagraphStyle(name="SubHead", fontSize=12, leading=17, textColor=ACCENT, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=15, textColor=TEXT_COLOR, fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=4))
    styles.add(ParagraphStyle(name="BulletC", fontSize=10, leading=15, textColor=TEXT_COLOR, fontName="Helvetica", leftIndent=16, spaceAfter=3, bulletIndent=4))
    styles.add(ParagraphStyle(name="CodeBlk", fontSize=9, leading=13, textColor=HexColor("#1b5e20"), fontName="Courier", backColor=HexColor("#f1f8e9"), borderColor=HexColor("#c5e1a5"), borderWidth=0.5, borderPadding=6, leftIndent=8, spaceAfter=8, spaceBefore=4))
    styles.add(ParagraphStyle(name="Highlight", fontSize=10, leading=15, textColor=PRIMARY, fontName="Helvetica-BoldOblique", backColor=HexColor("#e8eaf6"), borderColor=HexColor("#7986cb"), borderWidth=1, borderPadding=10, spaceBefore=8, spaceAfter=8))
    styles.add(ParagraphStyle(name="Warning", fontSize=10, leading=15, textColor=WARN_RED, fontName="Helvetica-Bold", backColor=HexColor("#ffebee"), borderColor=HexColor("#ef9a9a"), borderWidth=1, borderPadding=10, spaceBefore=8, spaceAfter=8))
    styles.add(ParagraphStyle(name="TodoItem", fontSize=10.5, leading=16, textColor=TEXT_COLOR, fontName="Helvetica", leftIndent=20, spaceAfter=4))
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


def todo(num, text, styles):
    return Paragraph(f"<font color='#0d47a1'><b>[ ]</b></font>  <b>Step {num}:</b> {text}", styles["TodoItem"])


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(PRIMARY)
    canvas.setLineWidth(2)
    canvas.line(40, A4[1] - 36, A4[0] - 40, A4[1] - 36)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SUBTLE)
    canvas.drawString(40, A4[1] - 30, "NILUT Style Transfer — Training Todo Checklist")
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

    # Title
    story.append(Paragraph("NILUT Model Training — Todo Checklist", s["DocTitle"]))
    story.append(Paragraph("Step-by-step guide for retraining with new reference &amp; content images", s["DocSubtitle"]))
    story.append(divider())

    # ── SECTION 1: Current State ──
    story.append(Paragraph("1. Current State (Before Changes)", s["SectionHead"]))

    story.append(Paragraph("Current Directory Structure:", s["SubHead"]))
    story.append(Paragraph(
        "test_images/training_data/<br/>"
        "&#9500;&#9472;&#9472; neon/input/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 36 content images<br/>"
        "&#9500;&#9472;&#9472; portrait/input/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 10 content images<br/>"
        "&#9492;&#9472;&#9472; reference/<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; natural_clean_film/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; ~11 reference images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; moody_cinematic/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; ~12 reference images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; bright_airy_cream_whites/ &nbsp;&#8594; ~11 reference images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; high_contrast_street_editorial/ &#8594; ~12 reference images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9492;&#9472;&#9472; warm_vintage_kodachrome/ &#8594; ~11 reference images",
        s["CodeBlk"]
    ))

    story.append(make_table(
        ["Item", "Current Count"],
        [
            ["Content images (neon + portrait)", "46"],
            ["Reference images (5 styles)", "~57"],
            ["Per-style models", "5 (.pt files)"],
            ["Universal model", "1 (universal.pt)"],
            ["train.py ref image limit", "First 6 per style ([:6])"],
        ],
        col_widths=[280, 210],
    ))

    # ── SECTION 2: Todo Steps ──
    story.append(PageBreak())
    story.append(Paragraph("2. Todo Checklist", s["SectionHead"]))

    # Step 1
    story.append(Paragraph("<font color='#0d47a1'><b>Step 1:</b></font> Add New Reference Images (40 per style)", s["SubHead"]))
    story.append(Paragraph("Add 40 new reference images to each existing style folder:", s["Body"]))
    story.append(make_table(
        ["Folder", "Existing", "Add", "Total (Train)", "Keep Aside (Test)"],
        [
            ["natural_clean_film/", "~11", "+40", "~41", "10"],
            ["moody_cinematic/", "~12", "+40", "~42", "10"],
            ["bright_airy_cream_whites/", "~11", "+40", "~41", "10"],
            ["high_contrast_street_editorial/", "~12", "+40", "~42", "10"],
            ["warm_vintage_kodachrome/", "~11", "+40", "~41", "10"],
        ],
        col_widths=[160, 60, 50, 80, 100],
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        '<b>Total:</b> ~200 new reference images for training + 50 for testing = 250 new images',
        s["Highlight"]
    ))

    # Step 2
    story.append(Paragraph("<font color='#0d47a1'><b>Step 2:</b></font> Create Test/Validation Folder", s["SubHead"]))
    story.append(Paragraph("Create a separate folder for test images (10 per style). These should NOT be used in training.", s["Body"]))
    story.append(Paragraph(
        "test_images/validation/<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; natural_clean_film/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 10 test images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; moody_cinematic/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 10 test images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; bright_airy_cream_whites/ &nbsp;&#8594; 10 test images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; high_contrast_street_editorial/ &#8594; 10 test images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9492;&#9472;&#9472; warm_vintage_kodachrome/ &#8594; 10 test images",
        s["CodeBlk"]
    ))
    story.append(Paragraph(
        'Keep test images separate — never mix with training data. Use these to verify the model learned the style, not just memorized images.',
        s["Warning"]
    ))

    # Step 3
    story.append(Paragraph("<font color='#0d47a1'><b>Step 3:</b></font> Add Landscape Content Images", s["SubHead"]))
    story.append(Paragraph("Create a new content image directory with 20 landscape images:", s["Body"]))
    story.append(Paragraph(
        "test_images/training_data/<br/>"
        "&#9500;&#9472;&#9472; neon/input/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 36 images (existing)<br/>"
        "&#9500;&#9472;&#9472; portrait/input/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 10 images (existing)<br/>"
        "&#9500;&#9472;&#9472; <b>landscape/input/</b> &nbsp;&nbsp;&nbsp;&#8594; <b>20 images (NEW)</b><br/>"
        "&#9492;&#9472;&#9472; reference/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; style folders",
        s["CodeBlk"]
    ))
    story.append(Paragraph(
        'New total content images: 36 (neon) + 10 (portrait) + 20 (landscape) = <b>66 content images</b>',
        s["Highlight"]
    ))

    # Step 4
    story.append(PageBreak())
    story.append(Paragraph("<font color='#0d47a1'><b>Step 4:</b></font> Update train.py — Reference Image Limit", s["SectionHead"]))
    story.append(Paragraph("Change the reference image limit from 6 to 50.", s["Body"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Current code (Line 63):", s["SubHead"]))
    story.append(Paragraph(
        'ref_images = sorted(glob.glob(f"{style_path}/*.*"))<b>[:6]</b>',
        s["CodeBlk"]
    ))

    story.append(Paragraph("Change to:", s["SubHead"]))
    story.append(Paragraph(
        'ref_images = sorted(glob.glob(f"{style_path}/*.*"))<b>[:50]</b>',
        s["CodeBlk"]
    ))

    story.append(Paragraph(
        'Without this change, the training will only use the first 6 reference images per style and ignore all the new ones you added!',
        s["Warning"]
    ))

    # Step 5
    story.append(Paragraph("<font color='#0d47a1'><b>Step 5:</b></font> Update train.py — Add Landscape Content Images", s["SectionHead"]))
    story.append(Paragraph("Update Line 46 to include the new landscape directory.", s["Body"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Current code (Line 46):", s["SubHead"]))
    story.append(Paragraph(
        'content = [cv2.imread(f) for f in<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;glob.glob(f"{BASE}/neon/input/*.*") +<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;glob.glob(f"{BASE}/portrait/input/*.*")]',
        s["CodeBlk"]
    ))

    story.append(Paragraph("Change to:", s["SubHead"]))
    story.append(Paragraph(
        'content = [cv2.imread(f) for f in<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;glob.glob(f"{BASE}/neon/input/*.*") +<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;glob.glob(f"{BASE}/portrait/input/*.*") +<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;<b>glob.glob(f"{BASE}/landscape/input/*.*")</b>]',
        s["CodeBlk"]
    ))

    # Step 6
    story.append(Paragraph("<font color='#0d47a1'><b>Step 6:</b></font> Run Training", s["SectionHead"]))
    story.append(Paragraph(
        'cd /home/shreyank06/Desktop/projects/upwork_projects/style_transfer<br/>'
        'python train.py',
        s["CodeBlk"]
    ))
    story.append(b("Choose <b>Option 1</b> to train all 5 per-style models + universal model", s))
    story.append(b("Choose <b>Option 2</b> to train only the universal model (faster)", s))
    story.append(Spacer(1, 4))

    story.append(Paragraph("What happens during training:", s["SubHead"]))
    story.append(make_table(
        ["Action", "Details"],
        [
            ["1. Load content images", "66 images (neon + portrait + landscape)"],
            ["2. Per-style training", "For each of 5 styles: stack up to 50 ref images, train MLP, save .pt"],
            ["3. Backup old universal", "Moves existing universal.pt to models/nilut/universal/{timestamp}/"],
            ["4. Train universal model", "Uses all references from all styles combined"],
            ["5. Save new universal", "Saves to models/nilut/latest/universal.pt"],
            ["6. Update meta.json", "Records timestamp, sample count, epochs"],
        ],
        col_widths=[150, 340],
    ))

    # Step 7
    story.append(PageBreak())
    story.append(Paragraph("<font color='#0d47a1'><b>Step 7:</b></font> Verify &amp; Test", s["SectionHead"]))
    story.append(b("Check that new .pt model files are generated in <b>models/nilut/</b>", s))
    story.append(b("Check that old universal model is backed up in <b>models/nilut/universal/{timestamp}/</b>", s))
    story.append(b("Test with validation images (from <b>test_images/validation/</b>)", s))
    story.append(b("Run the web app and test style transfer on new images", s))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        'uvicorn app:app --host 0.0.0.0 --port 8000',
        s["CodeBlk"]
    ))

    # ── SECTION 3: Model Storage Summary ──
    story.append(Paragraph("3. Model Storage Summary", s["SectionHead"]))

    story.append(make_table(
        ["Model", "Location", "Backup?"],
        [
            ["natural_clean_film.pt", "models/nilut/", "NO — overwritten directly"],
            ["moody_cinematic.pt", "models/nilut/", "NO — overwritten directly"],
            ["bright_airy_cream_whites.pt", "models/nilut/", "NO — overwritten directly"],
            ["high_contrast_street_editorial.pt", "models/nilut/", "NO — overwritten directly"],
            ["warm_vintage_kodachrome.pt", "models/nilut/", "NO — overwritten directly"],
            ["universal.pt (new)", "models/nilut/latest/", "YES — old moved to timestamped folder"],
            ["universal.pt (old backup)", "models/nilut/universal/{timestamp}/", "Timestamped by file modified date"],
        ],
        col_widths=[170, 170, 150],
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        'Per-style models get overwritten with no backup. If you want to keep old per-style models, manually copy them before retraining.',
        s["Warning"]
    ))

    # ── SECTION 4: Final Directory Structure ──
    story.append(Paragraph("4. Final Directory Structure (After All Changes)", s["SectionHead"]))

    story.append(Paragraph(
        "test_images/<br/>"
        "&#9500;&#9472;&#9472; training_data/<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; neon/input/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 36 content images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; portrait/input/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 10 content images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; <b>landscape/input/</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; <b>20 content images (NEW)</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9492;&#9472;&#9472; reference/<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; natural_clean_film/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; ~51 ref images (11 + 40)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; moody_cinematic/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; ~52 ref images (12 + 40)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; bright_airy_cream_whites/ &#8594; ~51 ref images (11 + 40)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; high_contrast_street_editorial/ &#8594; ~52 ref images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9492;&#9472;&#9472; warm_vintage_kodachrome/ &#8594; ~51 ref images (11 + 40)<br/>"
        "&#9492;&#9472;&#9472; <b>validation/</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; <b>(NEW)</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; natural_clean_film/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 10 test images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; moody_cinematic/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8594; 10 test images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; bright_airy_cream_whites/ &#8594; 10 test images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&#9472; high_contrast_street_editorial/ &#8594; 10 test images<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9492;&#9472;&#9472; warm_vintage_kodachrome/ &#8594; 10 test images",
        s["CodeBlk"]
    ))

    # ── SECTION 5: Summary Table ──
    story.append(Paragraph("5. Before vs After Summary", s["SectionHead"]))

    story.append(make_table(
        ["", "Before", "After"],
        [
            ["Content images", "46 (neon + portrait)", "66 (+ 20 landscape)"],
            ["Reference images (training)", "~57 (5 styles)", "~257 (+ 200 new)"],
            ["Reference images (testing)", "0", "50 (10 per style)"],
            ["train.py ref limit", "[:6]", "[:50]"],
            ["train.py content dirs", "neon + portrait", "neon + portrait + landscape"],
            ["Per-style models", "5", "5 (retrained)"],
            ["Universal model", "1", "1 (retrained, old backed up)"],
        ],
        col_widths=[160, 165, 165],
    ))

    # Bottom line
    story.append(Spacer(1, 10))
    story.append(divider())
    story.append(Paragraph(
        '<b>Quick checklist:</b> (1) Add 40 ref images per style (2) Create validation folder with 10 per style '
        '(3) Create landscape/input/ with 20 images (4) Change [:6] to [:50] in train.py line 63 '
        '(5) Add landscape glob to train.py line 46 (6) Run python train.py (7) Test results',
        s["Highlight"]
    ))

    # ── SECTION 6: How Training Actually Works ──
    story.append(PageBreak())
    story.append(Paragraph("6. How Training Actually Works (Explained)", s["SectionHead"]))

    story.append(Paragraph(
        "The content/target images are <b>not just loaded passively</b>. They are actively used in training. "
        "Here's what actually happens:",
        s["Body"]
    ))
    story.append(Spacer(1, 4))

    story.append(make_table(
        ["Step", "What Happens"],
        [
            ["1. Load content images", "Load all 66 content images (neon + portrait + landscape)"],
            ["2. Sample content colors", "Extract 50,000 random colors (A,B channels) from each content image"],
            ["3. Sample reference colors", "Extract 50,000 random colors (A,B channels) from reference images"],
            ["4. Pair colors", "Use histogram matching to pair content colors with reference colors (red with red, blue with blue)"],
            ["5. Train MLP", "Train the neural network on these pairs: content color &#8594; reference color"],
        ],
        col_widths=[140, 350],
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Role of Each Image Type:", s["SubHead"]))
    story.append(make_table(
        ["Image Type", "Role in Training"],
        [
            ["Content images (66)", "Provide the INPUT side — colors the model learns FROM"],
            ["Reference images (~257)", "Provide the OUTPUT side — colors the model learns TO"],
        ],
        col_widths=[160, 330],
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("What the model learns:", s["SubHead"]))
    story.append(Paragraph(
        '"When I see <b>this color from a content image</b>, convert it to <b>that color from the reference image</b>"',
        s["Highlight"]
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        '<b>Without content images</b>, the model would have nothing to learn <b>from</b>.<br/>'
        '<b>Without reference images</b>, the model would have nothing to learn <b>to</b>.',
        s["Warning"]
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Visual Flow:", s["SubHead"]))
    story.append(Paragraph(
        "Content Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Reference Image<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9474; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9474;<br/>"
        "Sample 50K colors (A,B) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sample 50K colors (A,B)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9474; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9474;<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#9492;&#9472;&#9472;&#9472;&#9472; Histogram Match &#9472;&#9472;&#9472;&#9472;&#9488;<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9474;<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Color Pairs (input &#8594; output)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9474;<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train MLP (500 epochs)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9474;<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Saved Model (.pt file)",
        s["CodeBlk"]
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
