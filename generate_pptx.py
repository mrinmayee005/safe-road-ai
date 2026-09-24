"""
Safe Road AI - Comprehensive PowerPoint Generator
Builds a professional 16:9 widescreen PPT presentation with:
- Exact PCCOE template branding (Dark navy banner, clean cards)
- Embedded images, charts, and diagrams
- Professional tables with high contrast
- Specific models (MobileNetV3-Small, Random Forest, FIFO filter)
- Both Phase 1 (Completed) & Phase 2 (Roadmap)
"""

import os
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Initialize Presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette
NAVY_HEADER = RGBColor(11, 35, 92)     # #0B235C
NAVY_DARK = RGBColor(15, 23, 42)        # #0F172A
SLATE_CARD = RGBColor(241, 245, 249)    # #F1F5F9
BORDER_COLOR = RGBColor(203, 213, 225)  # #CBD5E1
CYAN_ACCENT = RGBColor(2, 132, 199)     # #0284C7
AMBER_ACCENT = RGBColor(217, 119, 6)    # #D97706
GREEN_ACCENT = RGBColor(5, 150, 105)    # #059669
WHITE = RGBColor(255, 255, 255)
TEXT_DARK = RGBColor(30, 41, 59)        # #1E293B
TEXT_MUTED = RGBColor(100, 116, 139)    # #64748B

blank_layout = prs.slide_layouts[6]

ASSETS_DIR = Path("assets/slides")
RESULTS_DIR = Path("results")

def add_header(slide, title_text, category="Review 1 Presentation"):
    # Header Banner
    header_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05))
    header_box.fill.solid()
    header_box.fill.fore_color.rgb = NAVY_HEADER
    header_box.line.color.rgb = NAVY_HEADER
    
    tf = header_box.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    tf.margin_left = Inches(0.8)
    tf.margin_right = Inches(0.8)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = "Calibri"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    
    # Bottom Footer
    footer_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.15), Inches(13.333), Inches(0.35))
    footer_box.fill.solid()
    footer_box.fill.fore_color.rgb = RGBColor(241, 245, 249)
    footer_box.line.color.rgb = RGBColor(226, 232, 240)
    
    ftf = footer_box.text_frame
    ftf.vertical_anchor = MSO_ANCHOR.MIDDLE
    ftf.margin_left = Inches(0.8)
    ftf.margin_right = Inches(0.8)
    fp = ftf.paragraphs[0]
    fp.text = "Safe Road AI: Deep Learning & Telematics for Accident Detection  |  PCCOE Pune (2026-27)"
    fp.font.name = "Calibri"
    fp.font.size = Pt(10.5)
    fp.font.color.rgb = TEXT_MUTED
    fp.alignment = PP_ALIGN.LEFT

def add_card(slide, left, top, width, height, bg_color=SLATE_CARD, border_color=BORDER_COLOR):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    return shape

# =========================================================================
# SLIDE 1: Title Slide (Matches PCCOE Official Layout)
# =========================================================================
s1 = prs.slides.add_slide(blank_layout)

# Outer Border
border = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(0.4), Inches(12.533), Inches(6.7))
border.fill.background()
border.line.color.rgb = CYAN_ACCENT
border.line.width = Pt(3.0)

# Content Box
tbox = s1.shapes.add_textbox(Inches(1.0), Inches(0.7), Inches(11.333), Inches(5.2))
tf = tbox.text_frame
tf.word_wrap = True

p1 = tf.paragraphs[0]
p1.text = "Review 1 presentation on"
p1.font.name = "Georgia"
p1.font.size = Pt(18)
p1.font.italic = True
p1.font.color.rgb = TEXT_MUTED
p1.alignment = PP_ALIGN.CENTER

p2 = tf.add_paragraph()
p2.text = "“ Safe Road AI: Deep Learning & Multi-Sensor Telematics for Real-Time Four-Wheeler Accident Detection ”"
p2.font.name = "Calibri"
p2.font.size = Pt(25)
p2.font.bold = True
p2.font.color.rgb = NAVY_HEADER
p2.alignment = PP_ALIGN.CENTER
p2.space_before = Pt(8)
p2.space_after = Pt(12)

p3 = tf.add_paragraph()
p3.text = "Submitted in partial fulfillment for the award of degree of\nBachelor of Technology in Computer Engineering"
p3.font.name = "Calibri"
p3.font.size = Pt(14)
p3.font.color.rgb = TEXT_DARK
p3.alignment = PP_ALIGN.CENTER

p4 = tf.add_paragraph()
p4.text = "\nBy Project Team:"
p4.font.name = "Calibri"
p4.font.size = Pt(13)
p4.font.bold = True
p4.font.color.rgb = CYAN_ACCENT
p4.alignment = PP_ALIGN.CENTER

p5 = tf.add_paragraph()
p5.text = "Mrinmayee Kulkarni (124B2B025)   |   Soniya Lakade (123B1B177)   |   Sarthak Bagul (124B2B021)"
p5.font.name = "Calibri"
p5.font.size = Pt(14)
p5.font.bold = True
p5.font.color.rgb = TEXT_DARK
p5.alignment = PP_ALIGN.CENTER

p6 = tf.add_paragraph()
p6.text = "\nUnder the supervision of:\nProf. Madhuri Suryavanshi"
p6.font.name = "Calibri"
p6.font.size = Pt(15)
p6.font.bold = True
p6.font.color.rgb = NAVY_HEADER
p6.alignment = PP_ALIGN.CENTER

p7 = tf.add_paragraph()
p7.text = "Assistant Professor, Department of Computer Engineering"
p7.font.name = "Calibri"
p7.font.size = Pt(12)
p7.font.color.rgb = TEXT_MUTED
p7.alignment = PP_ALIGN.CENTER

p8 = tf.add_paragraph()
p8.text = "Pimpri Chinchwad College of Engineering (PCCOE), Pune  |  A. Y. 2026-27"
p8.font.name = "Calibri"
p8.font.size = Pt(13)
p8.font.bold = True
p8.font.color.rgb = TEXT_DARK
p8.alignment = PP_ALIGN.CENTER
p8.space_before = Pt(8)

# PCCOE Logo
logo_path = ASSETS_DIR / "page_1_img_1.jpeg"
if logo_path.exists():
    s1.shapes.add_picture(str(logo_path), Inches(6.1), Inches(5.8), width=Inches(1.1))

# =========================================================================
# SLIDE 2: Review 1 Outline
# =========================================================================
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "Review 1: Presentation Outline")

outline_items = [
    ("1", "Introduction & Motivation"),
    ("2", "Problem Statement"),
    ("3", "Research Objectives"),
    ("4", "Scope of Research Work"),
    ("5", "Key Features & Ecosystem"),
    ("6", "Literature Review (Document Count)"),
    ("7", "Literature Review (Comparative Table)"),
    ("8", "Research Gaps Identified"),
    ("9", "Proposed Approach"),
    ("10", "Research Methodology"),
    ("11", "Architecture Diagram for Proposed Work"),
    ("12", "Dataset Creation & Validation (3 Datasets)"),
    ("13", "Mathematical Model (Formulations 1 to 5)"),
    ("14", "Experimental Results & Benchmarks"),
    ("15", "Conclusion & References")
]

col1_items = outline_items[:8]
col2_items = outline_items[8:]

for i, (num, text) in enumerate(col1_items):
    top = Inches(1.3 + i * 0.68)
    card = add_card(s2, Inches(1.0), top, Inches(5.3), Inches(0.58), bg_color=WHITE)
    tf = card.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = f"{num}.   {text}"
    p.font.name = "Calibri"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = NAVY_HEADER

for i, (num, text) in enumerate(col2_items):
    top = Inches(1.3 + i * 0.68)
    card = add_card(s2, Inches(7.0), top, Inches(5.3), Inches(0.58), bg_color=WHITE)
    tf = card.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = f"{num}.   {text}"
    p.font.name = "Calibri"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = NAVY_HEADER

# =========================================================================
# SLIDE 3: Introduction & Motivation
# =========================================================================
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "Introduction & Motivation")

# Left Column Card (Stats)
card_left = add_card(s3, Inches(0.8), Inches(1.35), Inches(5.6), Inches(5.4), bg_color=WHITE)
tf_l = card_left.text_frame
tf_l.word_wrap = True
tf_l.margin_left = Inches(0.3)
tf_l.margin_top = Inches(0.3)

p = tf_l.paragraphs[0]
p.text = "The Global & Indian Road Safety Crisis"
p.font.name = "Calibri"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = NAVY_HEADER

bullets_l = [
    "• 1.35 Million road fatalities globally each year (WHO Global Status Report).",
    "• India ranks #1 in road deaths: ~1.68 Lakh fatalities annually across 4.6 lakh recorded crashes (MoRTH 2023–24).",
    "• In over 45% of highway collisions, occupants suffer severe concussion or loss of consciousness, rendering them unable to dial 108/112.",
    "• Notification Delay on Indian rural/highway corridors averages 22 to 35 minutes due to lack of bystander reporting.",
    "• Economic Burden: Road accidents consume ~3.14% of India's annual GDP in healthcare costs and productive loss."
]
for b in bullets_l:
    bp = tf_l.add_paragraph()
    bp.text = b
    bp.font.name = "Calibri"
    bp.font.size = Pt(13)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(8)

# Right Column Card (Golden Hour & Opportunity)
card_right = add_card(s3, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.4), bg_color=WHITE)
tf_r = card_right.text_frame
tf_r.word_wrap = True
tf_r.margin_left = Inches(0.3)
tf_r.margin_top = Inches(0.3)

p = tf_r.paragraphs[0]
p.text = "The 'Golden Hour' & Zero-Cost Solution"
p.font.name = "Calibri"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = AMBER_ACCENT

bullets_r = [
    "• The Golden Hour Principle: Emergency trauma survival increases by >60% if medical care reaches the victim within the first 60 minutes.",
    "• The Commercial Inequality Gap: Systems like Tesla eCall and BMW ConnectedDrive require expensive factory sensors ($₹5,000–₹12,000). Over 92% of everyday cars lack automated crash SOS.",
    "• Ubiquitous Smartphone Hardware: Over 1.5 billion smartphones feature HD cameras, 6-axis MEMS IMUs (accelerometers + gyroscopes), and multi-GNSS GPS.",
    "• Zero-Hardware-Cost Innovation: Safe Road AI converts any driver's existing smartphone into an automated accident detection and 108 emergency dispatch terminal with zero extra hardware!"
]
for b in bullets_r:
    bp = tf_r.add_paragraph()
    bp.text = b
    bp.font.name = "Calibri"
    bp.font.size = Pt(13)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(8)

# =========================================================================
# SLIDE 4: Problem Statement
# =========================================================================
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "Problem Statement")

# Formal Statement Box
stmt_box = add_card(s4, Inches(0.8), Inches(1.3), Inches(11.7), Inches(1.3), bg_color=WHITE, border_color=CYAN_ACCENT)
tf_s = stmt_box.text_frame
tf_s.word_wrap = True
tf_s.margin_left = Inches(0.3)
tf_s.margin_top = Inches(0.15)
p = tf_s.paragraphs[0]
p.text = "Core Problem Statement:"
p.font.name = "Calibri"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = CYAN_ACCENT

p2 = tf_s.add_paragraph()
p2.text = "To design, develop, and evaluate an automated, real-time, four-wheeler accident detection and emergency dispatch pipeline using purely an ordinary smartphone mounted on the windshield, achieving near-zero false alarms under aggressive braking, potholes, speed bumps, and night-time glare without requiring external OBD-II hardware."
p2.font.name = "Calibri"
p2.font.size = Pt(13)
p2.font.color.rgb = NAVY_HEADER
p2.space_before = Pt(3)

# 2 Column Cards: Why Single Modality Fails
c1 = add_card(s4, Inches(0.8), Inches(2.8), Inches(5.7), Inches(4.0), bg_color=WHITE)
tf1 = c1.text_frame
tf1.word_wrap = True
tf1.margin_left = Inches(0.25)
tf1.margin_top = Inches(0.2)
p = tf1.paragraphs[0]
p.text = "❌ Why Video-Only (Cameras) Fail:"
p.font.name = "Calibri"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = RGBColor(220, 38, 38)

b_c1 = [
    "• Blinding Environmental Factors: High-beam headlight glare, night darkness, rain streaks, and windshield wipers distort visual road semantics.",
    "• The 'Non-Ego Collision' Catastrophe: When two vehicles crash ahead in an adjacent lane, a camera-only system detects visual collision and panics (71.4% False Alarm Rate in CCD real dashcam benchmark), even though the host vehicle is 100% unharmed!",
    "• Heavy Computation: Heavy deep CNNs (ResNet-18) cause mobile battery drain and thermal throttling."
]
for b in b_c1:
    bp = tf1.add_paragraph()
    bp.text = b
    bp.font.name = "Calibri"
    bp.font.size = Pt(12)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(6)

c2 = add_card(s4, Inches(6.8), Inches(2.8), Inches(5.7), Inches(4.0), bg_color=WHITE)
tf2 = c2.text_frame
tf2.word_wrap = True
tf2.margin_left = Inches(0.25)
tf2.margin_top = Inches(0.2)
p = tf2.paragraphs[0]
p.text = "❌ Why Sensor-Only (Accelerometers) Fail:"
p.font.name = "Calibri"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = RGBColor(220, 38, 38)

b_c2 = [
    "• Indian Road Surface Anomalies: Sharp potholes, steep speed breakers, and expansion joints generate vertical shock spikes (>15 m/s²) that mimic real crashes.",
    "• Non-Crash Physical Events: Emergency braking at signals, sudden phone drops, or door slams produce false alarm rates exceeding 40%.",
    "• The Safe Road AI Solution: Mathematical Late Decision Fusion (P_final = α·Pv + (1-α)·Ps) combined with a FIFO Temporal Confirmation Filter (W=3, K=2) neutralizing both failure modes!"
]
for b in b_c2:
    bp = tf2.add_paragraph()
    bp.text = b
    bp.font.name = "Calibri"
    bp.font.size = Pt(12)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(6)

# =========================================================================
# SLIDE 5: Research Objectives (Phase 1 & Phase 2)
# =========================================================================
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "Research Objectives (Phase 1 & Phase 2)")

c_obj1 = add_card(s5, Inches(0.8), Inches(1.3), Inches(5.7), Inches(5.5), bg_color=WHITE)
tf_o1 = c_obj1.text_frame
tf_o1.word_wrap = True
tf_o1.margin_left = Inches(0.3)
tf_o1.margin_top = Inches(0.25)

p = tf_o1.paragraphs[0]
p.text = "Phase 1: Research, Models & Benchmarks (Completed)"
p.font.name = "Calibri"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = GREEN_ACCENT

b_o1 = [
    "1. Dual-Modality Temporal Synchronizer: Ingest and align 5 FPS dashcam video with 50 Hz 6-axis IMU time-series.",
    "2. Lightweight Deep CNN Development: Fine-tune Google MobileNetV3-Small (depthwise separable convolutions + SE attention) benchmarked against ResNet-18 for visual crash risk (Pv).",
    "3. Orientation-Invariant Kinematic Feature Pipeline: Formulate magnitude A=√(ax²+ay²+az²), G=√(ωx²+ωy²+ωz²), and Jerk dA/dt, training Random Forest (100 estimators) for sensor risk (Ps).",
    "4. Weighted Probability Fusion & Temporal Gating: Implement P_final = 0.55·Pv + 0.45·Ps with a FIFO confirmation gate (W=3, K=2) to suppress pothole false alarms.",
    "5. Cross-Dataset Empirical Validation: Rigorously benchmark across Synthetic (280 clips), CCD Real Dashcam (75,000 frames), and Real Telematics (120 trips)."
]
for b in b_o1:
    bp = tf_o1.add_paragraph()
    bp.text = b
    bp.font.name = "Calibri"
    bp.font.size = Pt(11.5)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(6)

c_obj2 = add_card(s5, Inches(6.8), Inches(1.3), Inches(5.7), Inches(5.5), bg_color=WHITE)
tf_o2 = c_obj2.text_frame
tf_o2.word_wrap = True
tf_o2.margin_left = Inches(0.3)
tf_o2.margin_top = Inches(0.25)

p = tf_o2.paragraphs[0]
p.text = "Phase 2: Mobile Edge & Emergency Ecosystem (In Progress)"
p.font.name = "Calibri"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = CYAN_ACCENT

b_o2 = [
    "1. Mobile Edge Quantization: Export PyTorch MobileNetV3 and Scikit-Learn Random Forest to ONNX Runtime and TFLite INT8 for zero-GPU on-device phone execution.",
    "2. Autonomous Emergency Dispatch: Integrate device GNSS and Twilio SMS/GSM API to auto-transmit crash severity and live location to 108 Emergency Ambulance triage.",
    "3. Driver Safety & Drowsiness Ecosystem: Implement Eye Aspect Ratio (EAR) driver monitoring via phone front camera and real-time harsh driving score (0–100).",
    "4. Black-Spot Proximity Warning: Real-time geofenced audio alerts when vehicle approaches high-accident highway black spots.",
    "5. Multi-Vehicle Dynamics: Implement Kalman-filtered lean angle compensation for two-wheeler motorcycle deployment."
]
for b in b_o2:
    bp = tf_o2.add_paragraph()
    bp.text = b
    bp.font.name = "Calibri"
    bp.font.size = Pt(11.5)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(6)

# =========================================================================
# SLIDE 6: Scope of Research Work
# =========================================================================
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "Scope of Research Work")

# Left Column: Detailed Scope
sc_card = add_card(s6, Inches(0.8), Inches(1.35), Inches(6.2), Inches(5.4), bg_color=WHITE)
tf_sc = sc_card.text_frame
tf_sc.word_wrap = True
tf_sc.margin_left = Inches(0.3)
tf_sc.margin_top = Inches(0.25)

p = tf_sc.paragraphs[0]
p.text = "System Scope & Operational Boundaries"
p.font.name = "Calibri"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = NAVY_HEADER

sc_bullets = [
    "• Vehicular Target: Four-wheelers (Hatchbacks, Sedans, SUVs, Light Trucks) mounted in a standard windshield suction cradle.",
    "• Collision Types Covered: Frontal collisions, rear-end impacts, and lateral T-bone collisions.",
    "• Environmental Envelope: Validated across daylight, night darkness, rainy weather, snowy conditions, and headlight glare.",
    "• Hardware Envelope: Mid-tier mobile processors (ARM Cortex-A55/A78, 4GB RAM) without requiring specialized GPU accelerators.",
    "• Offline On-Device Execution: Core crash inference runs 100% on-device (zero cloud latency); cellular connectivity is used solely for emergency SMS/GPS dispatch.",
    "• Energy Budget: Optimized at 5 FPS to consume <12% battery per hour of continuous navigation."
]
for b in sc_bullets:
    bp = tf_sc.add_paragraph()
    bp.text = b
    bp.font.name = "Calibri"
    bp.font.size = Pt(12)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(7)

# Right: Diagram from PDF (page 5 image)
p5_img = ASSETS_DIR / "page_5_img_1.png"
if p5_img.exists():
    s6.shapes.add_picture(str(p5_img), Inches(7.3), Inches(1.4), width=Inches(5.2))

# =========================================================================
# SLIDE 7: Key Features: Safe Road AI Ecosystem
# =========================================================================
s7 = prs.slides.add_slide(blank_layout)
add_header(s7, "Key Features: Safe Road AI Ecosystem")

features = [
    ("1. AI-Based Safety Core", [
        "• MobileNetV3-Small visual collision probability (Pv).",
        "• Random Forest 50 Hz motion shock probability (Ps).",
        "• Weighted probability fusion (P_final = 0.55Pv + 0.45Ps).",
        "• FIFO persistence filter (W=3, K=2) discarding potholes."
    ], CYAN_ACCENT),
    ("2. Smart Road Monitoring", [
        "• Live forward camera scene ingestion at 5 FPS.",
        "• Real-time HUD overlay showing live risk timeline.",
        "• Pre-crash hazard anticipation before physical impact.",
        "• Automatic high-risk event video recording clip capture."
    ], GREEN_ACCENT),
    ("3. Automated Emergency SOS", [
        "• Automated extraction of GPS coordinates (Lat/Lon).",
        "• 15-second driver emergency countdown cancellation timer.",
        "• Automated SMS dispatch with Google Maps link to 108.",
        "• Zero cloud delay: instant local trigger upon confirmation."
    ], RGBColor(220, 38, 38)),
    ("4. Driver Telematics & Scoring", [
        "• Continuous trip safety score out of 100.",
        "• Harsh braking and rapid acceleration detection.",
        "• Excessive rotational speed detection (G > 0.8 rad/s).",
        "• High-risk highway black-spot audio warnings."
    ], AMBER_ACCENT)
]

for idx, (title, items, color) in enumerate(features):
    col = idx % 2
    row = idx // 2
    left = Inches(0.8 + col * 5.9)
    top = Inches(1.35 + row * 2.8)
    card = add_card(s7, left, top, Inches(5.6), Inches(2.6), bg_color=WHITE)
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.18)
    
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Calibri"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = color
    
    for it in items:
        bp = tf.add_paragraph()
        bp.text = it
        bp.font.name = "Calibri"
        bp.font.size = Pt(11.5)
        bp.font.color.rgb = TEXT_DARK
        bp.space_before = Pt(3)

# =========================================================================
# SLIDE 8: Literature Review: Document Count Analysis
# =========================================================================
s8 = prs.slides.add_slide(blank_layout)
add_header(s8, "Literature Review: Document Count Analysis")

# Table 1: Year-Wise Publication Analysis
rows = 7
cols = 5
table_shape = s8.shapes.add_table(rows, cols, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.2))
tbl = table_shape.table

# Set Column Widths
tbl.columns[0].width = Inches(1.8)
tbl.columns[1].width = Inches(1.8)
tbl.columns[2].width = Inches(2.2)
tbl.columns[3].width = Inches(1.5)
tbl.columns[4].width = Inches(4.4)

headers = ["Publication Year", "IEEE Conferences", "Other Reputed Journals", "Total Papers", "Primary Research Trend / Focus"]
for j, h in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY_HEADER
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = "Calibri"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER if j < 4 else PP_ALIGN.LEFT

table_data = [
    ["2025", "1", "1", "2", "Edge AI, Vision Transformers, On-Device Neural Networks"],
    ["2024", "1", "5", "6", "Smartphone Edge Telematics, Accident Severity ML Models"],
    ["2023", "6", "4", "10", "Multimodal Sensor Fusion, Spatio-Temporal 3D-CNNs, IoT"],
    ["2022", "4", "2", "6", "Surveillance CCTV Video Encoding, OBD-II CAN-bus Analysis"],
    ["2020", "1", "0", "1", "Foundational MobileNet Architectures & Deep Learning Baselines"],
    ["TOTAL", "13", "12", "25", "Comprehensive Multimodal Evolution (2020 to 2025)"]
]

for i, row_data in enumerate(table_data):
    for j, val in enumerate(row_data):
        cell = tbl.cell(i+1, j)
        cell.fill.solid()
        if i == len(table_data) - 1:
            cell.fill.fore_color.rgb = RGBColor(226, 232, 240)
        else:
            cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(248, 250, 252)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = "Calibri"
        p.font.size = Pt(12)
        p.font.bold = (i == len(table_data) - 1 or j == 0)
        p.font.color.rgb = NAVY_HEADER if i == len(table_data) - 1 else TEXT_DARK
        p.alignment = PP_ALIGN.CENTER if j < 4 else PP_ALIGN.LEFT

# Summary Card Below Table
sum_card = add_card(s8, Inches(0.8), Inches(4.85), Inches(11.7), Inches(2.0), bg_color=WHITE)
tf_sm = sum_card.text_frame
tf_sm.word_wrap = True
tf_sm.margin_left = Inches(0.3)
tf_sm.margin_top = Inches(0.18)

p = tf_sm.paragraphs[0]
p.text = "Key Literature Survey Insights:"
p.font.name = "Calibri"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = CYAN_ACCENT

ins = [
    "• Paradigm Shift (2020 vs 2024): Research transitioned decisively from stationary roadside CCTV cameras (2020–2022) to vehicle-mounted in-cabin mobile edge telematics (2023–2025).",
    "• Single vs Multimodal: Over 68% of surveyed literature prior to 2023 relied on single modalities, universally suffering from environmental noise or extreme false alarms.",
    "• Real-Time Gap: Advanced 3D-CNN models in literature (e.g., 2023) achieve high accuracy but fail deployment tests due to >1.5-second latency on standard mobile processors."
]
for it in ins:
    bp = tf_sm.add_paragraph()
    bp.text = it
    bp.font.name = "Calibri"
    bp.font.size = Pt(12)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(4)

# =========================================================================
# SLIDE 9: Literature Review: Comparative Paper Analysis (Part 1)
# =========================================================================
s9 = prs.slides.add_slide(blank_layout)
add_header(s9, "Literature Review: Comparative Analysis Table (Part 1)")

rows = 4
cols = 6
t_shape9 = s9.shapes.add_table(rows, cols, Inches(0.6), Inches(1.35), Inches(12.133), Inches(5.4))
tbl9 = t_shape9.table

col_w9 = [Inches(1.8), Inches(1.8), Inches(1.6), Inches(1.6), Inches(2.6), Inches(2.7)]
for j, w in enumerate(col_w9):
    tbl9.columns[j].width = w

headers9 = ["Paper Title & Citation", "Objective", "Implementation Method", "Performance Metrics", "Critical Technical Limitations", "Safe Road AI Solution"]
for j, h in enumerate(headers9):
    cell = tbl9.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY_HEADER
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

data9 = [
    ["Bao et al. [2]\n(ACM MM 2020)", "Accident anticipation in dashcam video", "Spatio-Temporal Relational CNN (Video Only)", "Accuracy: 85.6%\nLatency: 500 ms", "High False Alarm Rate on non-ego crashes; blinded by night headlight glare.", "We cross-verify video with 50 Hz phone IMU; non-ego crashes have zero IMU shock and are discarded!"],
    ["Aloul et al. [3]\n(IEEE Trans. ITS 2018)", "Smartphone-based crash detection", "Heuristic G-force threshold trigger (IMU Only)", "Accuracy: 88.0%\nResponse: 50 ms", "Extreme False Alarms (>40%) on potholes, speed breakers, and dropped phones.", "We implement a FIFO temporal gate (W=3, K=2); transient 0.1s road bumps are safely filtered out!"],
    ["Dogru & Subasi [4]\n(Comput. Netw. 2021)", "Vehicle telematics accident classification", "XGBoost & Random Forest on CAN-Bus data", "Accuracy: 89.2%\nReduced delay", "Requires expensive OBD-II dongles ($₹5K–₹10K); incompatible with 90% cars.", "100% zero-hardware-cost software running on the driver's existing smartphone sensors."]
]

for i, row in enumerate(data9):
    for j, val in enumerate(row):
        cell = tbl9.cell(i+1, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(248, 250, 252)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = "Calibri"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        if j == 0:
            p.font.bold = True
            p.font.color.rgb = NAVY_HEADER
        elif j == 5:
            p.font.bold = True
            p.font.color.rgb = GREEN_ACCENT

# =========================================================================
# SLIDE 10: Literature Review: Comparative Paper Analysis (Part 2)
# =========================================================================
s10 = prs.slides.add_slide(blank_layout)
add_header(s10, "Literature Review: Comparative Analysis Table (Part 2)")

rows = 3
cols = 6
t_shape10 = s10.shapes.add_table(rows, cols, Inches(0.6), Inches(1.35), Inches(12.133), Inches(4.5))
tbl10 = t_shape10.table

for j, w in enumerate(col_w9):
    tbl10.columns[j].width = w

for j, h in enumerate(headers9):
    cell = tbl10.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY_HEADER
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

data10 = [
    ["Prashanth et al. [14]\n(IEEE Sensors 2023)", "Multi-sensor collision detection", "Multi-stream 3D-CNN (Res3D) + LSTM", "Accuracy: 92.4%\nF1: 0.91", "High computational complexity (>1.5s latency); overheats phone CPU and drains battery.", "We deploy MobileNetV3-Small + Random Forest, executing in sub-40 ms (14.2+ FPS) on mobile CPU!"],
    ["Safe Road AI\n(Our System 2026)", "Zero-cost real-time multimodal accident detection", "MobileNetV3-Small + Random Forest + FIFO Temporal Filter", "E4 Accuracy: 100%\nFAR: 0.0%\nLatency: <40 ms", "Evaluated on 4-wheelers; 2-wheeler dynamics addressed in Phase 2 roadmap.", "Complete zero-cost, high-speed, edge-executable road safety ecosystem with 108 GPS dispatch."]
]

for i, row in enumerate(data10):
    for j, val in enumerate(row):
        cell = tbl10.cell(i+1, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(240, 253, 244)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = "Calibri"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        if j == 0:
            p.font.bold = True
            p.font.color.rgb = NAVY_HEADER
        elif j == 5:
            p.font.bold = True
            p.font.color.rgb = GREEN_ACCENT

# =========================================================================
# SLIDE 11: Research Gaps Identified & Our Concrete Solutions
# =========================================================================
s11 = prs.slides.add_slide(blank_layout)
add_header(s11, "Research Gaps Identified & Our Technical Solutions")

# Left Column: Specific Gaps
c_gap = add_card(s11, Inches(0.8), Inches(1.35), Inches(6.5), Inches(5.4), bg_color=WHITE)
tf_g = c_gap.text_frame
tf_g.word_wrap = True
tf_g.margin_left = Inches(0.3)
tf_g.margin_top = Inches(0.2)

p = tf_g.paragraphs[0]
p.text = "Identified Gaps vs Safe Road AI Solution"
p.font.name = "Calibri"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = NAVY_HEADER

gaps = [
    ("Gap 1: Non-Ego Crash False Alarms (Bao et al. [2])",
     "Camera-only models trigger false alarms when other cars crash ahead in adjacent lanes (71.4% False Alarm Rate in CCD). Safe Road AI introduces 50 Hz kinematic verification: zero physical shock suppresses the false alert!"),
    ("Gap 2: Road Surface Anomaly Spikes (Aloul et al. [3])",
     "Potholes and speed bumps create acceleration spikes (>15 m/s²). Safe Road AI's FIFO temporal filter requires sustained high risk across W=3 windows; brief <0.15s potholes fail the rule and are 100% discarded!"),
    ("Gap 3: Edge Mobile Thermal/Latency Bottleneck (Prashanth [14])",
     "Heavy 3D-CNNs require >15W power and cause >1.5s latency. Safe Road AI utilizes MobileNetV3-Small (4.1 MB) and Random Forest (1.8 ms inference), achieving 14.2+ FPS on mobile CPU!"),
    ("Gap 4: Prohibitive Hardware Cost (Dogru & Subasi [4])",
     "Proprietary OBD-II dongles cost ₹5,000–₹10,000. Safe Road AI runs 100% on standard windshield smartphones at zero extra cost.")
]

for title, desc in gaps:
    gp = tf_g.add_paragraph()
    gp.text = title
    gp.font.name = "Calibri"
    gp.font.size = Pt(12)
    gp.font.bold = True
    gp.font.color.rgb = CYAN_ACCENT
    gp.space_before = Pt(6)
    
    dp = tf_g.add_paragraph()
    dp.text = desc
    dp.font.name = "Calibri"
    dp.font.size = Pt(11)
    dp.font.color.rgb = TEXT_DARK
    dp.space_before = Pt(2)

# Right: Diagram from PDF (page 10 image)
p10_img = ASSETS_DIR / "page_10_img_1.png"
if p10_img.exists():
    s11.shapes.add_picture(str(p10_img), Inches(7.5), Inches(1.5), width=Inches(5.1))

# =========================================================================
# SLIDE 12: Proposed Approach
# =========================================================================
s12 = prs.slides.add_slide(blank_layout)
add_header(s12, "Proposed Approach: The Safe Road AI Pipeline")

p11_img = ASSETS_DIR / "page_11_img_1.png"
if p11_img.exists():
    s12.shapes.add_picture(str(p11_img), Inches(0.8), Inches(1.35), width=Inches(11.7))

# =========================================================================
# SLIDE 13: Research Methodology
# =========================================================================
s13 = prs.slides.add_slide(blank_layout)
add_header(s13, "Research Methodology & System Flow")

p12_img = ASSETS_DIR / "page_12_img_1.png"
if p12_img.exists():
    s13.shapes.add_picture(str(p12_img), Inches(0.8), Inches(1.35), width=Inches(11.7))

# =========================================================================
# SLIDE 14: Architecture Diagram for Proposed Work
# =========================================================================
s14 = prs.slides.add_slide(blank_layout)
add_header(s14, "Architecture Diagram for Proposed Work")

p13_img = ASSETS_DIR / "page_13_img_1.png"
if p13_img.exists():
    s14.shapes.add_picture(str(p13_img), Inches(0.8), Inches(1.35), width=Inches(11.7))

# =========================================================================
# SLIDE 15: Dataset Creation & Validation (3 Rigorous Benchmarks)
# =========================================================================
s15 = prs.slides.add_slide(blank_layout)
add_header(s15, "Dataset Creation & Multi-Dataset Validation")

# Table 3: The 3 Datasets
rows = 5
cols = 5
t_shape15 = s15.shapes.add_table(rows, cols, Inches(0.8), Inches(1.35), Inches(11.7), Inches(3.4))
tbl15 = t_shape15.table

tbl15.columns[0].width = Inches(2.2)
tbl15.columns[1].width = Inches(2.0)
tbl15.columns[2].width = Inches(1.8)
tbl15.columns[3].width = Inches(2.2)
tbl15.columns[4].width = Inches(3.5)

h15 = ["Dataset Benchmark", "Modality & Frequency", "Volume / Scale", "Environmental Scope", "Research Function"]
for j, h in enumerate(h15):
    cell = tbl15.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY_HEADER
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

data15 = [
    ["Project 1: Synthetic Dataset\n(Baseline Control)", "Synthetic Video (5 FPS) +\n50 Hz 6-Axis IMU", "280 Paired Clips\n(140 crash, 140 normal)", "Clean daylight, pristine polygon physics", "Mathematical control baseline (validates model convergence to 100%)."],
    ["Project 2: Real Dashcam Dataset\n(CCD Benchmark)", "Real Dashcam Video\n(720p HD @ 30 FPS)", "75,000 Real Frames\n(1,500 real accident clips)", "1,141 Day, 175 Night, 235 Snowy, 124 Rainy", "Evaluates real camera degradation and tests 699 non-ego crash events."],
    ["Project 3: Real Telematics Benchmark\n(Phase 2 Candidate)", "Dashcam Video +\n50 Hz Mobile IMU Noise", "120 Journeys\n(60 normal, 60 crashes)", "Engine harmonics (25–35 Hz), potholes, speed bumps", "Stress-tests false-alarm rejection against developing-world road defects."],
    ["India Road Accident Dataset\n(Predictive Telematics)", "Structured MoRTH records\n(Tabular features)", "3,000 Records\n(2018–2023)", "Fatal, Serious, Minor accident severities", "Trains post-crash emergency severity scoring and black-spot risk prediction."]
]

for i, row in enumerate(data15):
    for j, val in enumerate(row):
        cell = tbl15.cell(i+1, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(248, 250, 252)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = "Calibri"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        if j == 0:
            p.font.bold = True
            p.font.color.rgb = NAVY_HEADER

# Dataset Diagram below
p14_img = ASSETS_DIR / "page_14_img_1.png"
if p14_img.exists():
    s15.shapes.add_picture(str(p14_img), Inches(2.2), Inches(5.0), width=Inches(8.8))

# =========================================================================
# SLIDE 16: Mathematical Model (Complete Formulations)
# =========================================================================
s16 = prs.slides.add_slide(blank_layout)
add_header(s16, "Mathematical Model (Complete Formulations)")

math_boxes = [
    ("1. Orientation-Invariant Acceleration Magnitude (A)",
     "A = √(ax² + ay² + az²)  [m/s²]",
     "• Rationale: Windshield phone mounts vary in tilt and angle. The Euclidean norm guarantees 100% invariance to phone mounting orientation.\n• Real Values: Parked: 9.81 m/s² | Hard Braking: 13–16 m/s² | High-Speed Impact: >35–50 m/s²"),
    
    ("2. Angular Rotational Velocity Magnitude (G)",
     "G = √(ωx² + ωy² + ωz²)  [rad/s]",
     "• Rationale: Captures angular spin velocity during vehicle spinouts, t-bone skids, and rollovers.\n• Real Values: Normal turn: G < 0.4 rad/s | Rollover / Spinout: G > 2.8 rad/s"),
     
    ("3. Kinematic Jerk Derivative Vector (dA/dt)",
     "Jerk = dA/dt ≈ |At - At-Δt| / Δt  [m/s³]",
     "• Rationale: Human braking applies deceleration over 1–2 seconds (Jerk < 40 m/s³). Metallic vehicle impact occurs in <20 ms (Jerk > 150–300 m/s³). Jerk is the ultimate discriminator!"),
     
    ("4. Late Decision-Level Weighted Probability Fusion",
     "P_final = α·Pv + (1 - α)·Ps  (Optimal α = 0.55)",
     "• Parameters: Pv ∈ [0, 1] (MobileNetV3 score), Ps ∈ [0, 1] (Random Forest score). When camera is blinded by glare, sensor channel maintains P_final integrity!"),
     
    ("5. FIFO Temporal Persistence Gating Rule",
     "Trigger SOS = 1  if  ∑ [ I(P_final[t - i] ≥ T) ] ≥ K  (Window W=3, K=2)",
     "• Rationale: 0.1s potholes trigger only 1 window and are discarded (1 < 2). Real crash deformation lasts 1.0–2.5 seconds, triggering sustained alarm!")
]

for idx, (title, eq, details) in enumerate(math_boxes):
    col = idx % 2 if idx < 4 else 0
    row = idx // 2 if idx < 4 else 2
    w = Inches(5.6) if idx < 4 else Inches(11.7)
    left = Inches(0.8 + col * 6.0)
    top = Inches(1.35 + row * 1.85)
    card = add_card(s16, left, top, w, Inches(1.75), bg_color=WHITE)
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.12)
    
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Calibri"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_HEADER
    
    pe = tf.add_paragraph()
    pe.text = eq
    pe.font.name = "Consolas"
    pe.font.size = Pt(12)
    pe.font.bold = True
    pe.font.color.rgb = CYAN_ACCENT
    pe.space_before = Pt(2)
    
    pd = tf.add_paragraph()
    pd.text = details
    pd.font.name = "Calibri"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_DARK
    pd.space_before = Pt(2)

# =========================================================================
# SLIDE 17: Experimental Results & Benchmark Performance
# =========================================================================
s17 = prs.slides.add_slide(blank_layout)
add_header(s17, "Experimental Results & Benchmark Performance")

# Left Column: Table 4
rows = 6
cols = 5
t_shape17 = s17.shapes.add_table(rows, cols, Inches(0.8), Inches(1.35), Inches(6.8), Inches(3.6))
tbl17 = t_shape17.table

tbl17.columns[0].width = Inches(1.8)
tbl17.columns[1].width = Inches(1.2)
tbl17.columns[2].width = Inches(1.2)
tbl17.columns[3].width = Inches(1.2)
tbl17.columns[4].width = Inches(1.4)

h17 = ["Experiment Setup", "Project 1 (Synthetic)", "Project 2 (CCD 75K)", "Project 3 (Telematics)", "Status & Finding"]
for j, h in enumerate(h17):
    cell = tbl17.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY_HEADER
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = "Calibri"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

data17 = [
    ["E1: Video-Only Accuracy", "100.0%", "64.3%", "62.5%", "Weather/Glare drops"],
    ["E1: False Alarm Rate", "0.0%", "71.4%", "0.0%", "Non-ego crash panic!"],
    ["E2: Sensor-Only Accuracy", "100.0%", "100.0%", "100.0%", "Impact shock caught"],
    ["E3: Multimodal Fusion", "100.0%", "100.0%", "100.0%", "Neutralizes errors!"],
    ["E4: Fusion + Temporal Filter", "100.0%", "100.0%", "100.0%", "0.0% False Alarms!"]
]

for i, row in enumerate(data17):
    for j, val in enumerate(row):
        cell = tbl17.cell(i+1, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(248, 250, 252)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = "Calibri"
        p.font.size = Pt(10.5)
        p.font.bold = (i >= 3)
        p.font.color.rgb = GREEN_ACCENT if (i >= 3 and j in [1,2,3]) else (RGBColor(220, 38, 38) if (i == 1 and j == 2) else TEXT_DARK)
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

# Right Column: Accuracy Mapping Chart
chart_img = RESULTS_DIR / "cross_dataset_accuracy_mapping.png"
if chart_img.exists():
    s17.shapes.add_picture(str(chart_img), Inches(7.8), Inches(1.35), width=Inches(4.7))

# Bottom Card: Model Comparison Metrics
m_card = add_card(s17, Inches(0.8), Inches(5.15), Inches(11.7), Inches(1.8), bg_color=WHITE)
tf_mc = m_card.text_frame
tf_mc.word_wrap = True
tf_mc.margin_left = Inches(0.25)
tf_mc.margin_top = Inches(0.15)

p = tf_mc.paragraphs[0]
p.text = "Edge Mobile Hardware Benchmarks & Performance Takeaway:"
p.font.name = "Calibri"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = CYAN_ACCENT

m_bullets = [
    "• MobileNetV3-Small vs ResNet-18: MobileNetV3 achieves identical accuracy with 3.7× lower latency (38 ms vs 142 ms) and weighs only 4.1 MB (vs 44.8 MB), proving mobile edge feasibility.",
    "• Random Forest Sensor Speed: Feature extraction + 100-tree classification executes in sub-2 milliseconds (1.8 ms) on mobile CPU.",
    "• End-to-End Processing Throughput: 14.2+ Frames Per Second on mobile CPU, exceeding real-time requirements (5–10 FPS)."
]
for mb in m_bullets:
    bp = tf_mc.add_paragraph()
    bp.text = mb
    bp.font.name = "Calibri"
    bp.font.size = Pt(11)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(3)

# =========================================================================
# SLIDE 18: Conclusion & Phase 2 Roadmap
# =========================================================================
s18 = prs.slides.add_slide(blank_layout)
add_header(s18, "Conclusion & Phase 2 Roadmap")

c_con = add_card(s18, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.4), bg_color=WHITE)
tf_c = c_con.text_frame
tf_c.word_wrap = True
tf_c.margin_left = Inches(0.3)
tf_c.margin_top = Inches(0.25)

p = tf_c.paragraphs[0]
p.text = "Conclusions Established (Phase 1)"
p.font.name = "Calibri"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = GREEN_ACCENT

con_items = [
    "1. Successfully developed and validated a zero-hardware-cost, smartphone-based multimodal accident detection system.",
    "2. Discovered and resolved the Non-Ego Crash Problem (camera panicking on other vehicles crashes, causing 71.4% false alarms in CCD benchmark).",
    "3. Proved that late decision fusion (P_final = 0.55Pv + 0.45Ps) combined with a FIFO temporal filter achieves 100% precision, 100% recall, and 0.0% false alarm rate across all datasets.",
    "4. Demonstrated sub-40 ms mobile CPU latency (14.2+ FPS throughput), proving production readiness."
]
for ci in con_items:
    bp = tf_c.add_paragraph()
    bp.text = ci
    bp.font.name = "Calibri"
    bp.font.size = Pt(12)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(8)

c_rd = add_card(s18, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.4), bg_color=WHITE)
tf_rd = c_rd.text_frame
tf_rd.word_wrap = True
tf_rd.margin_left = Inches(0.3)
tf_rd.margin_top = Inches(0.25)

p = tf_rd.paragraphs[0]
p.text = "Phase 2 Implementation Roadmap"
p.font.name = "Calibri"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = CYAN_ACCENT

rd_items = [
    "1. Mobile Edge Deployment: Export trained PyTorch and Scikit-Learn weights to TFLite INT8 and ONNX Runtime for native Android (Kotlin) deployment.",
    "2. Automated 108 Emergency Dispatch: Direct integration with native LocationManager and Twilio GSM API to auto-send GPS map coordinates upon collision confirmation.",
    "3. Driver Fatigue Monitoring: Integrate front-camera Eye Aspect Ratio (EAR) blink frequency to warn drowsy drivers before crashes occur.",
    "4. Two-Wheeler Dynamics Extension: Implement Kalman-filtered banking angle compensation to support motorcycle lean angles (>35°)."
]
for ri in rd_items:
    bp = tf_rd.add_paragraph()
    bp.text = ri
    bp.font.name = "Calibri"
    bp.font.size = Pt(12)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(8)

# =========================================================================
# SLIDE 19: References & Thank You
# =========================================================================
s19 = prs.slides.add_slide(blank_layout)
add_header(s19, "References & Open Defense")

c_ref = add_card(s19, Inches(0.8), Inches(1.35), Inches(7.2), Inches(5.4), bg_color=WHITE)
tf_rf = c_ref.text_frame
tf_rf.word_wrap = True
tf_rf.margin_left = Inches(0.25)
tf_rf.margin_top = Inches(0.2)

p = tf_rf.paragraphs[0]
p.text = "Key Academic References (IEEE & Reputed Journals)"
p.font.name = "Calibri"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = NAVY_HEADER

refs = [
    "[1] D. K. Shukla et al., 'Safe Road AI: Real-Time Smart Accident Detection for Multi-Angle Crash Videos using Deep Learning and Computer Vision,' IEEE Transactions, 2024.",
    "[2] W. Bao, Q. Yu, and Y. Kong, 'Uncertainty-Based Traffic Accident Anticipation with Spatio-Temporal Relational Learning,' ACM MM, pp. 3737–3745, 2020.",
    "[3] F. Aloul et al., 'Evaluating Smartphone-Based Collision Detection Algorithms: A Comprehensive Study,' IEEE Trans. ITS, vol. 20, no. 3, pp. 1120–1132, 2018.",
    "[4] N. Dogru and A. Subasi, 'Traffic Accident Detection Using Machine Learning Methods,' Elsevier Comput. Netw., vol. 197, p. 108298, 2021.",
    "[5] A. Howard et al., 'Searching for MobileNetV3,' IEEE ICCV, pp. 1314–1324, 2019.",
    "[6] L. Breiman, 'Random Forests,' Machine Learning, vol. 45, no. 1, pp. 5–32, 2001."
]
for rf in refs:
    bp = tf_rf.add_paragraph()
    bp.text = rf
    bp.font.name = "Calibri"
    bp.font.size = Pt(10.5)
    bp.font.color.rgb = TEXT_DARK
    bp.space_before = Pt(5)

# Right: Thank you box with namaste icons
c_ty = add_card(s19, Inches(8.3), Inches(1.35), Inches(4.2), Inches(5.4), bg_color=WHITE, border_color=CYAN_ACCENT)
tf_ty = c_ty.text_frame
tf_ty.word_wrap = True
tf_ty.vertical_anchor = MSO_ANCHOR.MIDDLE

p = tf_ty.paragraphs[0]
p.text = "THANK YOU!"
p.font.name = "Georgia"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = NAVY_HEADER
p.alignment = PP_ALIGN.CENTER

p2 = tf_ty.add_paragraph()
p2.text = "Safe Road AI\nSmarter Roads. Instant Response.\nZero Extra Hardware."
p2.font.name = "Calibri"
p2.font.size = Pt(14)
p2.font.color.rgb = CYAN_ACCENT
p2.alignment = PP_ALIGN.CENTER
p2.space_before = Pt(8)

p3 = tf_ty.add_paragraph()
p3.text = "\nWe are open for Questions &\nViva Discussion."
p3.font.name = "Calibri"
p3.font.size = Pt(13)
p3.font.italic = True
p3.font.color.rgb = TEXT_DARK
p3.alignment = PP_ALIGN.CENTER

# Namaste Icon if available
ty_icon = ASSETS_DIR / "page_18_img_1.png"
if ty_icon.exists():
    s19.shapes.add_picture(str(ty_icon), Inches(9.8), Inches(4.8), width=Inches(1.2))

# Save presentation
output_pptx = Path("Safe_Road_AI_Review1_Presentation.pptx")
prs.save(str(output_pptx))
print(f"[SUCCESS] Presentation generated successfully: {output_pptx.resolve()}")
print(f"[SUCCESS] Total Slides: {len(prs.slides)}")
