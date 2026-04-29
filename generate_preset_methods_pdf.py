#!/usr/bin/env python3
"""Generate preset_generation_methods.pdf documenting the two new XMP strategies."""

from fpdf import FPDF


class MethodsPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "XMP Preset Generation Methods", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def code_block(self, text):
        self.set_font("Courier", "", 8)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 4, text, fill=True)
        self.ln(2)


def build_pdf():
    pdf = MethodsPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- Page 1: Overview ----
    pdf.add_page()

    pdf.section_title("1. Overview")
    pdf.body_text(
        "This document describes two improved methods for generating Adobe Lightroom "
        "XMP preset values from image analysis. Both methods replace the original "
        "hardcoded scale-factor approach with more accurate techniques.\n\n"
        "Available methods:\n"
        "  - basic: Original hardcoded LAB/HSV delta mapping (default)\n"
        "  - color_science: Proper color science via colour-science library\n"
        "  - optimization: Forward model + scipy.optimize"
    )

    pdf.body_text(
        "Usage:\n"
        '  generator = XMPPresetGenerator()\n'
        '  params = generator.extract_params(target, reference, strength=1.0, method="color_science")\n'
        '  # or method="optimization"'
    )

    # ---- Method 1: Color Science ----
    pdf.section_title("2. Color Science Method")
    pdf.body_text(
        "Uses the colour-science Python library for physically accurate color "
        "calculations instead of hardcoded multipliers."
    )

    pdf.sub_title("2.1 Temperature (CCT)")
    pdf.body_text(
        "Problem: The basic method uses LAB b-channel shift * 100 to estimate "
        "color temperature in Kelvin. This is a rough linear approximation.\n\n"
        "Solution: Convert mean image color to CIE XYZ, then to CIE 1960 UCS uv "
        "chromaticity, then use the Ohno 2013 method to compute Correlated Color "
        "Temperature (CCT). The CCT difference between target and reference images "
        "gives the temperature slider value directly in Kelvin."
    )
    pdf.code_block(
        "xyz = colour.sRGB_to_XYZ(mean_rgb)\n"
        "ucs = colour.XYZ_to_UCS(xyz)\n"
        "uv = colour.UCS_to_uv(ucs)\n"
        "cct, duv = colour.uv_to_CCT(uv, method='Ohno 2013')"
    )

    pdf.sub_title("2.2 Tint (Duv)")
    pdf.body_text(
        "Problem: The basic method uses LAB a-channel shift * 3.5 for tint.\n\n"
        "Solution: Duv (distance from the Planckian locus) directly measures the "
        "green-magenta deviation, which is exactly what Lightroom's tint slider "
        "controls. Duv ranges from about -0.02 to +0.02 in practice; scaling by "
        "7500 maps this to Lightroom's -150 to +150 range."
    )

    pdf.sub_title("2.3 Exposure (log2)")
    pdf.body_text(
        "Problem: The basic method uses (L_shift / 255) * 6.0 which is linear.\n\n"
        "Solution: Photography exposure is logarithmic (each stop = 2x light). "
        "Convert LAB L* to relative luminance Y using the CIE definition:\n"
        "  Y = ((L* + 16) / 116)^3  for L* > 7.9996\n"
        "Then compute: exposure_EV = log2(Y_ref / Y_target)\n\n"
        "This directly gives the exposure difference in EV stops, which is what "
        "Lightroom's exposure slider measures."
    )
    pdf.code_block(
        "target_Y = l_to_luminance(mean_L_target)\n"
        "ref_Y = l_to_luminance(mean_L_ref)\n"
        "exposure_ev = log2(ref_Y / target_Y)"
    )

    pdf.sub_title("2.4 Contrast (std ratio)")
    pdf.body_text(
        "Problem: Basic method uses raw difference of L standard deviations.\n\n"
        "Solution: Use the ratio of L* standard deviations instead:\n"
        "  contrast = ((ref_std / target_std) - 1.0) * 100\n\n"
        "This is multiplicative rather than additive, matching how Lightroom's "
        "contrast slider works (it scales the spread around the midpoint)."
    )

    pdf.sub_title("2.5 Saturation (CIELAB chroma)")
    pdf.body_text(
        "Problem: Basic method uses HSV saturation which is not perceptually uniform.\n\n"
        "Solution: Use CIELAB chroma C* = sqrt(a*^2 + b*^2) which is perceptually "
        "uniform. Compare the ratio of mean chromas between reference and target."
    )

    pdf.sub_title("2.6 Vibrance (low-chroma targeting)")
    pdf.body_text(
        "Problem: Basic method computes vibrance as saturation * 0.6.\n\n"
        "Solution: Vibrance targets low-saturation pixels more than high-saturation "
        "ones. Compute chroma shift for only the bottom 50th percentile of chroma "
        "values, giving a more accurate vibrance estimate."
    )

    # ---- Method 2: Optimization ----
    pdf.add_page()
    pdf.section_title("3. Optimization Method")
    pdf.body_text(
        "Uses a simplified forward model of Lightroom's processing pipeline combined "
        "with scipy.optimize to find slider values that best reproduce the reference "
        "image when applied to the target image."
    )

    pdf.sub_title("3.1 Forward Model")
    pdf.body_text(
        "A LightroomForwardModel class approximates 14 key Lightroom operations in "
        "LAB color space:\n\n"
        "  1. Exposure: Convert L* to luminance Y, scale by 2^EV, convert back\n"
        "  2. Contrast: Expand/compress L around the midpoint\n"
        "  3. Highlights: Selectively adjust top-quarter luminance\n"
        "  4. Shadows: Selectively adjust bottom-quarter luminance\n"
        "  5. Whites: Adjust extreme high luminance\n"
        "  6. Blacks: Adjust extreme low luminance\n"
        "  7. Tone Curve: Parametric adjustments to 4 luminance zones\n"
        "  8. Temperature: Shift LAB b channel (blue-yellow axis)\n"
        "  9. Tint: Shift LAB a channel (green-magenta axis)\n"
        " 10. Saturation: Scale LAB chroma uniformly\n"
        " 11. Vibrance: Scale chroma with inverse-chroma weighting"
    )

    pdf.sub_title("3.2 Optimization Process")
    pdf.body_text(
        "1. Get initial guess from ColorScienceStrategy (warm start)\n"
        "2. Downsample both images to 256px max dimension for speed\n"
        "3. Define cost function: weighted MSE between forward model output and "
        "reference image in LAB space (L weighted 2x vs A/B)\n"
        "4. Run L-BFGS-B optimization with max 200 iterations\n"
        "5. Merge optimized values for the 14 key sliders with ColorScience "
        "values for the remaining 57 sliders (HSL, color grading, sharpening)"
    )
    pdf.code_block(
        "result = scipy.optimize.minimize(\n"
        "    cost, x0, method='L-BFGS-B',\n"
        "    bounds=slider_bounds,\n"
        "    options={'maxiter': 200, 'ftol': 1e-6}\n"
        ")"
    )

    pdf.sub_title("3.3 Why L-BFGS-B?")
    pdf.body_text(
        "L-BFGS-B is chosen because:\n"
        "  - Supports box constraints (slider min/max bounds)\n"
        "  - Memory-efficient for moderate-dimensional problems (14 params)\n"
        "  - Good convergence on smooth, continuous cost functions\n"
        "  - Much faster than global optimizers for this problem size"
    )

    pdf.sub_title("3.4 Fallback Behavior")
    pdf.body_text(
        "If optimization fails (convergence issues, numerical errors), the method "
        "falls back to ColorScienceStrategy results automatically."
    )

    # ---- Comparison Table ----
    pdf.add_page()
    pdf.section_title("4. Comparison")

    pdf.set_font("Helvetica", "B", 9)
    col_w = [40, 50, 50, 50]
    headers = ["Aspect", "Basic", "Color Science", "Optimization"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    rows = [
        ["Temperature", "b_shift * 100", "CCT via Ohno 2013", "Optimized"],
        ["Tint", "a_shift * 3.5", "Duv from Planckian", "Optimized"],
        ["Exposure", "L/255 * 6.0", "log2(Y_ref/Y_tgt)", "Optimized"],
        ["Contrast", "std diff * 2.0", "std ratio * 100", "Optimized"],
        ["Saturation", "HSV sat diff", "CIELAB chroma ratio", "Optimized"],
        ["Vibrance", "sat * 0.6", "Low-chroma targeting", "Optimized"],
        ["Speed", "Fast (<100ms)", "Fast (<200ms)", "Slower (~1-3s)"],
        ["Accuracy", "Rough approx", "Good for key sliders", "Best overall"],
        ["Dependencies", "numpy, opencv", "+ colour-science", "+ scipy"],
    ]
    for row in rows:
        for i, cell in enumerate(row):
            pdf.cell(col_w[i], 6, cell, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.section_title("5. When to Use Each Method")
    pdf.body_text(
        "basic: Default for backward compatibility. Good enough when exact "
        "preset accuracy is not critical.\n\n"
        "color_science: Recommended when you need better accuracy without "
        "added latency. Best improvement-to-cost ratio.\n\n"
        "optimization: Use when preset accuracy is paramount and you can "
        "tolerate 1-3 seconds of extra processing time per image."
    )

    # Save
    output_path = "preset_generation_methods.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    build_pdf()
