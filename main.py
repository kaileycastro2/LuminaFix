"""
LuminaFix Style Transfer - Milestone 1

Simple entry point for style transfer processing.
Edit the configuration below and run: python main.py
"""

from pathlib import Path
from src.batch_processor import BatchProcessor


# =============================================================================
# CONFIGURATION - Edit these paths
# =============================================================================

# Available categories with their input/output folders
CATEGORIES = {
    "neon": {
        "input": "test_images/neon/input",
        "output": "test_images/neon/output"
    },
    "portrait": {
        "input": "test_images/portrait/input",
        "output": "test_images/portrait/output"
    }
}

# Directory containing reference images to choose from
REFERENCE_DIR = "test_images/reference"

# =============================================================================
# PROCESSING OPTIONS
# =============================================================================

# Color transfer strength (0.0 - 1.0)
# Higher = more aggressive color matching
COLOR_STRENGTH = 0.7

# Luminance transfer strength (0.0 - 1.0)
# Higher = more brightness/contrast matching (0 = no luminance shift)
LUMINANCE_STRENGTH = 0.0

# Enable skin tone protection
# Reduces color shift on detected skin regions
ENABLE_SKIN_PROTECTION = True

# Enable neon/saturation protection
# Prevents oversaturation of vibrant colors
ENABLE_NEON_PROTECTION = True

# Output JPEG quality (1-100)
JPEG_QUALITY = 95

# =============================================================================
# MAIN SCRIPT
# =============================================================================


def get_category_choice() -> str | None:
    """Display available categories and get user's choice."""
    categories = list(CATEGORIES.keys())

    print("\nAvailable categories:")
    print("-" * 40)
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    print("-" * 40)

    while True:
        try:
            choice = input(f"\nSelect category (1-{len(categories)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(categories):
                return categories[choice_num - 1]
            print(f"Please enter a number between 1 and {len(categories)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None


def get_reference_choice(reference_dir: str) -> Path | None:
    """Display available reference images and get user's choice."""
    ref_path = Path(reference_dir)

    if not ref_path.exists():
        print(f"\nERROR: Reference directory not found: {reference_dir}")
        return None

    # Find all images in reference directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    reference_images = sorted([
        f for f in ref_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not reference_images:
        print(f"\nERROR: No reference images found in {reference_dir}")
        return None

    print("\nAvailable reference images:")
    print("-" * 40)
    for i, img in enumerate(reference_images, 1):
        print(f"  {i}. {img.name}")
    print("-" * 40)

    while True:
        try:
            choice = input(f"\nSelect reference image (1-{len(reference_images)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(reference_images):
                return reference_images[choice_num - 1]
            print(f"Please enter a number between 1 and {len(reference_images)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None


def main():
    print("=" * 60)
    print("LuminaFix Style Transfer - Milestone 1")
    print("=" * 60)

    # Get user's category choice
    category = get_category_choice()
    if category is None:
        return

    # Get input/output paths for selected category
    input_dir = CATEGORIES[category]["input"]
    output_dir = CATEGORIES[category]["output"]

    # Validate input path
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"\nERROR: Input directory not found: {input_dir}")
        print(f"Please create the {input_dir}/ folder and add your images.")
        return

    # Get user's reference image choice
    reference_path = get_reference_choice(REFERENCE_DIR)
    if reference_path is None:
        return

    # Find target images (exclude reference)
    target_images = BatchProcessor.find_images(input_dir)
    target_images = [p for p in target_images if Path(p).resolve() != reference_path.resolve()]

    if not target_images:
        print(f"\nERROR: No target images found in {input_dir}")
        print("Supported formats: jpg, jpeg, png, bmp, tiff, webp")
        return

    # Create output subdirectory based on reference image name
    reference_name = reference_path.stem  # e.g., "reference1" from "reference1.jpg"
    output_subdir = Path(output_dir) / reference_name

    print(f"\nCategory: {category}")
    print(f"Reference: {reference_path}")
    print(f"Targets: {len(target_images)} images")
    print(f"Output: {output_subdir}/")
    print(f"\nSettings:")
    print(f"  Color strength: {COLOR_STRENGTH}")
    print(f"  Luminance strength: {LUMINANCE_STRENGTH}")
    print(f"  Skin protection: {'ON' if ENABLE_SKIN_PROTECTION else 'OFF'}")
    print(f"  Neon protection: {'ON' if ENABLE_NEON_PROTECTION else 'OFF'}")

    # Initialize processor
    processor = BatchProcessor(
        color_strength=COLOR_STRENGTH,
        luminance_strength=LUMINANCE_STRENGTH,
        enable_skin_protection=ENABLE_SKIN_PROTECTION,
        enable_neon_protection=ENABLE_NEON_PROTECTION,
        jpeg_quality=JPEG_QUALITY
    )

    # Load reference style
    print("\n" + "-" * 60)
    processor.load_reference(str(reference_path))

    # Process all targets
    print("-" * 60)
    results = processor.process_batch(target_images, str(output_subdir))

    # Final summary
    success = sum(1 for r in results if r.success)
    failed = len(results) - success

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {success} images")
    if failed > 0:
        print(f"Failed: {failed} images")
    print(f"Output saved to: {output_subdir}/")


if __name__ == "__main__":
    main()
