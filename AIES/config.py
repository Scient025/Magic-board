import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths for notes2.py
LOGO_PATH = os.path.join(BASE_DIR, "AIES", "logo.png")
COLOR_SECTION_PATH = os.path.join(BASE_DIR, "AIES", "color_section.png")
ERASER_PATH = os.path.join(BASE_DIR, "AIES", "eraser.png")
TEST_SCRIPT_PATH = os.path.join(BASE_DIR, "AIES", "test.py")
EQUATION_OUTPUT_PATH = os.path.join(BASE_DIR, "equation.png")

# Paths for test.py
MODEL_PATH = os.path.join(BASE_DIR, "AIES", "eqn-detect-model.keras")
DATA_DIR = os.path.join(BASE_DIR, "archive", "data", "extracted_images")
