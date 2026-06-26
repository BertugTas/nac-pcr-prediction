from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
CLINICAL_DIR   = DATA_DIR / "clinical"

OUTPUTS_DIR    = BASE_DIR / "outputs"
PLOTS_DIR      = OUTPUTS_DIR / "plots"
MODELS_DIR     = OUTPUTS_DIR / "models"
RESULTS_DIR    = OUTPUTS_DIR / "results"

# DICOM ön işleme
IMG_SIZE       = (224, 224)
HU_MIN         = -1000
HU_MAX         = 400

# Model eğitimi
RANDOM_SEED    = 42
TEST_SIZE      = 0.20
VAL_SIZE       = 0.10
CV_FOLDS       = 5

# Hedef sütun (klinik tabloda pCR durumu)
TARGET_COL     = "pCR"
