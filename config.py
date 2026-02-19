
from pathlib import Path

# Classes
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
DROP_CLASS  = "UNK"
NUM_CLASSES = len(CLASS_NAMES)

# Image settings
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESIZE_PX   = 256

# Split fractions
VAL_FRACTION  = 0.10
TEST_FRACTION = 0.10
RANDOM_SEED   = 42

# Default training hyperparameters
BATCH_SIZE   = 64
NUM_EPOCHS   = 20
LR           = 2e-4
WEIGHT_DECAY = 1e-2

# DataLoader threading
NUM_WORKERS      = 0
PREFETCH_THREADS = 8     # overridden at runtime by min(8, os.cpu_count())
PREFETCH_QUEUE   = 16

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Paths (set at runtime by client.py, placeholders here)
BASE_PATH    : Path = None
IMG_ROOT     : Path = None
IMG_ROOT_256 : Path = None
GT_CSV       : Path = None
META_CSV     : Path = None
CKPT_PATH    : Path = None
DONE_FLAG    : Path = None


def set_paths(base: Path):
    """Called by client.py after the base path is confirmed."""
    global BASE_PATH, IMG_ROOT, IMG_ROOT_256, GT_CSV, META_CSV, CKPT_PATH, DONE_FLAG
    BASE_PATH    = Path(base)
    IMG_ROOT     = BASE_PATH / "ISIC_2019_Training_Input"
    IMG_ROOT_256 = BASE_PATH / "ISIC_2019_Training_Input_256"
    GT_CSV       = BASE_PATH / "ISIC_2019_Training_GroundTruth.csv"
    META_CSV     = BASE_PATH / "ISIC_2019_Training_Metadata.csv"
    CKPT_PATH    = BASE_PATH / "resnet18_isic2019_best.pt"
    DONE_FLAG    = BASE_PATH / "_resize_complete.json"
    BASE_PATH.mkdir(parents=True, exist_ok=True)