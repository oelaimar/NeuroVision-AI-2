import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

# Project root = parent of src/.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"

def find_dir(root : Path, name : str):
    """ Find a folder called `name` anywhere under the root """
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == p.name.lower():
            return p
    raise FileNotFoundError(f"'{name}' not found under {root}")

@dataclass(frozen=True)
class CFG:

    def __post_init__(self):
        SeedEverything.seed_everything(self.SEED)
    SEED: int = 42

    PROJECT_ROOT: Path = PROJECT_ROOT
    RAW_DIR: Path = RAW_DIR
    DATA_DIR: Path = DATA_DIR
    DATA_YAML: Path = DATA_DIR / "data.yaml"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    RUNS_DIR: Path = PROJECT_ROOT / "runs"
    
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    RAW_SPLIT_DIR: Path = DATA_DIR / "raw_split"

    CLASS_NAMES: tuple = ("Glioma", "Meningioma", "No Tumor", "Pituitary")
    NUM_CLASSES = len(CLASS_NAMES)

    IMAGE_SIZE: int = 640
    BATCH_SIZE: int = 16
    EPOCHS: int = 30
    PATIENCE: int = 10
    NUM_WORKERS: int = 4
    MODEL: str = "yolov8n.pt"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_EXTS: tuple = (".jpeg", ".jpg", ".bmp", ".png")

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

class SeedEverything():
    def seed_everything(seed: int = CFG.SEED) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

cfg = CFG()

if __name__ == "__main__":
    print("Device:", cfg.DEVICE)
    print("Raw data exists:", cfg.RAW_DIR.exists())
    for k, v in cfg.to_dict().items():
        print(f"{k}: {v}")
