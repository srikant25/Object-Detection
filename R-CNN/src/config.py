import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
current_dir = Path(__file__).resolve().parent
root_dir =current_dir.parents[0]
DATA_DIR = root_dir.parent /  "PennFudanPed"
