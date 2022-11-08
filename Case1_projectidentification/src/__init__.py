# Stores link to root directory regardless of built
import os
from pathlib import Path

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        default=os.path.join(os.path.abspath(__file__), "..", "..", "data")
    )
).resolve()
