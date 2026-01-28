# utils/common.py

from __future__ import annotations

import os
import csv
import random
from pathlib import Path
from typing import Dict, Any, Union, Optional

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: random seed
        deterministic: if True, tries to make CUDA ops deterministic (can be slower)
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)


def append_csv(path: Union[str, Path], row: Dict[str, Any], fieldnames: Optional[list] = None) -> None:
    """
    Append one dict row to CSV. Creates file + header if needed.

    Args:
        path: output csv path
        row: dictionary row
        fieldnames: optional explicit field order (recommended for stable columns)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    if fieldnames is None:
        fieldnames = list(row.keys())

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
