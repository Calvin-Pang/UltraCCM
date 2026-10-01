import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(manifest_path, split):
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if split not in manifest:
        raise KeyError(f"Split '{split}' is missing from {manifest_path}")

    base = manifest_path.parent

    def resolve(value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (base / path).resolve())

    entries = manifest[split]
    if isinstance(entries, list):
        return [resolve(path) for path in entries]
    if isinstance(entries, dict):
        return {resolve(path): bounds for path, bounds in entries.items()}
    raise TypeError(f"Split '{split}' must be a list or object")


def validate_case_paths(cases):
    paths = cases.keys() if isinstance(cases, dict) else cases
    missing = [path for path in paths if not Path(path).is_dir()]
    if missing:
        raise FileNotFoundError("Missing case directories:\n" + "\n".join(missing))
