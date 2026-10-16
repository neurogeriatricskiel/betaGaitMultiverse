"""Per-dataset directory lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import DATASETS, define_dir

if TYPE_CHECKING:
    from pathlib import Path


def get_dataset_dirs(dataset: str) -> dict[str, Path]:
    """Return the directories of one dataset, creating them if needed.

    Parameters
    ----------
    dataset : str
        Key in ``src.config.DATASETS``.

    Returns
    -------
    dict of str to Path
        Directory key (as in the dataset's ``"dirs"`` config) to path.
    """
    cfg = DATASETS[dataset]
    root = cfg["root"]
    dirs = {}
    for key, folder in cfg["dirs"].items():
        dirs[key] = define_dir(root, folder)
    return dirs
