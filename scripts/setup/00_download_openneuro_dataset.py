"""
00_download_openneuro_dataset.py -- setup: download the overground dataset from OpenNeuro.

Purpose
    Download OpenNeuro ds003039 (Jacobsen et al., overground walking)
    into datasets/jacobsen/d00_raw/ with openneuro-py. The treadmill
    dataset is not on OpenNeuro and is placed in datasets/stepup/d00_raw/
    by hand.
Inputs
    none (network)
Outputs
    datasets/jacobsen/d00_raw/ (BIDS)
Manuscript
    Methods 2.1 (datasets).
Run
    python scripts/setup/00_download_openneuro_dataset.py
"""

from src.config import config_jacobsen
from src.paths import get_dataset_dirs


def main():
    import openneuro as on  # imported here: openneuro-py fails at import in some environments

    on.download(dataset=config_jacobsen.ACCESSION, target_dir=get_dataset_dirs("jacobsen")["raw"])


if __name__ == "__main__":
    main()
