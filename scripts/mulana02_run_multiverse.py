"""
mulana02_run_multiverse.py -- multiverse, step 2: run all universes.

Purpose
    Run every universe script created by mulana01 sequentially
    (COMET Multiverse.run(parallel=1)). Each universe processes
    MULTIVERSE_SUBJECTS through src.multiverse_pipeline (ICA cached per
    (highpass_hz, asr_mode) branch under results/multiverse/<ds>/branches/).
    COMET deletes the previous comet/results/*.pkl (except
    forking_paths.pkl) before running.
Inputs
    results/multiverse/<ds>/comet/scripts/universe_*.py
Outputs
    results/multiverse/<ds>/comet/results/multiverse_results.pkl
    results/multiverse/<ds>/branches/<subject>/<branch>/ (ICA cache)
Manuscript
    Methods 2.7 (multiverse).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/mulana02_run_multiverse.py
    Runtime: ~16 h treadmill; longer overground (longer recordings).
"""

from comet.multiverse import Multiverse

from src.config import DIR_MULTIVERSE_COMET, MULTIVERSE_NAME


def get_multiverse() -> Multiverse:
    """Return the COMET handle on this dataset's multiverse (same path= as mulana01/03)."""
    return Multiverse(name=MULTIVERSE_NAME, path=str(DIR_MULTIVERSE_COMET))


def main() -> None:
    """Run every universe script of this dataset, one at a time."""
    get_multiverse().run(parallel=1)


if __name__ == "__main__":
    main()
