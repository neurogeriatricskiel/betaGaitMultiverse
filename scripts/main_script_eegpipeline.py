"""
main_script_eegpipeline.py -- run the reference pipeline (stages 1-7) for the active dataset.

Purpose
    Runs prepana01-07 in sequence, each in its own Python process, for
    all subjects of the active dataset (src.config.SUBJECTS). Each stage
    can also be run on its own (see its header).
Inputs / outputs
    See each prepana script.
Manuscript
    Methods 2.2-2.8 (reference pipeline).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/main_script_eegpipeline.py
    (ASR follows the dataset's reference setting; BETAGAIT_USE_ASR overrides it)
"""

import subprocess
import sys
import time

from src.config import (
    ASR_CUTOFF,
    DATASET,
    DIR_SCRIPTS,
    REFERENCE_HIGHPASS_HZ,
    REFERENCE_LOWPASS_HZ,
    SUBJECTS,
    USE_ASR,
)

STAGES = [
    (1, "prepana01_prep_gaitevents.py", "gait event detection"),
    (2, "prepana02_raw2ica.py", "preprocessing + ICA fit"),
    (3, "prepana03_ica2clean.py", "ICLabel + ICA apply"),
    (4, "prepana04_clean2gaitcycles.py", "gait segment extraction"),
    (5, "prepana05_gaitcycles2tfr.py", "TFR + ERSP"),
    (6, "prepana06_plotbetagait.py", "group ERSP figure"),
    (7, "prepana07_betaphase_stats.py", "beta phase group t-test"),
]


def main() -> None:
    """Run prepana01-07 in order, one subprocess per stage."""
    print(f"Dataset  : {DATASET}")
    print(f"Subjects : {SUBJECTS}")
    print(f"Band-pass: {REFERENCE_HIGHPASS_HZ:g}-{REFERENCE_LOWPASS_HZ:g} Hz (FIR)")
    print(f"ASR      : {f'SD {ASR_CUTOFF:g}' if USE_ASR else 'off'}")
    results = []
    t_total = time.time()
    for stage_num, script_name, description in STAGES:
        print(f"\n{'=' * 60}\nStage {stage_num}/7 -- {description}\n{'=' * 60}")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(DIR_SCRIPTS / script_name)], cwd=str(DIR_SCRIPTS.parent)
        )
        ok = result.returncode == 0
        status = "DONE" if ok else f"FAILED (code {result.returncode})"
        print(f"\n[{status}] Stage {stage_num} in {time.time() - t0:.1f}s")
        results.append((stage_num, description, ok))

    print(f"\n{'=' * 60}\nPipeline complete in {(time.time() - t_total) / 60:.1f} min\n{'=' * 60}")
    for stage_num, description, ok in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] Stage {stage_num}: {description}")


if __name__ == "__main__":
    main()
