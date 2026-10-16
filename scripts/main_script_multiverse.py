"""
main_script_multiverse.py -- run the multiverse (mulana01-03) for the active dataset.

Purpose
    Runs in sequence, each in its own Python process:
      1. mulana01_create_multiverse.py   -- create the 27 universe scripts
      2. mulana02_run_multiverse.py      -- run all universes
      3. mulana03_visualize_multiverse.py -- specification curve
    mulana04 (zoom figure) is run separately. mulana01 refuses to delete
    existing results; pass --force to this script to start over.
Inputs / outputs
    See each mulana script.
Manuscript
    Methods 2.7 (multiverse).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/main_script_multiverse.py [--force]
"""

import subprocess
import sys
import time

from src.config import DIR_SCRIPTS

STAGES = [
    (1, "mulana01_create_multiverse.py", "clear state + create universes"),
    (2, "mulana02_run_multiverse.py", "run all universes"),
    (3, "mulana03_visualize_multiverse.py", "specification curve"),
]


def main() -> None:
    """Run mulana01-03 in order, stopping at the first failing stage."""
    force = ["--force"] if "--force" in sys.argv[1:] else []
    print("\nbetaGaitMultiverse -- multiverse pipeline")
    results = []
    t_total = time.time()
    for stage_num, script_name, description in STAGES:
        print(f"\n{'=' * 60}\nStage {stage_num}/3 -- {description}\n{'=' * 60}")
        t0 = time.time()
        extra = force if stage_num == 1 else []
        result = subprocess.run(
            [sys.executable, str(DIR_SCRIPTS / script_name), *extra], cwd=str(DIR_SCRIPTS.parent)
        )
        ok = result.returncode == 0
        status = "OK" if ok else f"FAILED (code {result.returncode})"
        print(f"\n[{status}] Stage {stage_num} in {time.time() - t0:.1f}s")
        results.append((stage_num, description, ok))
        if not ok:
            break

    print(
        f"\n{'=' * 60}\nMultiverse pipeline complete in {(time.time() - t_total) / 60:.1f} min\n"
        f"{'=' * 60}"
    )
    for stage_num, description, ok in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] Stage {stage_num}: {description}")


if __name__ == "__main__":
    main()
