"""
mulana01_create_multiverse.py -- multiverse, step 1: generate the 27 universe scripts (COMET).

Purpose
    Define the decision nodes (src.config.MULTIVERSE_GRID, Table 2.1) and
    let COMET render one script per universe from analysis_template():
        highpass_hz  : 0.5 / 1.0 / 2.0 Hz FIR high-pass
        asr_mode     : off / sd3 / sd20 (ASR burst-correction cutoff, SD;
                       Mullen et al. 2015, Gorjan et al. 2022)
        iclabel_rule : conservative P(brain) > 0.9 / balanced > 0.7 /
                       liberal: reject P(muscle) or P(eye) > 0.9
                       (Pion-Tonachini et al. 2019), applied to one shared
                       ICA fit per (highpass_hz, asr_mode) branch
    Low-pass fixed at MULTIVERSE_LOWPASS_HZ. Each universe computes the
    same double-stance - swing beta contrast as the reference pipeline
    (src.multiverse_pipeline).
    Clears this dataset's previous COMET state first. Because that deletes
    comet/results/multiverse_results.pkl, the script stops if that file
    exists unless --force is given.
Inputs
    src/config.py (grid, cohorts)
Outputs
    results/multiverse/<ds>/comet/scripts/universe_*.py, multiverse_summary.csv,
    comet/results/forking_paths.pkl
Manuscript
    Methods 2.7 (multiverse).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/mulana01_create_multiverse.py [--force]
"""

import argparse
import shutil

from comet.multiverse import Multiverse

from src.config import (
    DIR_MULTIVERSE,
    DIR_MULTIVERSE_COMET,
    DIR_MULTIVERSE_OUTPUTS,
    DIR_PROJ,
    MULTIVERSE_GRID,
    MULTIVERSE_NAME,
)


# Rendered by COMET into each universe script ({{ ... }} = decision values);
# kept byte-for-byte as rendered, hence no formatting / import sorting.
# fmt: off
def analysis_template():
    import comet  # noqa: I001
    import numpy as np
    from scipy.stats import ttest_rel
    from src.config import MULTIVERSE_SUBJECTS as SUBJECTS
    from src.config import MULTIVERSE_LOWPASS_HZ
    from src.multiverse_pipeline import run_subject_multiverse
    # NOTE: comet's Jinja substitution already double-quotes string fork
    # values (see comet.multiverse.Multiverse._render_val) -- do not wrap
    # asr_mode / iclabel_rule in manual quotes here, that would double-quote.
    decisions = {
        "highpass_hz":  {{ highpass_hz }},
        "asr_mode":     {{ asr_mode }},
        "iclabel_rule": {{ iclabel_rule }},
        # Fixed low-pass (src.config.MULTIVERSE_LOWPASS_HZ), not a decision node.
        "lowpass_hz":   MULTIVERSE_LOWPASS_HZ,
    }
    subjects_ok, beta_diffs, beta_ds_list, beta_sw_list, n_ok = [], [], [], [], 0
    for subj in SUBJECTS:
        try:
            result = run_subject_multiverse(subj, decisions)
        except Exception as e:
            print(f"  sub-{subj}: FAILED - {e}")
            continue
        subjects_ok.append(subj)
        beta_diffs.append(result["beta_diff"])
        beta_ds_list.append(result["beta_double_stance"])
        beta_sw_list.append(result["beta_swing"])
        n_ok += 1

    # Group paired t-test (double_stance vs swing across subjects),
    # computed the same way as prepana07_betaphase_stats.py.
    if n_ok >= 2:
        t_stat, _ = ttest_rel(np.array(beta_ds_list), np.array(beta_sw_list))
        group_t_mean = float(t_stat)
    else:
        group_t_mean = float("nan")

    comet.utils.save_universe_results({
        "group_t_mean":  group_t_mean,   # single group-level paired t (prepana07 method)
        "t_stats":       beta_diffs,      # list of per-subject beta contrasts (dB) for CI
        "subjects_ok":   subjects_ok,     # subject IDs, positionally aligned with t_stats/
                                           # beta_ds_list/beta_sw_list -- added so any post-hoc
                                           # subject-subset re-analysis (e.g. dropping known
                                           # duplicate recordings) doesn't have to assume ordering.
        "beta_ds":       beta_ds_list,    # per-subject double-stance scalar (dB), same order
        "beta_sw":       beta_sw_list,    # per-subject swing scalar (dB), same order
        "n_subjects_ok": n_ok,
    })
# fmt: on


def clear_previous_state():
    """Remove this dataset's previous COMET state (never touches the other dataset)."""
    for pattern in ("*.pkl", "*.json"):
        for f in DIR_MULTIVERSE.glob(pattern):
            f.unlink()
            print(f"  Removed: {f.relative_to(DIR_PROJ)}")
    for f in DIR_MULTIVERSE_OUTPUTS.glob("*.pkl"):
        f.unlink()
        print(f"  Removed: {f.relative_to(DIR_PROJ)}")
    if DIR_MULTIVERSE_COMET.exists():
        shutil.rmtree(DIR_MULTIVERSE_COMET)
        print(f"  Removed: {DIR_MULTIVERSE_COMET.relative_to(DIR_PROJ)}/")
    print("  Old COMET state cleared.\n")


def main():
    parser = argparse.ArgumentParser(description="Create the 27 universe scripts (COMET).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete an existing multiverse_results.pkl and start over",
    )
    args = parser.parse_args()

    results_pkl = DIR_MULTIVERSE_COMET / "results" / "multiverse_results.pkl"
    if results_pkl.exists() and not args.force:
        raise SystemExit(
            f"{results_pkl.relative_to(DIR_PROJ)} exists and would be deleted. "
            "Re-run with --force to start the multiverse over."
        )
    clear_previous_state()

    # path= pins COMET's working directory to results/multiverse/<ds>/comet/.
    mverse = Multiverse(name=MULTIVERSE_NAME, path=str(DIR_MULTIVERSE_COMET))
    mverse.create(analysis_template, dict(MULTIVERSE_GRID))
    mverse.summary()


if __name__ == "__main__":
    main()
