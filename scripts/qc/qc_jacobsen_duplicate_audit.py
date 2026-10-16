"""
qc_jacobsen_duplicate_audit.py -- overground raw-data audit: hashes, duplicates, integrity, cohorts.

Purpose
    Read-only duplicate and integrity audit of the overground BIDS data
    (all 19 downloaded subjects; src.duplicate_audit, Checks 1-5) and the
    source of the overground cohorts (FULL n = 18, PRIMARY n = 14,
    MINIMAL n = 16) with their double-stance - swing statistics (prepana07
    reduction). Every other script reads the cohorts from
    jacobsen_duplicate_recordings.txt (src.qc.load_cohort_variants).
Inputs
    datasets/jacobsen/d00_raw/ (all files hashed), datasets/jacobsen/d05_ersp/ (Check 5);
    network: OpenNeuro S3 listing (Check 2, supplementary)
Outputs
    results/pipeline/jacobsen/qc/jacobsen_dataquality_audit.txt (Checks 1-4)
    results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt (Check 5; cohorts)
Manuscript
    Methods 2.1 (duplicate recordings, primary cohort); Results, overground statistic.
Run
    python scripts/qc/qc_jacobsen_duplicate_audit.py   (always overground; ~1 min)
"""

import os

os.environ["BETAGAIT_DATASET"] = "jacobsen"

import warnings  # noqa: E402

from src.config import DIR_QC  # noqa: E402
from src.duplicate_audit import (  # noqa: E402
    Report,
    check_duplicate_groups,
    check_hashes,
    check_load_integrity,
    check_local_manifest,
    check_s3_etags,
    classify_groups,
    duplicate_report_lines,
    summary_lines,
)
from src.paths import get_dataset_dirs  # noqa: E402
from src.qc import DUPLICATE_REPORT  # noqa: E402

AUDIT_REPORT = "jacobsen_dataquality_audit.txt"


def main() -> None:
    """Run Checks 1-5 and write both reports."""
    warnings.filterwarnings("ignore")
    dirs = get_dataset_dirs("jacobsen")
    raw_dir = dirs["raw"]
    report = Report()
    hash_rows, by_fdt, fdt_groups = check_hashes(report, raw_dir)
    report.log("")
    report.header("CHECK 2 -- Cross-check against a shipped manifest / checksum source")
    check_local_manifest(report, raw_dir)
    check_s3_etags(report, raw_dir)
    load_results = check_load_integrity(report, raw_dir)
    check_duplicate_groups(report, raw_dir, fdt_groups, hash_rows)

    report.log("")
    report.header(f"CHECK 5 -- Duplicate-recording group-statistic report (see {DUPLICATE_REPORT})")
    clean, inconsistent = classify_groups(fdt_groups, hash_rows)
    duplicates_path = DIR_QC / DUPLICATE_REPORT
    lines = duplicate_report_lines(clean, inconsistent, dirs["ersp"])
    duplicates_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDuplicate-recording report written -> {duplicates_path}")

    audit_path = DIR_QC / AUDIT_REPORT
    summary = summary_lines(len(by_fdt), fdt_groups, load_results)
    audit_path.write_text("\n".join(summary) + "\n" + "\n".join(report.lines) + "\n", "utf-8")
    print(f"\nReport written -> {audit_path}")


if __name__ == "__main__":
    main()
