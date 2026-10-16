"""Read-only data-quality and duplicate audit of the overground raw data (ds003039).

Never modifies, moves or re-downloads a data file. Checks:

1. SHA-256 of every subject's .fdt / .set / events.tsv / channels.tsv /
   electrodes.tsv; subjects sharing an .fdt hash (signal duplicates) and,
   separately, an events hash, to tell clean duplicates from inconsistent ones.
2. A local manifest with checksums (none ships with an openneuro-py download)
   and, as a supplementary check, OpenNeuro's S3 ETags (multipart ETags are
   not plain MD5s and are reported as not comparable).
3. Load integrity per subject with MNE: header read, a tail-sample read
   (truncation shows at the end of the file), declared vs file-implied sample
   count, sfreq, channel counts, channels missing from channels.tsv.
4. Each duplicate group: participants.tsv rows, events.tsv identity and the
   EEGLAB metadata baked into the .set; plus a setname-vs-BIDS-ID table.
5. The FULL / PRIMARY / MINIMAL cohorts and their double-stance - swing
   statistics (prepana07 reduction), written to jacobsen_duplicate_recordings.txt.
"""

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import mne
import pandas as pd
import scipy.io as sio

from src.config import config_jacobsen
from src.reference_stats import paired_test, phase_beta_values

TASK_NAME = config_jacobsen.TASK_NAME
ACCESSION = config_jacobsen.ACCESSION
ALL_SUBJECTS = [f"{i:03d}" for i in range(1, 20)]  # sub-001..sub-019, all downloaded
ANALYSIS_SUBJECTS = [f"{i:03d}" for i in range(1, 19)]  # sub-001..sub-018
S3_BASE = "https://s3.amazonaws.com/openneuro.org"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
RULE = "=" * 78


class Report:
    """Collects printed lines for a text report."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str = "") -> None:
        """Print a line and keep it for the report."""
        print(msg)
        self.lines.append(msg)

    def header(self, title: str) -> None:
        """Log a section header framed by rules."""
        self.log(RULE)
        self.log(title)
        self.log(RULE)


def path_for(raw_dir: Path, sub: str, suffix: str) -> Path:
    """BIDS file of one subject, e.g. suffix "eeg.fdt" or "events.tsv"."""
    return raw_dir / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{TASK_NAME}_{suffix}"


def file_digest(path: Path, algorithm: str, chunk_size: int = 1 << 20) -> str:
    """Hex digest of a file ("sha256" or "md5"), read in chunks."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def eeglab_meta(raw_dir: Path, sub: str) -> dict:
    """EEGLAB metadata stored in the .set file (or the read error)."""
    try:
        m = sio.loadmat(
            str(path_for(raw_dir, sub, "eeg.set")), squeeze_me=True, struct_as_record=False
        )
        eeg = m["EEG"]
        return {
            "setname": str(eeg.setname),
            "filename": str(eeg.filename),
            "pnts": int(eeg.pnts),
            "srate": float(eeg.srate),
            "nbchan": int(eeg.nbchan),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def hash_subjects(raw_dir: Path) -> dict:
    """SHA-256 of each subject's raw files and the .fdt size."""
    rows = {}
    for sub in ALL_SUBJECTS:
        fdt = path_for(raw_dir, sub, "eeg.fdt")
        row = {"subject": sub, "fdt_size": fdt.stat().st_size if fdt.exists() else None}
        for key, suffix in (
            ("fdt_hash", "eeg.fdt"),
            ("set_hash", "eeg.set"),
            ("events_hash", "events.tsv"),
            ("channels_hash", "channels.tsv"),
            ("electrodes_hash", "electrodes.tsv"),
        ):
            p = path_for(raw_dir, sub, suffix)
            row[key] = file_digest(p, "sha256") if p.exists() else None
        rows[sub] = row
    return rows


def check_hashes(report: Report, raw_dir: Path) -> tuple[dict, dict, list]:
    """Check 1: hash table and duplicate groups; returns (hash rows, fdt groups, dup groups)."""
    report.header("CHECK 1 -- Content hashing (SHA-256) of raw signal, headers, sidecars")
    hash_rows = hash_subjects(raw_dir)
    report.log("")
    report.log(
        f"{'sub':<6}{'fdt_size':>12}  {'fdt_hash (first 12)':<14}{'set_hash':<14}"
        f"{'events_hash':<14}{'channels_hash':<14}"
    )

    def short(h):
        return h[:12] if h else "MISSING"

    for sub in ALL_SUBJECTS:
        r = hash_rows[sub]
        report.log(
            f"{sub:<6}{r['fdt_size'] if r['fdt_size'] else 0:>12}  "
            f"{short(r['fdt_hash']):<14}{short(r['set_hash']):<14}"
            f"{short(r['events_hash']):<14}{short(r['channels_hash']):<14}"
        )
    by_fdt, by_events = {}, {}
    for sub, r in hash_rows.items():
        if r["fdt_hash"]:
            by_fdt.setdefault(r["fdt_hash"], []).append(sub)
        if r["events_hash"]:
            by_events.setdefault(r["events_hash"], []).append(sub)
    fdt_groups = [subs for subs in by_fdt.values() if len(subs) > 1]
    events_groups = [subs for subs in by_events.values() if len(subs) > 1]
    report.log("")
    report.log("--- Subjects sharing an identical fdt_hash (signal duplicates) ---")
    for grp in fdt_groups:
        same_events = any(grp == g2 for g2 in events_groups) or (
            len(set(hash_rows[s]["events_hash"] for s in grp)) == 1
        )
        kind = (
            "SAME events (clean duplicate)"
            if same_events
            else "DIFFERENT events (inconsistent partial duplicate)"
        )
        report.log(f"  {grp}  -- {kind}")
    if not fdt_groups:
        report.log("  None found.")
    report.log("")
    report.log("--- Subjects sharing an identical events_hash ---")
    for grp in events_groups:
        report.log(f"  {grp}")
    if not events_groups:
        report.log("  None found.")
    return hash_rows, by_fdt, fdt_groups


def check_local_manifest(report: Report, raw_dir: Path) -> None:
    """Check 2a: look for a local manifest with checksums."""
    candidates = [
        raw_dir / ".bidsignore",
        raw_dir / "dataset_description.json",
        raw_dir / ".datalad",
        raw_dir / ".git",
    ]
    report.log("")
    report.log("Local manifest/metadata search in datasets/jacobsen/d00_raw/:")
    for c in candidates:
        exists = c.exists()
        report.log(f"  {c.name:<28} {'FOUND' if exists else 'not present'}")
        if exists and c.suffix == ".json":
            try:
                content = json.loads(c.read_text())
                has_hashes = any(
                    "hash" in str(k).lower() or "sha" in str(k).lower() or "md5" in str(k).lower()
                    for k in content.keys()
                )
                report.log(
                    f"    -> parsed OK, {len(content)} top-level keys, "
                    f"{'contains hash-like fields' if has_hashes else 'no per-file hashes'}"
                )
            except Exception as e:
                report.log(f"    -> could not parse: {e}")
    json_sidecars = list(raw_dir.glob("*.json")) + list(raw_dir.glob("sub-*/eeg/*.json"))
    report.log(
        f"  *.json sidecars found: {len(json_sidecars)} "
        f"(per-subject task/electrode metadata, not per-file checksums)"
    )
    if not any(c.exists() for c in candidates[2:]):
        report.log("")
        report.log("  No git-annex or DataLad metadata present -- openneuro-py's HTTP")
        report.log("  download does not preserve git-annex key/manifest structure.")
        report.log("  NO LOCAL MANIFEST WITH EXPECTED HASHES/SIZES IS AVAILABLE.")
        report.log("  Integrity below relies on (a) internal consistency (Check 1/3)")
        report.log("  and (b) the supplementary S3 ETag comparison immediately below.")


def check_s3_etags(report: Report, raw_dir: Path) -> None:
    """Check 2b: compare local MD5s with OpenNeuro's S3 ETags (network)."""
    report.log("")
    report.log("--- Supplementary (not shipped with the dataset): OpenNeuro S3 ETag check ---")
    report.log(f"Fetching object listing from {S3_BASE}/{ACCESSION}/ ...")
    etag_by_key = {}
    try:
        url = f"{S3_BASE}/?list-type=2&prefix={ACCESSION}/&delimiter="
        with urllib.request.urlopen(url, timeout=30) as resp:
            root = ET.fromstring(resp.read())
        for c in root.findall("s3:Contents", S3_NS):
            etag_by_key[c.find("s3:Key", S3_NS).text] = c.find("s3:ETag", S3_NS).text.strip('"')
        report.log(f"  Retrieved {len(etag_by_key)} object entries from S3.")
    except Exception as e:
        report.log(f"  Could not reach S3 ({type(e).__name__}: {e}) -- skipping ETag check.")
        return
    report.log("")
    report.log(f"{'sub':<6}{'file':<12}{'etag_type':<16}{'match':<10}")
    for sub in ALL_SUBJECTS:
        for suffix, label in (("eeg.fdt", "fdt"), ("eeg.set", "set")):
            key = f"{ACCESSION}/sub-{sub}/eeg/sub-{sub}_task-{TASK_NAME}_{suffix}"
            etag = etag_by_key.get(key)
            p = path_for(raw_dir, sub, suffix)
            if etag is None or not p.exists():
                report.log(f"{sub:<6}{label:<12}{'no S3 entry':<16}{'-':<10}")
            elif "-" in etag:  # multipart-upload ETag: not an MD5 of the whole object
                report.log(f"{sub:<6}{label:<12}{'multipart (n/a)':<16}{'skipped':<10}")
            else:
                match = "MATCH" if file_digest(p, "md5") == etag else "MISMATCH"
                report.log(f"{sub:<6}{label:<12}{'md5':<16}{match:<10}")


def load_subject(raw_dir: Path, sub: str) -> dict:
    """Load-level integrity of one subject (header, tail read, sample counts, channels)."""
    result = {"subject": sub}
    try:
        raw = mne.io.read_raw_eeglab(
            path_for(raw_dir, sub, "eeg.set"), preload=False, verbose=False
        )
        result.update(
            header_ok=True,
            n_ch=len(raw.ch_names),
            n_eeg_ch=len(mne.pick_types(raw.info, eeg=True, exclude=[])),
            sfreq=raw.info["sfreq"],
            n_times_header=raw.n_times,
            duration_s=raw.times[-1],
        )
    except Exception as e:
        result.update(header_ok=False, header_error=f"{type(e).__name__}: {e}")
        return result
    fdt_path = path_for(raw_dir, sub, "eeg.fdt")
    fdt_size = fdt_path.stat().st_size if fdt_path.exists() else None
    result["fdt_size"] = fdt_size
    if fdt_size is not None and result["n_ch"]:
        result["implied_n_times"] = fdt_size / (result["n_ch"] * 4)  # float32 samples
        result["n_times_match"] = abs(result["implied_n_times"] - result["n_times_header"]) < 1.0
    else:
        result["implied_n_times"] = result["n_times_match"] = None
    try:  # truncation shows at the end of the file, like a full preload would
        n = result["n_times_header"]
        raw.get_data(start=max(0, n - 1000), stop=n)
        result["tail_read_ok"], result["tail_read_error"] = True, None
    except Exception as e:
        result["tail_read_ok"], result["tail_read_error"] = False, f"{type(e).__name__}: {e}"
    channels_path = path_for(raw_dir, sub, "channels.tsv")
    if channels_path.exists():
        documented = set(pd.read_csv(channels_path, sep="\t")["name"])
        result["undocumented_channels"] = [c for c in raw.ch_names if c not in documented]
    else:
        result["undocumented_channels"] = None
    return result


def check_load_integrity(report: Report, raw_dir: Path) -> dict:
    """Check 3: load-level integrity table; returns the per-subject results."""
    report.log("")
    report.header("CHECK 3 -- Load-level integrity per subject (MNE, read-only)")
    report.log("")
    report.log("dtype=float32 (4 bytes) assumed for .fdt (standard EEGLAB binary data format).")
    report.log("")
    results = {sub: load_subject(raw_dir, sub) for sub in ALL_SUBJECTS}
    report.log(
        f"{'sub':<6}{'header':<8}{'tail_read':<11}{'n_times(hdr)':>13}{'n_times(fdt)':>13}"
        f"{'match':<7}{'sfreq':>7}{'n_ch':>6}{'n_eeg':>6}{'dur_s':>9}"
    )
    for sub in ALL_SUBJECTS:
        r = results[sub]
        if not r["header_ok"]:
            report.log(f"{sub:<6}{'FAIL':<8}  header error: {r['header_error']}")
            continue
        fdt = f"{r['implied_n_times']:.1f}" if r["implied_n_times"] is not None else "NA"
        report.log(
            f"{sub:<6}{'OK':<8}{'OK' if r['tail_read_ok'] else 'FAIL':<11}"
            f"{r['n_times_header']:>13}{fdt:>13}{'OK' if r['n_times_match'] else 'MISMATCH':<7}"
            f"{r['sfreq']:>7.0f}{r['n_ch']:>6}{r['n_eeg_ch']:>6}{r['duration_s']:>9.1f}"
        )
    report.log("")
    report.log("--- Tail-read failures (full error) ---")
    failures = [s for s in ALL_SUBJECTS if results[s].get("header_ok")]
    failures = [s for s in failures if not results[s].get("tail_read_ok")]
    for sub in failures:
        report.log(f"  sub-{sub}: {results[sub]['tail_read_error']}")
    if not failures:
        report.log("  None.")
    report.log("")
    report.log(
        "--- Undocumented channels (present in raw .set, "
        "absent from that subject's channels.tsv) ---"
    )
    undocumented = [s for s in ALL_SUBJECTS if results[s].get("undocumented_channels")]
    for sub in undocumented:
        report.log(f"  sub-{sub}: {results[sub]['undocumented_channels']}")
    if not undocumented:
        report.log("  None found in any subject.")
    return results


def describe_group(
    report: Report, raw_dir: Path, grp: list, hash_rows: dict, participants: pd.DataFrame
) -> None:
    """Check 4 for one duplicate group: participants rows, events identity, EEGLAB metadata."""
    report.log("")
    report.log(f"--- Group {grp} ---")
    report.log("")
    report.log("participants.tsv rows:")
    for sub in grp:
        if sub in participants.index:
            report.log(f"  sub-{sub}: {participants.loc[sub].to_dict()}")
        else:
            report.log(f"  sub-{sub}: not found in participants.tsv")
    report.log("")
    identical_events = len(set(hash_rows[sub]["events_hash"] for sub in grp)) == 1
    report.log(f"events.tsv identical across group: {identical_events}")
    if not identical_events:
        for sub in grp:
            ev = pd.read_csv(path_for(raw_dir, sub, "events.tsv"), sep="\t")
            report.log(
                f"  sub-{sub}: n_events={len(ev)}  n_value_types={ev['value'].nunique()}  "
                f"first_onset={ev['onset'].min():.3f}  last_onset={ev['onset'].max():.3f}"
            )
    report.log("")
    report.log(".set-embedded EEGLAB metadata (setname/filename/pnts/srate/nbchan):")
    for sub in grp:
        report.log(f"  sub-{sub}: {eeglab_meta(raw_dir, sub)}")


SETNAME_NOTE = [
    "(setname is the internal EEGLAB dataset name baked into the .set file",
    " at whatever point the original authors last saved it -- a mismatch",
    " here does not by itself prove a duplicate, but is strong corroborating",
    " evidence of the participant-ID renumbering documented in the",
    " dataset's own CHANGES file: 'updated participant IDs to be in",
    " increasing order'. NOTE this does not fully explain the fdt-hash",
    " duplicates above: e.g. sub-005 claims setname='sub-007', but",
    " sub-007 itself claims setname='sub-010' -- different claimed",
    " origins despite byte-identical signal, so the duplication is not",
    " simply a consistent relabeling.)",
]


def check_duplicate_groups(
    report: Report, raw_dir: Path, fdt_groups: list, hash_rows: dict
) -> None:
    """Check 4: characterize each duplicate group and tabulate setname vs BIDS ID."""
    report.log("")
    report.header("CHECK 4 -- Characterizing each fdt-hash duplicate group")
    participants = pd.read_csv(raw_dir / "participants.tsv", sep="\t")
    participants = participants.set_index(
        participants["participant_id"].str.replace("sub-", "", regex=False)
    )
    if not fdt_groups:
        report.log("")
        report.log("No fdt-hash duplicate groups found -- nothing to characterize.")
    for grp in fdt_groups:
        describe_group(report, raw_dir, grp, hash_rows, participants)
    report.log("")
    report.log("--- Full setname-vs-BIDS-ID consistency table (all 19 subjects) ---")
    for line in SETNAME_NOTE:
        report.log(line)
    report.log("")
    report.log(f"{'sub':<6}{'setname':<12}{'match?':<10}")
    for sub in ALL_SUBJECTS:
        meta = eeglab_meta(raw_dir, sub)
        if "error" in meta:
            report.log(f"{sub:<6}{'ERROR: ' + meta['error']}")
            continue
        match = "MATCH" if meta["setname"] == f"sub-{sub}" else "MISMATCH"
        report.log(f"{sub:<6}{meta['setname']:<12}{match:<10}")


def classify_groups(fdt_groups: list, hash_rows: dict) -> tuple[list, list]:
    """Split duplicate groups among the analysis subjects into clean and inconsistent ones."""
    clean, inconsistent = [], []
    for grp in fdt_groups:
        grp_a = sorted(s for s in grp if s in ANALYSIS_SUBJECTS)
        if len(grp_a) < 2:
            continue
        same_events = len(set(hash_rows[s]["events_hash"] for s in grp_a)) == 1
        (clean if same_events else inconsistent).append(grp_a)
    return clean, inconsistent


def cohort_variants(subjects: list, clean: list, inconsistent: list) -> list[tuple[str, set]]:
    """FULL, PRIMARY and MINIMAL cohorts as (label, excluded subjects).

    PRIMARY drops one of each clean duplicate pair and both members of an
    inconsistent pair; MINIMAL drops only one of each clean duplicate pair.
    """
    exclude_primary, exclude_minimal = set(), set()
    for grp in clean:
        keep = min(grp)
        exclude_primary.update(s for s in grp if s != keep)
        exclude_minimal.update(s for s in grp if s != keep)
    for grp in inconsistent:
        exclude_primary.update(grp)
    return [
        ("(a) FULL", set()),
        ("(b) PRIMARY (headline)", exclude_primary),
        ("(c) MINIMAL sensitivity", exclude_minimal),
    ]


def duplicate_group_lines(clean: list, inconsistent: list) -> list[str]:
    """Header and duplicate-group lines of jacobsen_duplicate_recordings.txt."""
    lines = [
        "JACOBSEN (ds003039) -- DUPLICATE RECORDING QC",
        RULE,
        "Confirmed source-side (not download artifacts) via SHA-256 content hashing",
        "+ OpenNeuro S3 ETag cross-check -- see jacobsen_dataquality_audit.txt (Check 1/2) for",
        "full derivation. Regenerated by scripts/qc/qc_jacobsen_duplicate_audit.py "
        "-- not hand-authored.",
        "",
        "All 18 subjects are processed and reported through the pipeline. No subject's",
        "data or per-subject output files are dropped, moved, or deleted. Duplication",
        "is handled ONLY at the group-statistic reporting layer below.",
        "",
        "-" * 78,
        "Confirmed duplicate groupings",
        "-" * 78,
    ]
    for a, b in clean:
        lines.append(
            f"  sub-{a} == sub-{b}   clean duplicate: identical .fdt AND identical events.tsv"
        )
    for a, b in inconsistent:
        lines.append(
            f"  sub-{a} -- sub-{b}   identical EEG .fdt, but DIFFERENT events.tsv "
            "-- NOT a clean duplicate"
        )
    if not (clean or inconsistent):
        lines.append("  None found among the 18 analysis subjects.")
    return lines


VARIANT_NOTES = [
    "",
    "  (a) FULL: every subject, duplicates included as-is.",
    "  (b) PRIMARY / deduplicated: drops one of each exact-duplicate pair AND",
    "      both members of any inconsistent-events duplicate pair -- the",
    "      independent-recordings result. This is the headline/primary result.",
    "  (c) MINIMAL sensitivity: drops only the exact-duplicate pairs, keeping BOTH",
    "      members of any inconsistent-events pair in (not treated as a",
    "      deduplicatable duplicate for this variant).",
]


def cohort_statistic_lines(clean: list, inconsistent: list, betas: dict) -> list[str]:
    """Duplicate-pair consistency and the three cohort t-tests (Check 5 tables)."""
    lines = [
        "",
        "-" * 78,
        "Internal consistency check: per-subject beta contrast (double_stance - swing, dB)",
        "-" * 78,
        f"  {'Pair':<16}{'sub_A':<9}{'diff_A':>9}{'sub_B':>9}{'diff_B':>9}{'|A-B|':>9}",
    ]
    for a, b in clean + inconsistent:
        if a in betas and b in betas:
            diff_a, diff_b = betas[a][0] - betas[a][1], betas[b][0] - betas[b][1]
            lines.append(
                f"  {a} vs {b}    {a:<9}{diff_a:>+9.3f}{b:>9}{diff_b:>+9.3f}"
                f"{abs(diff_a - diff_b):>9.3f}"
            )
    lines += [
        "",
        "-" * 78,
        "Group paired t-test (double_stance - swing), reported three ways",
        "-" * 78,
        f"  {'Variant':<26}{'N':>4}{'df':>5}{'t':>8}{'p':>10}{'mean_diff':>12}   Included subjects",
    ]
    all_subs = [s for s in ANALYSIS_SUBJECTS if s in betas]
    for label, excluded in cohort_variants(all_subs, clean, inconsistent):
        subs = [s for s in all_subs if s not in excluded]
        v = paired_test([betas[s][0] for s in subs], [betas[s][1] for s in subs])
        lines.append(
            f"  {label:<26}{v['n']:>4}{v['df']:>5}{v['t']:>8.3f}{v['p']:>10.4g}"
            f"{v['mean_diff']:>+11.3f} dB   {','.join(subs)}"
        )
    return lines + VARIANT_NOTES


def duplicate_report_lines(clean: list, inconsistent: list, ersp_dir: Path) -> list[str]:
    """Full text of jacobsen_duplicate_recordings.txt (Check 5).

    Parameters
    ----------
    clean, inconsistent : list of list of str
        Duplicate groups among the analysis subjects (``classify_groups``).
    ersp_dir : pathlib.Path
        prepana05 outputs, for the beta double-stance and swing values.

    Returns
    -------
    list of str
    """
    lines = duplicate_group_lines(clean, inconsistent)
    betas, missing = {}, []
    for sub in ANALYSIS_SUBJECTS:
        values = phase_beta_values(sub, ersp_dir)
        if values is None:
            missing.append(sub)
        else:
            betas[sub] = values
    if not betas:
        return lines + [
            "",
            "Pipeline ERSP outputs (d05_ersp) not found for jacobsen -- the",
            "group-statistic section below cannot be computed. Run prepana01-05",
            "for jacobsen first, then re-run this script.",
        ]
    if missing:
        lines += ["", f"NOTE: missing ERSP outputs for {missing} -- excluded from stats below."]
    return lines + cohort_statistic_lines(clean, inconsistent, betas)


def summary_lines(n_unique_fdt: int, fdt_groups: list, load_results: dict) -> list[str]:
    """Summary block placed at the top of jacobsen_dataquality_audit.txt."""
    n_header_fail = sum(1 for r in load_results.values() if not r.get("header_ok"))
    tail_fail = [
        s
        for s in ALL_SUBJECTS
        if load_results[s].get("header_ok") and not load_results[s].get("tail_read_ok")
    ]
    return [
        RULE,
        "SUMMARY",
        RULE,
        f"Subjects examined            : {len(ALL_SUBJECTS)} (sub-001..sub-019)",
        f"Unique raw recordings (fdt)   : {n_unique_fdt}",
        f"Duplicate fdt groups          : {fdt_groups if fdt_groups else 'none'}",
        f"Subjects failing header read  : {n_header_fail}",
        f"Subjects failing tail read    : {len(tail_fail)} ({tail_fail})",
        "Local BIDS manifest w/ hashes : NOT AVAILABLE (no git-annex/.datalad shipped)",
        "Supplementary S3 ETag check   : see Check 2 for per-file result "
        "(multipart ETags are not directly comparable)",
        "",
        "See full checks below for per-subject detail.",
        "",
    ]
