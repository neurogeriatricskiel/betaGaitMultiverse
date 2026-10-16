"""
Per-subject, per-stage resumability for the prepana0X scripts.

A stage's output for one subject counts as already done only if every
expected output file exists, is non-empty, is loadable (via a
caller-supplied check), AND is not older than any of the stage's own
input files. That last condition is the one that matters after an
interrupted run: it catches a subject left with a fresh early-stage
artifact sitting next to a stale later-stage one (e.g. a freshly
re-saved concat_raw.fif next to an ica.fif left over from a prior,
different run) -- a plain "does the output file exist" check would
wrongly treat that subject as complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def stage_already_done(
    outputs: Iterable[str | Path],
    inputs: Iterable[str | Path] = (),
    validate: Callable[[], object] | None = None,
) -> bool:
    """Check whether a stage's outputs exist, are fresh, and (optionally) load.

    Fresh means no input file is newer than the oldest output file.

    Parameters
    ----------
    outputs : list[Path]
        Every file this stage must produce for one subject. All must
        exist and be non-empty.
    inputs : list[Path], optional
        Files this stage reads to produce `outputs`. If any exists and
        is newer than the oldest file in `outputs`, the outputs are
        considered stale (produced by an earlier, now-superseded input).
    validate : callable, optional
        Zero-arg callable that raises if the outputs aren't actually
        loadable (e.g. ``lambda: mne.preprocessing.read_ica(ica_path)``).
        Only called once the existence/freshness checks already pass,
        so a genuinely-missing output never pays for a load attempt.

    Returns
    -------
    bool
        True if this stage can be skipped for this subject.
    """
    outputs = [Path(p) for p in outputs]
    if not outputs or any(not p.exists() or p.stat().st_size == 0 for p in outputs):
        return False

    oldest_output = min(p.stat().st_mtime for p in outputs)
    for inp in inputs:
        inp = Path(inp)
        if inp.exists() and inp.stat().st_mtime > oldest_output:
            return False

    if validate is not None:
        try:
            validate()
        except Exception:
            return False

    return True
