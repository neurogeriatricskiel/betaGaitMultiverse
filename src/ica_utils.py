"""ICA utilities: fitting, ICLabel classification, and QC plotting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_icalabel.iclabel import iclabel_label_components

from src.config import (
    ICA_DECIM,
    ICA_METHOD,
    ICLABEL_P_ARTIFACT_LIBERAL,
    ICLABEL_P_BRAIN_BALANCED,
    ICLABEL_P_BRAIN_CONSERVATIVE,
    N_COMPONENTS,
    RANDOM_STATE,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Column order of the raw ICLabel probability matrix
# (mne_icalabel.label_components.ICALABEL_METHODS_NUMERICAL_TO_STRING["iclabel"]).
ICLABEL_CLASSES = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]


def run_ica(
    epochs: mne.Epochs,
    n_components: float | int = N_COMPONENTS,
    method: str = ICA_METHOD,
    *,
    fit_params: dict | None,
    decim: int = ICA_DECIM,
    random_state: int = RANDOM_STATE,
) -> mne.preprocessing.ICA:
    """Fit ICA on clean epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        AutoReject-repaired epochs to fit on.
    n_components : float | int
        Number of components, or fraction of variance to retain.
    method : str
        ICA method.
    fit_params : dict | None
        Required keyword: ``src.config.ICA_FIT_PARAMS`` (Extended Infomax).
    decim : int
        Decimation factor used for fitting.
    random_state : int
        Random seed.

    Returns
    -------
    mne.preprocessing.ICA
        The fitted ICA.
    """
    rank = mne.compute_rank(epochs, rank="info")
    print(f"  Data rank        : {rank}")

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(epochs, decim=decim)
    print(f"  ICA components   : {ica.n_components_}")
    return ica


def label_and_mark_ica(
    ica: mne.preprocessing.ICA,
    epochs: mne.Epochs,
    rule: str = "balanced",
    artifact_rule_fn: Callable[[np.ndarray], list[int]] | None = None,
) -> dict:
    """
    Run ICLabel on epochs and set ``ica.exclude`` in place.

    Calls the exact same select_ics_by_rule() the multiverse pipeline uses
    (src.multiverse_pipeline.run_subject_multiverse) so the reference
    pipeline's component-selection logic can never drift from the
    multiverse's again -- one function, one place the rule is defined.
    Previously this used its own compound condition (argmax label=="brain"
    AND max_prob>=0.7), which is NOT equivalent to select_ics_by_rule's
    "balanced" (P(brain)>0.7 alone) -- identified as one of two causes
    (the other: lowpass_hz) of the reference pipeline failing to reproduce
    a matching multiverse universe's effect size even at matched
    highpass/asr_mode/iclabel_rule settings.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA; its ``exclude`` is overwritten.
    epochs : mne.Epochs
        Epochs used to fit ``ica``.
    rule : str
        Rule name passed to select_ics_by_rule().
    artifact_rule_fn : callable(probs) -> list[int], optional
        If given, used INSTEAD of select_ics_by_rule(probs, rule) -- e.g.
        select_ics_jacobsen_paper_rule below, kept deliberately decoupled
        from this shared rule/multiverse vocabulary. `rule` is used only
        for the printed label in that case.

    Returns
    -------
    dict
        Keys ``labels``, ``probs``, ``brain_ics``, ``exclude_ics``.
    """
    probs, labels = iclabel_probabilities(ica, epochs)
    if artifact_rule_fn is not None:
        exclude_ics = artifact_rule_fn(probs)
    else:
        exclude_ics = select_ics_by_rule(probs, rule)
    brain_ics = [i for i in range(len(labels)) if i not in exclude_ics]

    print(f"\n  ICLabel classification (rule={rule}):")
    for i, label in enumerate(labels):
        marker = "+" if i in brain_ics else "-"
        print(f"    {marker}  IC{i:03d}  {label:<20}  p_brain={probs[i, 0]:.2f}")

    ica.exclude = exclude_ics
    print(f"\n  Keeping  : {len(brain_ics)} brain ICs  {brain_ics}")
    print(f"  Excluding: {len(exclude_ics)} artefact ICs")

    return {
        "labels": labels,
        "probs": probs,
        "brain_ics": brain_ics,
        "exclude_ics": exclude_ics,
    }


def iclabel_probabilities(
    ica: mne.preprocessing.ICA,
    epochs: mne.Epochs,
) -> tuple[np.ndarray, list[str]]:
    """
    Run ICLabel and return the full per-class probability matrix.

    Unlike label_and_mark_ica() / mne_icalabel.label_components(), this
    keeps every class's probability (not just the argmax), so a caller
    can apply more than one exclusion rule to the same ICLabel run
    without recomputing it (see select_ics_by_rule()).

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA.
    epochs : mne.Epochs
        Epochs used to fit `ica` (ICLabel needs these for features).

    Returns
    -------
    probs  : ndarray, shape (n_components, 7)
        Columns ordered per ICLABEL_CLASSES: brain, muscle artifact,
        eye blink, heart beat, line noise, channel noise, other.
    labels : list of str, length n_components
        Argmax class label per component (for logging only).
    """
    probs = iclabel_label_components(epochs, ica)
    labels = [ICLABEL_CLASSES[i] for i in probs.argmax(axis=1)]
    return probs, labels


def select_ics_by_rule(probs: np.ndarray, rule: str) -> list[int]:
    """
    Select ICA components to exclude from an existing ICLabel probability matrix.

    No refitting of ICA or ICLabel is needed.

    Parameters
    ----------
    probs : ndarray, shape (n_components, 7)
        From iclabel_probabilities(), columns ordered per ICLABEL_CLASSES.
    rule  : {"conservative", "balanced", "liberal"}
        conservative -> keep only ICs with P(brain) > ICLABEL_P_BRAIN_CONSERVATIVE (0.9)
        balanced     -> keep only ICs with P(brain) > ICLABEL_P_BRAIN_BALANCED (0.7)
        liberal      -> reject only ICs with P(muscle) or P(eye) > ICLABEL_P_ARTIFACT_LIBERAL (0.9)

    Returns
    -------
    exclude_ics : list of int
        Component indices to exclude (assign to ica.exclude).
    """
    p_brain, p_muscle, p_eye = probs[:, 0], probs[:, 1], probs[:, 2]
    if rule == "conservative":
        keep = p_brain > ICLABEL_P_BRAIN_CONSERVATIVE
    elif rule == "balanced":
        keep = p_brain > ICLABEL_P_BRAIN_BALANCED
    elif rule == "liberal":
        keep = ~((p_muscle > ICLABEL_P_ARTIFACT_LIBERAL) | (p_eye > ICLABEL_P_ARTIFACT_LIBERAL))
    else:
        raise ValueError(f"Unknown iclabel_rule: {rule!r}")
    return [i for i, k in enumerate(keep) if not k]


def select_ics_jacobsen_paper_rule(probs: np.ndarray) -> list[int]:
    """
    Select ICs to exclude with Jacobsen et al.'s paper-exact ICLabel rule.

    Exclude a component if P(eye) > 0.9 OR P(muscle) > 0.9.

    Deliberately a standalone function, NOT a branch of
    select_ics_by_rule() above and NOT wired through iclabel_rule/
    ICLABEL_RULE -- this is the Jacobsen reference pipeline's own paper-
    fidelity rule, kept decoupled from the multiverse's shared
    "conservative"/"balanced"/"liberal" vocabulary (whose "liberal"
    branch happens to compute the same threshold today, but the two are
    intentionally not the same code path, so a future change to one
    rule's meaning can never silently change the other's).

    Parameters
    ----------
    probs : ndarray, shape (n_components, 7)
        From iclabel_probabilities(), columns ordered per ICLABEL_CLASSES.

    Returns
    -------
    exclude_ics : list of int
        Component indices to exclude.
    """
    p_muscle, p_eye = probs[:, 1], probs[:, 2]
    keep = ~((p_muscle > ICLABEL_P_ARTIFACT_LIBERAL) | (p_eye > ICLABEL_P_ARTIFACT_LIBERAL))
    return [i for i, k in enumerate(keep) if not k]


def save_ica_component_plots(
    ica: mne.preprocessing.ICA,
    output_dir: Path,
    subject: str,
) -> None:
    """Save topography plots for all ICA components to the output directory.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA.
    output_dir : Path
        Directory for the PNG files.
    subject : str
        Subject ID used in the file names.
    """
    figs = ica.plot_components(show=False)
    if not isinstance(figs, list):
        figs = [figs]

    for k, fig in enumerate(figs):
        out = output_dir / f"sub-{subject}_ica_topos_{k:02d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ICA topos        -> {out.name}")
