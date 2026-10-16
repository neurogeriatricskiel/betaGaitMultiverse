"""Lowest / median / highest multiverse universe figure (mulana04).

Universes are ranked by the group-mean double-stance - swing beta contrast
(src.multiverse_results.universe_table). The group ROI ERSP (8-40 Hz) of
the three selected universes is computed from each universe's cached
ICA-corrected data (src.multiverse_pipeline.cached_universe_roi_ersp) and
cached to results/multiverse/<ds>/outputs/<ds>_zoom_arrays.npz.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage

from src import plot_style as ps
from src.config import MULTIVERSE_LOWPASS_HZ, TFR_N_POINTS
from src.ersp import BETA_FMAX, BETA_FMIN, phase_split_indices
from src.multiverse_pipeline import FREQS, cached_universe_roi_ersp
from src.multiverse_results import universe_table

ORDER = ["LOWEST", "MEDIAN", "HIGHEST"]
BETA_MASK = (FREQS >= BETA_FMIN) & (FREQS <= BETA_FMAX)


def rank_universes(all_universes: dict, measure: str) -> pd.DataFrame:
    """Universes sorted by group-mean contrast, with t, p and n.

    Parameters
    ----------
    all_universes : dict
        Combined multiverse results (pkl contents).
    measure : str
        Per-subject contrast field ("t_stats" or "dedup_diffs").

    Returns
    -------
    pandas.DataFrame
        universe, decisions, effect_db, t_stat, p_val, n_subjects; ascending effect.
    """
    table = universe_table(all_universes, measure).rename(
        columns={"mean": "effect_db", "t": "t_stat", "p": "p_val", "n": "n_subjects"}
    )
    if measure == "t_stats":  # group_t_mean is the paired t of the same subjects
        for _, row in table.iterrows():
            saved_t = all_universes[f"universe_{int(row['universe'])}"]["group_t_mean"]
            if np.isfinite(saved_t) and not np.isclose(row["t_stat"], saved_t, atol=1e-6):
                print(f"  [WARN] universe_{int(row['universe'])}: t differs from group_t_mean")
    return table


def select_universes(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the lowest, median (middle of an odd count) and highest universe."""
    median_idx = (len(table) - 1) // 2
    return {"LOWEST": table.iloc[0], "MEDIAN": table.iloc[median_idx], "HIGHEST": table.iloc[-1]}


def describe(row: pd.Series) -> str:
    """One-line description of a universe row."""
    return (
        f"universe_{int(row['universe'])}: HP={row['highpass_hz']} Hz, ASR={row['asr_mode']}, "
        f"IC={row['iclabel_rule']}  |  {row['effect_db']:+.3f} dB  t={row['t_stat']:.3f}  "
        f"p={row['p_val']:.4g}"
    )


def print_selection(table: pd.DataFrame, selection: dict) -> None:
    """Print the selected universes and what a ranking by t would pick instead."""
    print(f"\n  {len(table)} universes; median = sorted position {(len(table) - 1) // 2 + 1}.")
    for label, row in selection.items():
        print(f"    {label:8s} {describe(row)}  (n={int(row['n_subjects'])} subjects)")
    by_t = select_universes(table.sort_values("t_stat").reset_index(drop=True))
    print("\n  Ranking by t instead would pick:")
    for label, row in by_t.items():
        same = "same" if row["universe"] == selection[label]["universe"] else "DIFFERS"
        print(f"    {label:8s} {describe(row)}  ({same})")


def load_cached_panels(cache_path: Path, selection: dict) -> tuple[dict, dict] | None:
    """Panel arrays from the cache if it matches the selection, else None."""
    if not cache_path.exists():
        return None
    cache = np.load(cache_path)
    if (
        all(int(cache[f"{lb}_universe"]) == int(selection[lb]["universe"]) for lb in ORDER)
        and np.array_equal(cache["freqs"], FREQS)
        and all(int(cache[f"{lb}_n"]) == int(selection[lb]["n_subjects"]) for lb in ORDER)
    ):
        print(f"\n  Panel arrays loaded from cache {cache_path}")
        return {lb: cache[f"{lb}_ersp"] for lb in ORDER}, {
            lb: int(cache[f"{lb}_n"]) for lb in ORDER
        }
    print(f"\n  Cache {cache_path} does not match the current selection -- recomputing")
    return None


def group_universe_ersp(
    row: pd.Series, subjects: list[str], gait_events_dir: Path, anchors: tuple[float, float, float]
) -> list:
    """Per-subject ROI ERSP of one universe (subjects without cached data are skipped)."""
    decisions = {
        "highpass_hz": float(row["highpass_hz"]),
        "asr_mode": row["asr_mode"],
        "iclabel_rule": row["iclabel_rule"],
        "lowpass_hz": MULTIVERSE_LOWPASS_HZ,
    }
    per_subject = []
    for subject in subjects:
        try:
            per_subject.append(
                cached_universe_roi_ersp(subject, decisions, gait_events_dir, anchors)
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"    [SKIP] sub-{subject}: {e}")
    return per_subject


def panel_arrays(
    selection: dict,
    subjects: list[str],
    gait_events_dir: Path,
    anchors: tuple[float, float, float],
    cache_path: Path,
) -> tuple[dict, dict]:
    """Group ROI ERSP and n per selected universe (from the cache when it matches).

    Parameters
    ----------
    selection : dict
        Output of ``select_universes``.
    subjects : list of str
        Cohort averaged.
    gait_events_dir : pathlib.Path
        Folder of the cycle tables.
    anchors : tuple of float
        Group-median warp anchors.
    cache_path : pathlib.Path
        .npz cache of the panel arrays.

    Returns
    -------
    group_ersp : dict
        Label -> (n_freqs, TFR_N_POINTS) array.
    n_used : dict
        Label -> subjects averaged.
    """
    cached = load_cached_panels(cache_path, selection)
    if cached is not None:
        return cached
    group_ersp, n_used = {}, {}
    for label in ORDER:
        print(
            f"\n  Computing group ERSP for {label} (universe_{int(selection[label]['universe'])})"
        )
        per_subject = group_universe_ersp(selection[label], subjects, gait_events_dir, anchors)
        if len(per_subject) < 2:
            raise SystemExit(f"{label}: only {len(per_subject)} usable subjects -- cannot average.")
        group_ersp[label] = np.mean(np.stack(per_subject), axis=0)
        n_used[label] = len(per_subject)
    np.savez(
        cache_path,
        freqs=FREQS,
        anchors=np.array(anchors),
        **{f"{lb}_ersp": group_ersp[lb] for lb in ORDER},
        **{f"{lb}_universe": int(selection[lb]["universe"]) for lb in ORDER},
        **{f"{lb}_n": n_used[lb] for lb in ORDER},
    )
    print(f"\n  Panel arrays cached -> {cache_path}")
    return group_ersp, n_used


def check_contrasts(
    selection: dict, group_ersp: dict, n_used: dict, anchors: tuple[float, float, float]
) -> None:
    """Stop unless each shown array's double-stance - swing contrast matches the pkl within 1 %."""
    double_stance_idx, swing_idx = phase_split_indices(anchors, n_points=TFR_N_POINTS)
    for label in ORDER:
        row = selection[label]
        beta_rows = group_ersp[label][BETA_MASK]
        shown_db = float(beta_rows[:, double_stance_idx].mean() - beta_rows[:, swing_idx].mean())
        rel = abs(shown_db - row["effect_db"]) / abs(row["effect_db"])
        print(
            f"  {label:8s} DS - swing from shown array: {shown_db:+.4f} dB  vs pkl "
            f"{row['effect_db']:+.4f} dB  (rel. diff {rel:.2%})"
        )
        if n_used[label] != row["n_subjects"] or not np.isclose(
            shown_db, row["effect_db"], rtol=0.01, atol=0
        ):
            raise SystemExit(f"{label}: shown array does not reproduce the pkl contrast.")


def draw_columns(
    fig: Figure,
    gs_outer: GridSpec,
    selection: dict,
    group_ersp: dict,
    anchors: tuple[float, float, float],
    vlim: float,
    dataset: str,
) -> AxesImage:
    """Draw heatmap, beta trace, events and phase bar for each selected universe."""
    im, ax_first_trace = None, None
    for col, label in enumerate(ORDER):
        row = selection[label]
        ax_heat, ax_beta, ax_events, ax_phases = ps.gait_column(fig, gs_outer[col])
        if ax_first_trace is None:
            ax_first_trace = ax_beta
        else:
            ax_beta.sharey(ax_first_trace)
        im = ps.draw_heatmap(
            ax_heat,
            group_ersp[label],
            FREQS,
            vlim,
            anchors,
            beta_band=(BETA_FMIN, BETA_FMAX),
            ylabel=(col == 0),
        )
        trace = group_ersp[label][BETA_MASK].mean(axis=0)
        ylabel = "Beta 13-30 Hz\n(dB)" if col == 0 else None
        ps.draw_beta_trace(ax_beta, trace, ps.TRACE_YLIM[dataset], anchors, ylabel=ylabel)
        ps.draw_event_row(ax_events, anchors)
        ps.draw_phase_bar(ax_phases, anchors)
        ax_heat.set_title(
            f"{label.capitalize()}: {ps.universe_label(row)}\n"
            f"{row['effect_db']:+.2f} dB, t({int(row['n_subjects']) - 1}) = {row['t_stat']:.2f}, "
            f"{ps.format_p(row['p_val'])}",
            fontsize=ps.FS_LABEL,
        )
    return im


def render_zoom_figure(
    selection: dict,
    group_ersp: dict,
    anchors: tuple[float, float, float],
    dataset: str,
    out_path: Path,
) -> None:
    """Save the three-column zoom figure with one shared colour range and colourbar."""
    vlim = ps.ERSP_VLIM[(dataset, "zoom")]
    for label in ORDER:
        arr = group_ersp[label]
        trace = arr[BETA_MASK].mean(axis=0)
        print(
            f"    {label:8s} min={arr.min():+.3f}  max={arr.max():+.3f}  clipped at "
            f"+/-{vlim:g}: {100 * ps.clip_fraction([arr], vlim):.2f}%  "
            f"beta trace {trace.min():+.3f} .. {trace.max():+.3f} dB"
        )
    ps.report_clipping("zoom (3 panels)", list(group_ersp.values()), vlim)
    print(f"  Beta-trace y-limits (fixed): {ps.TRACE_YLIM[dataset]}")

    ps.apply_rc()
    fig = plt.figure(figsize=ps.FIGSIZE["zoom"])
    gs_outer = gridspec.GridSpec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.25,
        top=0.86,
        bottom=0.06,
        left=0.06,
        right=0.95,
        figure=fig,
    )
    n_fig = int(selection["MEDIAN"]["n_subjects"])
    ps.add_header(
        fig,
        ps.figure_title(dataset, "lowest, median and highest multiverse specifications", n_fig),
        ps.figure_subtitle(FREQS[0], FREQS[-1])
        + "; ranked by double-stance - swing beta contrast (dB)",
        ps.scale_note(vlim),
    )
    im = draw_columns(fig, gs_outer, selection, group_ersp, anchors, vlim, dataset)
    ps.add_ersp_colorbar(fig, im, None, vlim, cax=fig.add_subplot(gs_outer[3]))
    ps.save(fig, out_path)
