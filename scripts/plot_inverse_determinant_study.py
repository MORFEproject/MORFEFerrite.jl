#!/usr/bin/env python3
"""Plot the CSVs written by ``inverse_determinant_study.jl``.

Reads
    scripts/results/inverse_determinant_study/data/{accuracy,cost,radius}.csv

Writes
    scripts/results/inverse_determinant_study/figures/accuracy.png
    scripts/results/inverse_determinant_study/figures/cost.png
    scripts/results/inverse_determinant_study/figures/radius.png

The numerical CSV values are never clipped. A small floor is applied only to
values displayed on logarithmic axes so exact agreement at theta=0 remains
visible without changing the underlying data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results" / "inverse_determinant_study"
DATA_DIR = RESULT_ROOT / "data"
FIGURE_DIR = RESULT_ROOT / "figures"

# Keep matplotlib's cache beside the other ignored generated artifacts. This
# avoids warnings in sandboxed or read-only home directories.
MPL_CACHE = RESULT_ROOT / ".matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MESHES = ("small", "medium", "large")
MESH_COLORS = {
    "small": "#009E73",   # green
    "medium": "#0072B2",  # blue
    "large": "#D55E00",   # vermilion/red
}
VH_STYLES = {1: "--", 2: "-"}
PLOT_FLOOR = 1e-16


def load_csv(name: str, expected_rows: int) -> np.ndarray:
    path = DATA_DIR / name
    if not path.is_file():
        sys.exit(f"missing {path} — run inverse_determinant_study.jl first")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data = np.atleast_1d(data)
    if data.size != expected_rows:
        sys.exit(f"{path} has {data.size} rows; expected {expected_rows}")
    return data


def shown(values: np.ndarray) -> np.ndarray:
    """Apply the logarithmic display floor without modifying CSV data."""
    return np.maximum(np.asarray(values, dtype=float), PLOT_FLOOR)


def ordered_rows(data: np.ndarray, mask: np.ndarray, column: str) -> np.ndarray:
    rows = data[mask]
    return rows[np.argsort(rows[column])]


def style_error_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.22)


def plot_accuracy(data: np.ndarray) -> Path:
    geometries = set(np.unique(data["geometry"]))
    if geometries != {"stretch", "wavy"}:
        sys.exit(f"accuracy.csv has unexpected geometries: {sorted(geometries)}")
    if set(np.unique(data["mesh"])) != set(MESHES):
        sys.exit("accuracy.csv does not contain the three expected meshes")

    wavy = data[data["geometry"] == "wavy"]
    stretch = data[data["geometry"] == "stretch"]
    stretch_gap = float(np.max(stretch["max_abs"]))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    ax_max, ax_rms, ax_residual, ax_dof = axes.ravel()

    for mesh in MESHES:
        color = MESH_COLORS[mesh]
        for vh in (1, 2):
            mask = (wavy["mesh"] == mesh) & (wavy["vh_order"] == vh)
            rows = ordered_rows(wavy, mask, "theta")
            label = f"{mesh}, $V_h$ order {vh}"
            common = dict(color=color, ls=VH_STYLES[vh], lw=1.9, label=label)
            ax_max.plot(rows["theta"], shown(rows["max_abs"]), **common)
            ax_rms.plot(rows["theta"], shown(rows["rms"]), **common)

            ax_residual.plot(
                rows["theta"], shown(rows["residual_m3"]),
                color=color, ls=VH_STYLES[vh], lw=1.8,
                label=f"{mesh} M3/$V_{vh}$",
            )

        # Method 4 does not depend on V_h; select one copy from the long-form CSV.
        mask = (wavy["mesh"] == mesh) & (wavy["vh_order"] == 1)
        rows = ordered_rows(wavy, mask, "theta")
        ax_residual.plot(
            rows["theta"], shown(rows["residual_m4"]),
            color=color, ls=":", lw=2.1, label=f"{mesh} M4",
        )

    for ax, title, ylabel in (
        (ax_max, "Maximum Method 3–Method 4 gap", r"$\max|s_3-s_4|$"),
        (ax_rms, "RMS Method 3–Method 4 gap", "RMS gap"),
        (ax_residual, r"Each method against $\det J\cdot s=1$", "reciprocal residual"),
    ):
        ax.set_title(title)
        ax.set_xlabel(r"geometry parameter $\theta$")
        style_error_axis(ax, ylabel)
        ax.axvline(0.0, color="0.65", lw=0.8)

    ax_max.legend(fontsize=8, ncol=2)
    ax_rms.legend(fontsize=8, ncol=2)
    ax_residual.legend(fontsize=7, ncol=3)

    # Convergence across the exact 1:2:4 model-DOF sequence. Connecting lines
    # encode theta and V_h; marker face colours retain the mesh colour mapping.
    selected_thetas = (-0.8, 0.2, 0.8)
    theta_markers = {-0.8: "s", 0.2: "o", 0.8: "^"}
    theta_grays = {-0.8: "0.25", 0.2: "0.48", 0.8: "0.68"}
    for theta in selected_thetas:
        for vh in (1, 2):
            mask = np.isclose(wavy["theta"], theta) & (wavy["vh_order"] == vh)
            rows = ordered_rows(wavy, mask, "model_dofs")
            if rows.size != 3:
                sys.exit(f"expected three mesh points at theta={theta}, V_h={vh}")
            ax_dof.plot(
                rows["model_dofs"], shown(rows["max_abs"]),
                color=theta_grays[theta], ls=VH_STYLES[vh], lw=1.5,
            )
            for row in rows:
                ax_dof.scatter(
                    row["model_dofs"], shown(np.array([row["max_abs"]]))[0],
                    s=48, marker=theta_markers[theta],
                    facecolor=MESH_COLORS[str(row["mesh"])], edgecolor="white",
                    linewidth=0.6, zorder=3,
                )

    dofs = np.unique(wavy["model_dofs"])
    ax_dof.set_xscale("log", base=2)
    ax_dof.set_xticks(dofs)
    ax_dof.set_xticklabels([f"{int(d):,}" for d in dofs])
    ax_dof.set_xlabel("global model displacement DOFs")
    ax_dof.set_title(r"Mesh convergence at representative $\theta$")
    style_error_axis(ax_dof, r"$\max|s_3-s_4|$")

    theta_handles = [
        Line2D([0], [0], color=theta_grays[t], marker=theta_markers[t], lw=1.5,
               label=fr"$\theta={t:+.1f}$")
        for t in selected_thetas
    ]
    vh_handles = [
        Line2D([0], [0], color="0.35", ls=VH_STYLES[vh], lw=1.6,
               label=f"$V_h$ order {vh}")
        for vh in (1, 2)
    ]
    mesh_handles = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="white",
               markerfacecolor=MESH_COLORS[mesh], markersize=8, label=mesh)
        for mesh in MESHES
    ]
    ax_dof.legend(handles=theta_handles + vh_handles + mesh_handles,
                  fontsize=7.5, ncol=2)

    fig.suptitle(
        "Inverse-determinant accuracy across exact 1:2:4 model DOFs\n"
        f"constant-stretch control: max gap {stretch_gap:.2e}",
        fontsize=14,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "accuracy.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_cost(data: np.ndarray) -> Path:
    method_specs = (
        ("method4", 0, "Method 4", "#D62728"),
        ("method3", 1, "Method 3 / $V_1$", "#009E73"),
        ("method3", 2, "Method 3 / $V_2$", "#0072B2"),
    )
    model_dofs = np.sort(np.unique(data["model_dofs"]))
    fig, ax_time = plt.subplots(figsize=(9.8, 6.0), constrained_layout=True)
    ax_memory = ax_time.twinx()
    memory_envelope_lo = np.inf
    memory_envelope_hi = 0.0

    for method, vh, _, color in method_specs:
        med_time, lo_time, hi_time = [], [], []
        med_alloc, lo_alloc, hi_alloc = [], [], []
        for dofs in model_dofs:
            mask = ((data["model_dofs"] == dofs) & (data["method"] == method) &
                    (data["vh_order"] == vh))
            rows = data[mask]
            if rows.size != 7:
                sys.exit(f"expected seven timings at {dofs} DOFs for {method}, V_h={vh}")
            time_ms = 1e3 * rows["elapsed_seconds"].astype(float)
            alloc_mib = rows["allocated_bytes"].astype(float) / 2**20
            for values, med, lo, hi in (
                (time_ms, med_time, lo_time, hi_time),
                (alloc_mib, med_alloc, lo_alloc, hi_alloc),
            ):
                midpoint = float(np.median(values))
                med.append(midpoint)
                lo.append(midpoint - float(np.quantile(values, 0.25)))
                hi.append(float(np.quantile(values, 0.75)) - midpoint)

        memory_envelope_lo = min(
            memory_envelope_lo, np.min(np.asarray(med_alloc) - np.asarray(lo_alloc)))
        memory_envelope_hi = max(
            memory_envelope_hi, np.max(np.asarray(med_alloc) + np.asarray(hi_alloc)))

        ax_time.errorbar(
            model_dofs, med_time, yerr=np.array([lo_time, hi_time]),
            color=color, ls="-", marker="o", lw=2.0, capsize=4, zorder=3,
        )
        ax_memory.errorbar(
            model_dofs, med_alloc, yerr=np.array([lo_alloc, hi_alloc]),
            color=color, ls="--", marker="o", lw=1.8, capsize=4, alpha=0.9,
        )

    ax_time.set_xscale("log", base=2)
    ax_time.set_yscale("log")
    ax_memory.set_yscale("log")
    ax_time.set_xticks(model_dofs)
    ax_time.set_xticklabels([f"{int(d):,}" for d in model_dofs])
    ax_time.set_xlabel("global model displacement DOFs")
    ax_time.set_ylabel("median elapsed time [ms] — solid lines")
    ax_memory.set_ylabel("median allocated memory [MiB] — dashed lines")
    ax_time.grid(True, which="both", alpha=0.22)

    # Give both y-axes the same logarithmic span so the solid time and dashed
    # memory curves retain parallel slopes. Shift the complete right axis by
    # two minor logarithmic tick intervals so every dashed curve moves down
    # by the same screen-space amount without changing its slope or data.
    time_lo, time_hi = ax_time.get_ylim()
    log_half_span = 0.5 * np.log(time_hi / time_lo)
    minor_ticks_per_decade = 9
    memory_tick_shift = 2
    memory_axis_shift = 0.75 * 10 ** (
        memory_tick_shift / minor_ticks_per_decade)
    memory_centre = memory_axis_shift * np.sqrt(
        memory_envelope_lo * memory_envelope_hi)
    memory_factor = np.exp(log_half_span)
    ax_memory.set_ylim(memory_centre / memory_factor,
                       memory_centre * memory_factor)

    method_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=label)
        for _, _, label, color in method_specs
    ]
    cost_handles = [
        Line2D([0], [0], color="0.25", lw=2.0, ls="-", label="time cost"),
        Line2D([0], [0], color="0.25", lw=2.0, ls="--", label="memory cost"),
    ]
    ax_time.legend(handles=method_handles + cost_handles, fontsize=9, ncol=2,
                   loc="upper left")
    ax_time.set_title(
        "Inverse-determinant cache cost versus mesh size\n"
        "seven post-warm-up measurements; error bars show the interquartile range"
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "cost.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_radius(data: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
    colors = plt.get_cmap("turbo")
    for maxt in range(1, 17):
        rows = ordered_rows(data, data["maxt"] == maxt, "theta")
        if rows.size != 77:
            sys.exit(f"expected 77 theta samples for MAXT={maxt}")
        ax.plot(
            rows["theta"], shown(rows["reciprocal_residual"]),
            color=colors((maxt - 1) / 15), lw=1.7, label=f"MAXT {maxt}",
        )

    measured_lower = float(data["measured_radius_lower"][0])
    measured_upper = float(data["measured_radius_upper"][0])
    ax.set_yscale("log")
    ax.set_xlim(-0.95, 0.95)
    ax.set_xticks((-0.95, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.95))
    ax.set_xlabel(r"geometry parameter $\theta$")
    ax.set_ylabel(r"reciprocal residual $|s\cdot\det J-1|$")
    ax.set_title(
        "Method 4 residual across the parameter interval\n"
        f"measured theta interval ({measured_lower:.4f}, {measured_upper:.4f}); "
        "analytic interval (-1, 1)"
    )
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(title="maximum theta power", fontsize=8, ncol=4, loc="lower right")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "radius.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    accuracy = load_csv("accuracy.csv", 924)
    cost = load_csv("cost.csv", 63)
    radius = load_csv("radius.csv", 1232)

    outputs = (plot_accuracy(accuracy), plot_cost(cost), plot_radius(radius))
    print("wrote inverse-determinant figures:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
