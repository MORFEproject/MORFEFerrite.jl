#!/usr/bin/env python3
"""figures_geometry.py — the actual geometries across the parameter range.

Reads   results/data/geometry_gallery.csv   (written by geometry_gallery.jl)
Writes  results/figures/geometry_gallery.png
Usage   python3 figures_geometry.py            select the grid below; the data
        python3 figures_geometry.py --force    regenerates automatically

Small multiples: one panel per sampled parameter pair, the ACTUAL geometry drawn
over the reference geometry, with dL/L0 and h/L(theta) printed on each. A single
overlaid axes would collapse fifteen outlines onto one another; the grid keeps
each configuration legible and lets the two parameter directions be read as the
two directions of the grid.

Everything is at TRUE 1:1 SCALE. The beam is 1000 long and 10 thick, so it looks
as slender as it actually is — that slenderness is information, and exaggerating
the section would distort the one ratio (rise against span) the figure exists to
show. Every panel shares limits, so panels are directly comparable.

h/L(theta) is normalised by the ACTUAL length, so it varies in BOTH grid
directions even though the rise depends only on the column: the same physical
rise is a larger fraction of a shortened beam. That is why it is printed on
every panel rather than labelled once on an axis.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "data" / "geometry_gallery.csv"
OUTDIR = HERE / "results" / "figures"

# ══ THE GRID — edit these two lists ═════════════════════════════════════════
#
# Both in PERCENT OF THE REFERENCE LENGTH L0, which is what the panels are
# labelled with. Changing either list is enough: this script re-invokes
# geometry_gallery.jl automatically whenever the CSV on disk does not match, so
# there is nothing to keep in step by hand.
#
#   DL_OVER_L0_PCT   ROWS    — span change,  dL/L0   (exactly theta1)
#   H_OVER_L0_PCT    COLUMNS — arch height,  h/L0    (rise = theta2 * h0)
#
# Same sense as the heatmaps: arch along x, span change along y.
#
# h/L0 is the COLUMN label and is constant down a column. The h/L(theta)
# printed in each panel divides by the ACTUAL length instead, so it varies in
# BOTH directions — the same physical rise is a larger fraction of a
# shortened beam.
#
# Julia owns h0 and L0 and does the conversion, so those constants exist in
# exactly one place (config.jl) and cannot drift.

DL_OVER_L0_PCT = [-20.0, 0.0, 20.0]
H_OVER_L0_PCT = [0.0, 2.5, 5.0, 7.5, 10.0]

# ═══════════════════════════════════════════════════════════════════════════

REF_COLOUR = "0.72"
ACT_COLOUR = "#1f5fa9"
ACT_FILL = "#1f5fa9"
GRID_RTOL = 1e-9


def _grid_in(path: Path):
    """The (dL/L0, h/L0) grid the CSV on disk actually holds, or None."""
    if not path.is_file():
        return None
    try:
        raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                            encoding="utf-8")
        return (np.unique(raw["dL_over_L0_pct"]), np.unique(raw["h_over_L0_pct"]))
    except (ValueError, KeyError):
        return None      # older schema without h_over_L0_pct — regenerate


def ensure_data(force: bool = False) -> None:
    """Regenerate the CSV if it does not match the requested grid.

    The geometry stays in Julia — it is drawn from `sine_bend_displacement`, the
    same function the model integrates — so selecting the grid here does not
    duplicate any physics in Python.
    """
    want = (np.sort(np.asarray(DL_OVER_L0_PCT, float)),
            np.sort(np.asarray(H_OVER_L0_PCT, float)))
    have = _grid_in(CSV)
    fresh = (not force and have is not None
             and len(have[0]) == len(want[0]) and len(have[1]) == len(want[1])
             and np.allclose(have[0], want[0], rtol=GRID_RTOL)
             and np.allclose(have[1], want[1], rtol=GRID_RTOL))
    if fresh:
        print(f"grid unchanged — reusing {CSV.name}")
        return

    julia = shutil.which("julia")
    if julia is None:
        sys.exit("the grid changed but `julia` is not on PATH — cannot regenerate")
    env = dict(os.environ)
    env["GALLERY_DL_PCT"] = ",".join(repr(float(v)) for v in DL_OVER_L0_PCT)
    env["GALLERY_H_PCT"] = ",".join(repr(float(v)) for v in H_OVER_L0_PCT)
    print(f"grid changed — running geometry_gallery.jl "
          f"({len(DL_OVER_L0_PCT)} x {len(H_OVER_L0_PCT)} panels)")
    r = subprocess.run([julia, f"--project={HERE}", str(HERE / "geometry_gallery.jl")],
                       cwd=HERE, env=env)
    if r.returncode != 0:
        # Do NOT fall through to plotting: that would silently draw the old grid
        # under the new grid's labels.
        sys.exit(f"geometry_gallery.jl failed (exit {r.returncode}); nothing plotted")


def load(path: Path):
    if not path.is_file():
        sys.exit(f"no gallery at {path} — run geometry_gallery.jl first")
    raw = np.genfromtxt(path, delimiter=",", names=True,
                        dtype=None, encoding="utf-8")

    # Panels are keyed by the PARAMETER VALUES, not by the row/col indices the
    # emitter happens to write. The layout is then a property of the data, so
    # reordering the loops on the Julia side cannot silently transpose the
    # figure, and a stale CSV cannot be drawn under the wrong labels.
    panels = defaultdict(lambda: {"reference": [], "actual": [], "meta": None})
    by_curve = defaultdict(list)
    curve_key = {}
    for rec in raw:
        c = int(rec["curve"])
        by_curve[c].append((float(rec["x"]), float(rec["y"])))
        if c not in curve_key:
            curve_key[c] = (float(rec["dL_over_L0_pct"]),
                            float(rec["h_over_L0_pct"]),
                            float(rec["h_over_L_pct"]),
                            str(rec["part"]), str(rec["kind"]))

    for c, pts in by_curve.items():
        dl, h0, hl, part, kind = curve_key[c]
        pan = panels[(dl, h0)]
        pan[part].append((kind, np.asarray(pts)))
        pan["meta"] = (dl, h0, hl)
    return panels


def main() -> None:
    ensure_data(force="--force" in sys.argv)
    panels = load(CSV)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Same sense as the heatmaps: the ARCH runs along the columns (x there) and
    # the SPAN CHANGE down the rows (y there), with the span increasing upward.
    # Keyed by parameter VALUE, so the layout is a property of the data rather
    # than of the order the emitter happened to write its loops in.
    row_vals = sorted({dl for dl, _ in panels}, reverse=True)   # top = longest
    col_vals = sorted({h0 for _, h0 in panels})                 # left = straight

    # Common limits across every panel, so nothing is silently rescaled.
    allx = np.concatenate([p[:, 0] for pan in panels.values()
                           for _, p in pan["actual"] + pan["reference"]])
    ally = np.concatenate([p[:, 1] for pan in panels.values()
                           for _, p in pan["actual"] + pan["reference"]])
    padx = 0.03 * (allx.max() - allx.min())
    pady = 0.12 * max(ally.max() - ally.min(), 1.0)
    xlim = (allx.min() - padx, allx.max() + padx)
    ylim = (ally.min() - pady, ally.max() + pady)

    # Size the figure from the panel aspect and the grid shape, so an arbitrary
    # grid still comes out proportioned: a 9:1 beam across many columns would
    # otherwise give a figure that is either enormous or unreadably squat.
    nrow, ncol = len(row_vals), len(col_vals)
    data_aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    panel_w = min(3.8, 17.0 / ncol)
    panel_h = max(0.66, panel_w / data_aspect)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * panel_w + 1.5, nrow * panel_h + 1.9),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(nrow, ncol)

    for i, dl_v in enumerate(row_vals):
        for j, h0_v in enumerate(col_vals):
            ax = axes[i, j]
            pan = panels[(dl_v, h0_v)]
            dl, h0, hl = pan["meta"]

            for kind, p in pan["reference"]:
                ax.plot(p[:, 0], p[:, 1], color=REF_COLOUR,
                        lw=1.1 if kind == "boundary" else 0.4, zorder=1)

            # Fill between the two boundary curves so the body reads as a band
            # rather than two lines. They are emitted lower-surface first.
            bounds = [p for kind, p in pan["actual"] if kind == "boundary"]
            spanwise = [p for p in bounds if len(p) > 2]
            if len(spanwise) == 2:
                lo, hi = sorted(spanwise, key=lambda p: p[:, 1].mean())
                ax.fill(np.concatenate([lo[:, 0], hi[::-1, 0]]),
                        np.concatenate([lo[:, 1], hi[::-1, 1]]),
                        color=ACT_FILL, alpha=0.16, lw=0, zorder=2)
            for kind, p in pan["actual"]:
                ax.plot(p[:, 0], p[:, 1], color=ACT_COLOUR,
                        lw=1.4 if kind == "boundary" else 0.45,
                        alpha=1.0 if kind == "boundary" else 0.55, zorder=3)

            ax.set_aspect("equal")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            # dL/L0 is constant down a column, so it is a column header (below);
            # h/L(theta) varies panel by panel and is printed here. zorder above
            # the curves — at +20 % span the arch limb runs under this corner.
            ax.text(0.012, 0.90, f"$h/L(\\theta) = {hl:.2f}\\,\\%$",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="0.75", lw=0.7, alpha=0.95))
            # h/L0 is the SELECTED arch value, constant down a column, so it is
            # the column header. dL/L0 is constant along a row and labels the
            # row. h/L(theta) divides by the ACTUAL length, so it varies in BOTH
            # directions and has to be printed on every panel.
            if i == 0:
                ax.set_title(f"$h/L_0 = {h0:.4g}\\,\\%$", fontsize=11, pad=6)
            ax.tick_params(labelsize=7)
            if i == nrow - 1:
                ax.set_xlabel("$x$", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"$\\Delta L/L_0 = {dl:+.4g}\\,\\%$", fontsize=10)

    fig.suptitle(
        "Actual geometries over the parameter range, drawn over the reference "
        "geometry (grey)\n"
        "true 1:1 scale — element edges from the reference mesh, carried through "
        "the same analytic map",
        fontsize=12)

    out = OUTDIR / "geometry_gallery.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")

    # Aspect must be 1 on every panel, or a panel is silently stretched and the
    # rise-against-span ratio the figure exists to show is not what it appears.
    bad = [(i, j) for i in range(axes.shape[0]) for j in range(axes.shape[1])
           if axes[i, j].get_aspect() != 1.0]
    print(f"  panels          : {nrow} rows (dL/L0) x {ncol} cols (h/L0)")
    print(f"  aspect == 1     : {'all' if not bad else f'VIOLATED at {bad}'}")
    print(f"  x range         : {allx.min():.1f} .. {allx.max():.1f}")
    print(f"  y range         : {ally.min():.1f} .. {ally.max():.1f}")


if __name__ == "__main__":
    main()
