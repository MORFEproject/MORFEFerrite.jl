#!/usr/bin/env python3
"""figures.py — accuracy of the parametric ROM's first bending frequency over (theta1, theta2).

Reads   results/data/frequency_accuracy_map.csv   (written by frequency_accuracy_map.jl)
Writes  results/figures/frequency_accuracy_map.png
Usage   python3 figures.py            (no arguments)

Left panel is the requested heatmap: log10 of the relative error in the first
bending frequency, against an EXACT frozen-theta assembly (a one-term series, so
nothing is truncated on the reference side), with isolines at exact powers of
ten. Right panel is the percent CHANGE in that frequency relative to the
expansion point, so the accuracy is read against how much physics the
parametrisation is actually being asked to capture.

Axes are physical: span change dL/L and arch height h/L, both in percent. The
grid is clustered about the expansion point (dL/L = 0, h/L = 0), which is where
the truncation error falls fastest and where the arch does most of its
stiffening.

Two things the plot has to say honestly, both established by measurement:

  * theta2 is EXACTLY isochoric. grad(psi2) is nilpotent, so det J = 1 + theta1
    independent of theta2, and adj J is degree 1 in theta2. The stiffness is
    therefore an exact polynomial of theta2-degree 2 and the box bound of 2 loses
    nothing. All truncation error on this map belongs to theta1.

  * The map has a NUMERICAL FLOOR near 1e-9. At theta1 = 0 the operators agree
    entrywise to 6e-16 of max|K|, yet the frequencies differ by ~1e-9: max|K| is
    set by axial stiffness while the bending eigenvalue sits ~1e-6 below it, so
    operator-scale roundoff is amplified by ~1e6 at the modal scale. Values at or
    below the floor mean "exact", not "slightly wrong".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "data" / "frequency_accuracy_map.csv"
OUTDIR = HERE / "results" / "figures"
FLOOR = 1e-9          # conditioning floor, see the module docstring


def load(path: Path):
    """Grid is CLUSTERED about the expansion point, so the sampling is not
    uniform; and the vertical coordinate is normalised by the ACTUAL length,
    which makes it depend on theta1 as well. The plotting grid is therefore a
    genuine 2D quadmesh, not a tensor product of two axis vectors — imshow, or
    pcolormesh with 1D coordinates, would silently misplace every cell."""
    if not path.is_file():
        sys.exit(f"no map at {path} — run frequency_accuracy_map.jl first")
    raw = np.genfromtxt(path, delimiter=",", names=True)
    t1 = np.unique(raw["theta1"])
    t2 = np.unique(raw["theta2"])
    shape = (t2.size, t1.size)

    def grid(name):
        g = np.full(shape, np.nan)
        i1 = np.searchsorted(t1, raw["theta1"])
        i2 = np.searchsorted(t2, raw["theta2"])
        g[i2, i1] = raw[name]
        return g

    # Each normalisation belongs to its QUANTITY, not to a screen direction: the
    # span change is measured against the REFERENCE length and the arch height
    # against the ACTUAL one. Swapping which quantity is drawn
    # horizontally therefore leaves both definitions untouched.
    #
    # Horizontal: arch height over the actual length,
    #     h / L(theta) = theta2 * h0 / ((1 + theta1) * L0)
    # The transverse displacement does not depend on theta1 but the span does, so
    # the same physical rise is a LARGER fraction of a shortened beam — which is
    # what turns the plotting domain from a rectangle into a fan.
    names = raw.dtype.names
    if "rise_over_Lact_pct" in names:
        X = grid("rise_over_Lact_pct")
    else:  # derive it from the two columns every version of the CSV carries
        X = grid("rise_over_L_pct") / (1.0 + grid("theta1"))

    # Vertical: span change over the reference length, dL / L0 = theta1.
    Y = np.tile(100.0 * t1, (t2.size, 1))

    # omega0 is the reference configuration (straight beam, nominal span) — the
    # point the theta-series is expanded about. Both the ROM and the FOM changes
    # are normalised by the SAME omega0, which is what makes the two panels
    # directly comparable. At theta = 0 the two agree to 4e-16, so the choice of
    # which omega0 is immaterial in value but not in principle.
    at0 = (raw["theta1"] == 0.0) & (raw["theta2"] == 0.0)
    omega0 = float(raw["omega_ref"][at0][0])

    dom_fom = grid("domega_pct")
    if "domega_rom_pct" in names:
        dom_rom = grid("domega_rom_pct")
    else:  # derive it — omega_rom is in every version of the CSV
        dom_rom = 100.0 * (grid("omega_rom") - omega0) / omega0

    return (X, Y, grid("rel_err"), dom_rom, dom_fom, grid("idx_ref"))


def main() -> None:
    X, Y, err, dom_rom, dom_fom, idx = load(CSV)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Clip at the floor so the colour scale shows truncation, not roundoff.
    shown = np.maximum(err, FLOOR)
    log_err = np.log10(shown)

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.6), constrained_layout=True)

    # ── left: the accuracy heatmap ──
    ax = axes[0]
    lo = np.floor(np.log10(FLOOR))
    hi = np.ceil(log_err.max())
    levels = np.arange(lo, hi + 1)
    # Smooth, continuous shading: `shading="gouraud"` interpolates between grid
    # points instead of drawing one flat cell each, which also lets the mesh take
    # 2D coordinates. The decade structure is carried by the isolines below
    # rather than by banded colour.
    norm = Normalize(vmin=levels[0], vmax=levels[-1])
    im = ax.pcolormesh(X, Y, log_err, cmap="magma_r", norm=norm,
                       shading="gouraud", rasterized=True)
    # Isolines at exact powers of ten. `levels` are integer log10 values, so each
    # contour is a decade boundary; they coincide with the colour bands, which is
    # what makes the decade structure readable rather than decorative.
    inner = levels[1:-1]
    if inner.size:
        # linestyles="solid" is not cosmetic: matplotlib dashes contours at
        # NEGATIVE levels by default, and every level here is a negative log10.
        cs = ax.contour(X, Y, log_err, levels=inner, colors="k",
                        linewidths=1.1, alpha=0.85, linestyles="solid")
        ax.clabel(cs, inline=True, fontsize=8,
                  fmt=lambda v: f"$10^{{{v:.0f}}}$")
    cb = fig.colorbar(im, ax=ax, ticks=levels)
    cb.ax.set_yticklabels([f"$10^{{{v:.0f}}}$" for v in levels])
    cb.set_label(r"relative error in $\omega$  (decades)")
    # The near-vertical contours ARE the result: theta2 is exactly isochoric, so
    # all truncation error belongs to theta1. Said in the title rather than as a
    # floating annotation, which collided with the contour labels.
    ax.set_title("Relative error in the first bending frequency\n"
                 r"(vs an exact frozen-$\theta$ assembly)" "\n"
                 r"contours are horizontal: error depends on $\Delta L/L_0$ alone",
                 fontsize=11)

    # theta1 = 0 is exact by construction; mark it so the floor is not misread.
    ax.axhline(0.0, color="#00e5ff", lw=1.4, ls="--", alpha=0.95)
    ax.text(np.nanmin(X) + 0.62 * (np.nanmax(X) - np.nanmin(X)), 0.0,
            r"$\Delta L/L_0=0$: expansion exact" "\n" r"(map is at its $10^{-9}$ floor)",
            color="#0097a7", fontsize=8, va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#00e5ff",
                      lw=0.8, alpha=0.85))

    # ── middle and right: the physics, as a PERCENT CHANGE ──
    # Two panels on ONE shared scale so they can be read against each other: what
    # the parametric ROM predicts, and what the full-order model actually gives.
    # Diverging about zero — blue = softer than the reference configuration,
    # red = stiffer.
    lo_d = float(min(np.nanmin(dom_rom), np.nanmin(dom_fom)))
    hi_d = float(max(np.nanmax(dom_rom), np.nanmax(dom_fom)))
    dnorm = TwoSlopeNorm(vcenter=0.0, vmin=min(lo_d, -1e-9), vmax=hi_d)

    # The frequency DROPS where the span grows and the arch is shallow, so this
    # field is genuinely two-sided. The negative band is narrow (about a tenth of
    # the full range), so a uniform level step would put at most one line in it —
    # step finely below zero and coarsely above, and give the colourbar the SAME
    # levels so the two readings agree.
    neg = np.arange(-10.0 * np.ceil(abs(min(lo_d, 0.0)) / 10.0), 0.0, 10.0)
    pos = np.arange(50.0, hi_d + 1.0, 50.0)
    dlevels = np.concatenate([neg, [0.0], pos])
    dlevels = dlevels[(dlevels >= lo_d) & (dlevels <= hi_d)]
    thin = dlevels[dlevels != 0.0]   # zero drawn once, thicker, below

    panels = ((axes[1], dom_rom, "predicted by the parametric ROM"),
              (axes[2], dom_fom, "computed in the full-order model"))
    for ax, field, what in panels:
        im2 = ax.pcolormesh(X, Y, field, cmap="RdBu_r", norm=dnorm,
                            shading="gouraud", rasterized=True)
        cs2 = ax.contour(X, Y, field, levels=thin, colors="k", linewidths=0.7,
                         alpha=0.6, linestyles="solid")
        ax.clabel(cs2, fontsize=7, fmt="%+.0f%%")
        # Zero is the boundary between softening and stiffening.
        if lo_d < 0.0 < hi_d:
            z = ax.contour(X, Y, field, levels=[0.0], colors="k", linewidths=2.0)
            ax.clabel(z, fontsize=8, fmt="%+.0f%%")
        # Where the tracked bending mode is no longer mode 1. omega is CONTINUOUS
        # across this line — the e2 bending mode simply overtakes the e3 lateral
        # mode, which sits at a constant 1.262 because the arch bends in y and
        # does not stiffen z. Taking omega_1 instead of tracking by MAC would
        # have silently switched to that lateral mode here.
        if np.nanmax(idx) > 1:
            ax.contour(X, Y, idx, levels=[1.5], colors="r", linewidths=1.6)
        ax.set_title("Change in the first bending frequency\n"
                     r"relative to the expansion point $\omega_0$" "\n"
                     f"{what}", fontsize=11)

    # One colourbar for both, because they share a scale — two would invite the
    # reader to compare them on different footings.
    cb2 = fig.colorbar(im2, ax=[axes[1], axes[2]], ticks=dlevels, pad=0.02)
    cb2.ax.set_yticklabels([f"{v:+.0f}" for v in dlevels])
    cb2.set_label(r"$\Delta\omega/\omega_0$   [%]   (negative = softer)")

    for ax in axes:
        ax.set_xlabel(r"arch height   $h / L(\theta)$   [%]   "
                      r"(actual length)")
        ax.set_ylabel(r"span change   $\Delta L / L_0$   [%]   "
                      r"(reference length)")

    out = OUTDIR / "frequency_accuracy_map.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")

    finite = err[np.isfinite(err)]
    print(f"  points          : {finite.size}")
    print(f"  worst rel error : {finite.max():.3e}")
    print(f"  median          : {np.median(finite):.3e}")
    print(f"  ROM d-omega     : {np.nanmin(dom_rom):+.1f} % .. {np.nanmax(dom_rom):+.1f} %")
    print(f"  FOM d-omega     : {np.nanmin(dom_fom):+.1f} % .. {np.nanmax(dom_fom):+.1f} %")
    print(f"  dL/L0 = 0 col   : {np.nanmax(err[:, np.argmin(np.abs(Y[0]))]):.3e}"
          f"   (numerical floor, expansion is exact there)")


if __name__ == "__main__":
    main()
