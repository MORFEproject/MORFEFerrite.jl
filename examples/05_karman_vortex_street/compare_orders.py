#!/usr/bin/env python3
"""compare_orders.py — truncation-order comparison of lift and period-averaged TKE (STEP 3).

The DPIM run at MAX_ORD contains the lower-order ROMs exactly (graded solve), so the
order comparison uses ONE run: for every rom_branch_ord<N>.csv written by solve_rom.jl,
the observables are evaluated with the lift polynomial and TKE Gram truncated to
monomials of total degree <= N, along each branch point's circular orbit
z(t) = rho*exp(i*Omega*t):
  - avg_TKE      via tke_from_gram (validation/average_tke.py)
  - max_abs_lift via the lift polynomial L(z) (includes the constant base-flow row)

Outputs results/comparison/{comparison.csv, lift_vs_Re.png, tke_vs_Re.png}.
Usage:  python3 compare_orders.py            (no arguments)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "validation"))
from average_tke import tke_from_gram  # noqa: E402

NS = 256  # orbit samples per period


def load_gram(data_dir: Path) -> dict:
    G = (np.loadtxt(data_dir / "tke_gram_re.csv", delimiter=",")
         + 1j * np.loadtxt(data_dir / "tke_gram_im.csv", delimiter=","))
    A = np.loadtxt(data_dir / "tke_avector.csv", delimiter=",").astype(int)
    return {"G": np.atleast_2d(G), "Avector": np.atleast_2d(A)}


def load_lift(data_dir: Path):
    """Lift polynomial, NVAR inferred from the header.

    A promoted run has exp_1..exp_NVAR, not exp_1..exp_3. Hardcoding three columns read
    the FIRST PROMOTED COORDINATE as the η exponent and ignored η entirely, which put the
    lift at 1e19 instead of 1e-2.
    """
    exps, coeffs = [], []
    with open(data_dir / "L_coefficients.csv") as f:
        rdr = csv.DictReader(f)
        ecols = [c for c in rdr.fieldnames if c.startswith("exp_")]
        for row in rdr:
            exps.append(tuple(int(row[c]) for c in ecols))
            coeffs.append(float(row["L_re"]) + 1j * float(row["L_im"]))
    return np.array(exps, int), np.array(coeffs, complex)


def load_r_poly(data_dir: Path):
    """Reduced dynamics R, needed to SLAVE the promoted coordinates."""
    exps, comps = [], None
    with open(data_dir / "R_coefficients.csv") as f:
        rdr = csv.DictReader(f)
        ecols = [c for c in rdr.fieldnames if c.startswith("exp_")]
        rcols = [c for c in rdr.fieldnames if c.endswith("_re") and c.startswith("R")]
        rows = list(rdr)
    exps = np.array([[int(r[c]) for c in ecols] for r in rows], int)
    comps = np.array([[float(r[c]) + 1j * float(r[c[:-3] + "_im"]) for c in rcols]
                      for r in rows], complex)
    return exps, comps


def eval_poly(state: np.ndarray, exps: np.ndarray, coeffs: np.ndarray):
    """Σ_m coeffs[m] · Π_i state[..., i]**exps[m, i].  state is (..., nvar)."""
    out = np.zeros(state.shape[:-1] + coeffs.shape[1:], dtype=complex)
    for m in range(exps.shape[0]):
        term = np.ones(state.shape[:-1], dtype=complex)
        for i, e in enumerate(exps[m]):
            if e:
                term = term * state[..., i] ** e
        out += term[..., None] * coeffs[m] if coeffs.ndim > 1 else term * coeffs[m]
    return out


def slaved_states(z: np.ndarray, eta: float, rexps: np.ndarray, rcoef: np.ndarray):
    """Full state (ns, nvar) along one orbit, with promoted coordinates slaved.

    Mirrors solver/rom_palc.jl's `_rom_R1` and invariants.jl: on the slow manifold the
    promoted coordinates satisfy R_k = 0, and R is exactly AFFINE in them, so this is one
    small linear solve per sample rather than an iteration. Setting them to zero instead
    would drop the mean-flow distortion from every observable.
    """
    nvar = rexps.shape[1]
    npro = nvar - 3
    ns = z.size
    state = np.zeros((ns, nvar), dtype=complex)
    state[:, 0] = z
    state[:, 1] = np.conj(z)
    state[:, -1] = eta
    if npro == 0:
        return state
    rows = slice(2, 2 + npro)
    b = eval_poly(state, rexps, rcoef)[:, rows]                 # R_k at y = 0
    A = np.empty((ns, npro, npro), dtype=complex)
    for j in range(npro):
        sj = state.copy()
        sj[:, 2 + j] = 1.0
        A[:, :, j] = eval_poly(sj, rexps, rcoef)[:, rows] - b
    # b is a STACK of right-hand sides: (ns, npro, 1), not an (npro, ns) matrix.
    state[:, rows] = np.linalg.solve(A, -b[:, :, None])[:, :, 0]
    return state


def lift_series(state: np.ndarray, exps: np.ndarray, coeffs: np.ndarray):
    return eval_poly(state, exps, coeffs).real


def truncate_gram(gram: dict, order: int) -> dict:
    sel = gram["Avector"].sum(axis=1) <= order
    return {"G": gram["G"][np.ix_(sel, sel)], "Avector": gram["Avector"][sel]}


def tke_from_state(state: np.ndarray, gram: dict):
    """Period-averaged fluctuation TKE from the full state, for ANY number of coordinates.

    validation/average_tke.py's `tke_from_gram` builds ζ_m = z^A0 · z̄^A1 · η^A2, which is
    the NVAR = 3 layout. With promoted coordinates the Avector is NVAR-wide and that
    hardcoding silently reads a promoted exponent as η's. This is the same formula over
    whatever coordinates the run actually has.
    """
    A, G = np.asarray(gram["Avector"]), np.asarray(gram["G"])
    L = A.shape[0]
    if G.shape != (L, L):
        raise ValueError("gram['G'] must be (L, L) with L = rows of Avector")
    zeta = np.empty((state.shape[0], L), dtype=complex)
    for m in range(L):
        term = np.ones(state.shape[0], dtype=complex)
        for i, e in enumerate(A[m]):
            if e:
                term = term * state[:, i] ** e
        zeta[:, m] = term
    zf = zeta - zeta.mean(axis=0)
    ke_t = 0.5 * np.real(np.sum((zf @ G) * zf, axis=1))
    return float(ke_t.mean()), ke_t


def process_branch(branch_csv: Path, gram: dict, exps: np.ndarray, coeffs: np.ndarray,
                   rexps: np.ndarray, rcoef: np.ndarray):
    order = int(re.search(r"rom_branch_ord(\d+)\.csv$", branch_csv.name).group(1))
    gram_N = truncate_gram(gram, order)
    keep = (exps.sum(axis=1) <= order) & ~((exps[:, 0] == 0) & (exps[:, 1] == 0))  # drop base-flow (z=0) rows: base flow has zero lift
    exps_N, coeffs_N = exps[keep], coeffs[keep]
    # Truncate R the same way the branch was traced, so the slaved y matches the orbit.
    rkeep = rexps.sum(axis=1) <= order
    rexps_N, rcoef_N = rexps[rkeep], rcoef[rkeep]
    rows = []
    with open(branch_csv) as f:
        for row in csv.DictReader(f):
            eta, Re = float(row["eta"]), float(row["Re"])
            rho, om, T = float(row["rho"]), float(row["omega"]), float(row["T"])
            th = 2 * np.pi * np.arange(NS) / NS          # uniform over one period
            z = rho * np.exp(1j * th)
            state = slaved_states(z, eta, rexps_N, rcoef_N)
            tke, _ = tke_from_state(state, gram_N)
            max_lift = float(np.max(np.abs(lift_series(state, exps_N, coeffs_N))))
            rows.append((order, eta, Re, rho, om, T, tke, max_lift))
    return rows


def process_run(run_dir: Path):
    data = run_dir / "data"
    branches = sorted(data.glob("rom_branch_ord*.csv"))
    if not branches:
        return []
    gram = load_gram(data)
    exps, coeffs = load_lift(data)
    rexps, rcoef = load_r_poly(data)
    return [r for b in branches
            for r in process_branch(b, gram, exps, coeffs, rexps, rcoef)]


def load_fom_reference(run_dir: Path):
    """(Re, max_abs_lift, avg_TKE, converged) rows from fom_reference.jl, if present."""
    path = run_dir / "data" / "fom_reference.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((float(row["Re"]), float(row["max_abs_lift"]),
                         float(row["avg_TKE"]),
                         row["converged"].strip().lower() == "true"))
    return rows


def run_n_free(run_dir):
    """Free-DOF count the run was computed with, from summary.txt; None if absent."""
    path = run_dir / "summary.txt"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*n_free\s*:\s*(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def main():
    run_dirs = sorted(p for p in (HERE / "results").glob("Re*_ord*")
                      if any((p / "data").glob("rom_branch_ord*.csv")))
    if not run_dirs:
        sys.exit("no rom_branch_ord*.csv found — run main.jl then solve_rom.jl first")

    # The order comparison is only meaningful across truncations of ONE run: orders 3/5/7/9
    # of a single order-9 solve share a mesh, so their differences are TRUNCATION error.
    # A FAST run (14 436 DOFs) sitting beside a FULL one (57 860) would be drawn as if it
    # were another truncation, when it is a different DISCRETISATION — the coarse curve
    # would read as order-3 truncation error. Keep the finest mesh and say what was dropped.
    sizes = {d: run_n_free(d) for d in run_dirs}
    known = {n for n in sizes.values() if n is not None}
    if len(known) > 1:
        keep = max(known)
        dropped = [d for d in run_dirs if sizes[d] is not None and sizes[d] != keep]
        run_dirs = [d for d in run_dirs if sizes[d] is None or sizes[d] == keep]
        print(f"mesh mismatch: keeping the {keep}-DOF run(s); dropped "
              + ", ".join(f"{d.name} ({sizes[d]} DOFs)" for d in dropped))
    # ONE FIGURE PER RUN. Truncation orders of a single solve belong on the same axes —
    # their differences are truncation error. A promoted run (extra master coordinates) is
    # a different COORDINATE SYSTEM, not another truncation, so it gets its own figure.
    per_run = [(d, process_run(d)) for d in run_dirs]
    per_run = [(d, rows) for d, rows in per_run if rows]
    all_rows = [r for _d, rows in per_run for r in rows]
    # The FOM reference is looked up across ALL result dirs, not just the ones holding
    # ROM branches. It comes from direct time integration, so it is independent of the
    # DPIM run entirely and often outlives it: the order-9 DPIM artefacts were discarded
    # when a conjugate-pairing fix invalidated them, while the FOM data stayed valid.
    # Tying the two together silently dropped the reference from these plots.
    fom_dirs = sorted(p for p in (HERE / "results").glob("Re*_ord*")
                      if (p / "data" / "fom_reference.csv").exists())
    fom_rows = [r for d in fom_dirs for r in load_fom_reference(d)]
    if fom_rows:
        print(f"FOM reference: {len(fom_rows)} points from "
              f"{', '.join(d.name for d in fom_dirs)}")
    else:
        print("FOM reference: none found (run fom_reference.jl)")
    orders = sorted({int(r[0]) for r in all_rows})
    print(f"ROM branches: order(s) {orders} from "
          f"{', '.join(d.name for d in run_dirs)}")
    missing = [o for o in (3, 5, 7, 9) if o not in orders]
    if missing:
        print(f"  NOTE: order(s) {missing} absent — they come from the FULL order-9 run "
              f"(config.jl TRUNC_ORDERS); re-run main.jl without MORFE_FAST.")

    out = HERE / "results" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    hdr = "order,eta,Re,rho,omega,T,avg_TKE,max_abs_lift"

    import matplotlib
    import matplotlib.pyplot as plt

    for run_dir, rows in per_run:
        tag = run_dir.name
        np.savetxt(out / f"comparison_{tag}.csv", np.array(rows), delimiter=",",
                   header=hdr, comments="", fmt="%.10e")
        make_figures(np.array(rows), fom_rows, out, tag, plt, matplotlib)

    # Combined CSV kept for convenience; the figures are per run.
    np.savetxt(out / "comparison.csv", np.array(all_rows), delimiter=",",
               header=hdr, comments="", fmt="%.10e")
    print(f"wrote {out / 'comparison.csv'}  ({len(all_rows)} rows, all runs)")


def make_figures(arr, fom_rows, out, tag, plt, matplotlib):
    colors = {3: "k", 5: "r", 7: "g", 9: "b"}
    for col, stem, ylabel in ((7, "lift_vs_Re", "max |lift|"),
                              (6, "tke_vs_Re", "period-averaged TKE")):
        fname = f"{stem}_{tag}.png"
        plt.figure(figsize=(4, 4), dpi=150)

        for o in sorted(set(arr[:, 0].astype(int)))[::-1]:
            sel = arr[arr[:, 0] == o]
            diff = np.where(np.sign(sel[1:, 2] - sel[0:-1, 2]) == -1)[0] + 1
            if len(diff) > 0:
                plt.plot(sel[:diff[0], 2], sel[:diff[0], col], ms=3, label=f"order {o}",
                         ls="-", color=colors[o], lw=2)
                plt.plot(sel[diff[0]:, 2], sel[diff[0]:, col], ms=3, ls="--",
                         color=colors[o], lw=2)
            else:
                plt.plot(sel[:, 2], sel[:, col], ms=3, label=f"order {o}",
                         color=colors[o], lw=2)

        if fom_rows:
            fr = np.array([(re_, lift, tke) for re_, lift, tke, conv in fom_rows if conv])
            if fr.size:
                plt.plot(fr[:, 0], fr[:, 1] if col == 7 else fr[:, 2], marker="o", ms=6,
                         lw=0, markeredgewidth=2, markeredgecolor="k",
                         markerfacecolor="None", label="FOM")

        plt.plot([0, 48.9844], [0.0, 0.0], color="k", ls="-", marker="o", ms=7, lw=2)
        plt.plot([48.9844, 100], [0.0, 0.0], color="k", ls="--", lw=2)

        # Axis limits: the paper values are the FLOOR, expanded if the data runs past
        # them. Hardcoding them alone silently cropped the branch when FOM_REF_RE was
        # extended to Re = 64 — the curve was drawn, just outside the axes.
        ydata = [arr[arr[:, 0] == o][:, col].max()
                 for o in set(arr[:, 0].astype(int))]
        if fom_rows:
            ydata.append(max((lift if col == 7 else tke)
                             for _re, lift, tke, conv in fom_rows if conv))
        ymax = max(max(ydata) * 1.12, 0.02 if col == 6 else 0.016)
        xmax = max(57.0, np.ceil(max(arr[:, 2].max(),
                   max((r[0] for r in fom_rows), default=0.0)) + 1.0))
        plt.ylim([-0.001, ymax])
        plt.xlim([48.0, xmax])
        plt.xlabel("Re")
        plt.ylabel(ylabel)
        plt.title(tag, fontsize=9)
        # bbox_inches="tight" so the y-label is not clipped out of the saved PNG.
        # (tight_layout is left off deliberately — it would change the axes geometry.)
        plt.savefig(out / fname, dpi=200, bbox_inches="tight")
        # Only show interactively. plt.show() on a non-interactive backend blocks
        # forever waiting for a window that never opens, which stalls any scripted run.
        if matplotlib.is_interactive() or matplotlib.get_backend().lower() not in (
                "agg", "pdf", "ps", "svg", "template"):
            plt.show()
        else:
            plt.close(plt.gcf())
        print(f"wrote {out / fname}")



if __name__ == "__main__":
    main()
