#!/usr/bin/env python3
"""figures.py — lift and period-averaged TKE against Re, one figure per run.

The DPIM run at MAX_ORD contains the lower-order ROMs exactly (the cohomological solve is
graded), so the order comparison uses ONE run: for every `rom_branch_ord<N>.csv` the
observables are evaluated with the lift polynomial and TKE Gram truncated to degree <= N,
along each branch point's circular orbit z(t) = rho*exp(i*Omega*t).

Reads   results/Re*_ord*_*modes/data/
Writes  results/comparison/{comparison_<tag>.csv, lift_vs_Re_<tag>.png, tke_vs_Re_<tag>.png}
Usage   python3 figures.py            (no arguments)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NS = 256          # orbit samples per period
RE_MIN, RE_MAX = 48.0, 70.0


# ── loading ──────────────────────────────────────────────────────────────────

def load_gram(data_dir: Path) -> dict:
    G = (np.loadtxt(data_dir / "tke_gram_re.csv", delimiter=",")
         + 1j * np.loadtxt(data_dir / "tke_gram_im.csv", delimiter=","))
    A = np.loadtxt(data_dir / "tke_avector.csv", delimiter=",").astype(int)
    return {"G": np.atleast_2d(G), "Avector": np.atleast_2d(A)}


def load_lift(data_dir: Path):
    """Lift polynomial, NVAR inferred from the header.

    A promoted run has exp_1..exp_NVAR, not exp_1..exp_3. Hardcoding three columns read
    the FIRST PROMOTED COORDINATE as the eta exponent and ignored eta entirely, which put
    the lift at 1e19 instead of 1e-2.
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
    with open(data_dir / "R_coefficients.csv") as f:
        rdr = csv.DictReader(f)
        ecols = [c for c in rdr.fieldnames if c.startswith("exp_")]
        rcols = [c for c in rdr.fieldnames if c.endswith("_re") and c.startswith("R")]
        rows = list(rdr)
    exps = np.array([[int(r[c]) for c in ecols] for r in rows], int)
    comps = np.array([[float(r[c]) + 1j * float(r[c[:-3] + "_im"]) for c in rcols]
                      for r in rows], complex)
    return exps, comps


def load_convergence(data_dir: Path):
    """rho_conv written by the notebook's diagnostic cell; None when absent."""
    path = data_dir / "convergence.txt"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*rho_conv\s*=\s*([\d.eE+-]+)", line)
        if m:
            v = float(m.group(1))
            return v if np.isfinite(v) and v > 0 else None
    return None


def run_n_free(run_dir: Path):
    """Free-DOF count the run was computed with, from summary.txt; None if absent."""
    path = run_dir / "summary.txt"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*n_free\s*:\s*(\d+)", line)
        if m:
            return int(m.group(1))
    return None


# ── evaluation ───────────────────────────────────────────────────────────────

def eval_poly(state: np.ndarray, exps: np.ndarray, coeffs: np.ndarray):
    """Sum_m coeffs[m] * prod_i state[..., i]**exps[m, i].  state is (..., nvar)."""
    out = np.zeros(state.shape[:-1] + coeffs.shape[1:], dtype=complex)
    for m in range(exps.shape[0]):
        term = np.ones(state.shape[:-1], dtype=complex)
        for i, e in enumerate(exps[m]):
            if e:
                term = term * state[..., i] ** e
        out += term[..., None] * coeffs[m] if coeffs.ndim > 1 else term * coeffs[m]
    return out


def slaved_states(rho: float, th: np.ndarray, eta: float, omega: float,
                  rexps: np.ndarray, rcoef: np.ndarray):
    """Full state (ns, nvar) along one orbit, promoted coordinates closed by HARMONIC
    BALANCE.

    PORT of `_harmonic_R1` in src/FluidNavierStokes/rom_analysis.jl — keep the two in
    step, or the lift and TKE are evaluated with a different closure from the branch.

    Each monomial z1^a z1bar^b eta^c y_j sits at harmonic s = a - b, so on the orbit the
    drive of promoted row k splits as b_k(t) = sum_s b_ks e^{i s omega t} and its response
    solves (i s omega I - A0) y_s = b_s. The previous quasi-steady form solved A y = -b,
    which is the s = 0 case and is exact only for a drive constant on the orbit — true for
    the mean-flow modes, false for a pair driven at s = +-1, where the denominators differ
    by a factor 2.9 in magnitude plus a large phase error.

    Only the s = 0 part of A is kept, matching the Julia side: A's other harmonics come
    from monomials like z1^2 y_j, higher order in rho than the lambda_k y_k diagonal.
    """
    nvar = rexps.shape[1]
    npro = nvar - 3
    ns = th.size
    state = np.zeros((ns, nvar), dtype=complex)
    state[:, 0] = rho * np.exp(1j * th)
    state[:, 1] = np.conj(state[:, 0])
    state[:, -1] = eta
    if npro == 0:
        return state

    bs = {}                                   # harmonic s -> drive vector (npro,)
    A0 = np.zeros((npro, npro), dtype=complex)
    for m in range(rexps.shape[0]):
        e = rexps[m]
        a, b, c = int(e[0]), int(e[1]), int(e[-1])
        s = a - b
        v = rho ** (a + b) * (1.0 if c == 0 else eta ** c)
        j0 = -1                               # promoted coordinate carried, -1 = none
        for t in range(2, nvar - 1):
            if e[t] != 0:
                j0 = t - 2
                break
        for k in range(npro):
            ck = rcoef[m, 2 + k]
            if ck == 0:
                continue
            if j0 < 0:
                bs.setdefault(s, np.zeros(npro, dtype=complex))[k] += ck * v
            elif s == 0:
                A0[k, j0] += ck * v

    eye = np.eye(npro, dtype=complex)
    ys = {s: np.linalg.solve(1j * s * omega * eye - A0, bvec) for s, bvec in bs.items()}
    for j in range(npro):
        yj = np.zeros(ns, dtype=complex)
        for s, yv in ys.items():
            yj += yv[j] * np.exp(1j * s * th)
        state[:, 2 + j] = yj
    return state


def core_degree(exps: np.ndarray) -> np.ndarray:
    """Degree in the CORE coordinates (z1, z1bar, eta) — promoted ones excluded.

    This must match `truncate_dynamics` in src/FluidNavierStokes/rom_analysis.jl, which
    truncates on `e[1] + e[2] + e[nvar]`. Promoted coordinates are first order by
    construction and are not part of the order hierarchy, so counting them would drop the
    mean-flow coupling z1*y_k at low N.

    Using the TOTAL degree here instead was a real bug: the branch was traced with
    core-degree truncation while the observables were truncated on total degree, so for a
    promoted run a monomial like z1*y_k*eta^2 (core 3, total 4) was in the dynamics but
    absent from the lift and TKE. Identical for NVAR = 3, where core == total.
    """
    return exps[:, 0] + exps[:, 1] + exps[:, -1]


def truncate_gram(gram: dict, order: int) -> dict:
    sel = core_degree(gram["Avector"]) <= order
    return {"G": gram["G"][np.ix_(sel, sel)], "Avector": gram["Avector"][sel]}


def tke_from_state(state: np.ndarray, gram: dict):
    """Period-averaged fluctuation TKE, for ANY number of coordinates."""
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
    return float(ke_t.mean())


def process_branch(branch_csv: Path, gram, exps, coeffs, rexps, rcoef):
    order = int(re.search(r"rom_branch_ord(\d+)\.csv$", branch_csv.name).group(1))
    gram_N = truncate_gram(gram, order)
    # Drop base-flow (z=0) rows: the base flow has zero lift.
    keep = (core_degree(exps) <= order) & ~((exps[:, 0] == 0) & (exps[:, 1] == 0))
    exps_N, coeffs_N = exps[keep], coeffs[keep]
    # Truncate R the same way the branch was traced, so the slaved y matches the orbit.
    rkeep = core_degree(rexps) <= order
    rexps_N, rcoef_N = rexps[rkeep], rcoef[rkeep]
    rows = []
    with open(branch_csv) as f:
        for row in csv.DictReader(f):
            eta, Re = float(row["eta"]), float(row["Re"])
            rho, om, T = float(row["rho"]), float(row["omega"]), float(row["T"])
            # `fold` is written by trace_limit_cycle_branch. Older CSVs lack it; those
            # predate the fold-aware tracer and are treated as a single sheet.
            fold = int(row.get("fold", 0) or 0)
            th = 2 * np.pi * np.arange(NS) / NS
            state = slaved_states(rho, th, eta, om, rexps_N, rcoef_N)
            tke = tke_from_state(state, gram_N)
            max_lift = float(np.max(np.abs(eval_poly(state, exps_N, coeffs_N).real)))
            rows.append((order, eta, Re, rho, om, T, tke, max_lift, fold))
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
    """(Re, max_abs_lift, avg_TKE, converged) rows from the DNS sweep, if present."""
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


# ── plotting ─────────────────────────────────────────────────────────────────

COLORS = {3: "k", 5: "r", 7: "g", 9: "b"}


def make_figures(arr, fom_rows, out, tag, rho_conv, plt, matplotlib):
    """One figure per observable, in the usual bifurcation-diagram convention.

    SOLID = stable, DASHED = unstable. On this branch the limit cycle is born stable at
    the Hopf point and loses stability at the first TURNING POINT (saddle-node of the
    periodic orbit), so the line style follows the `fold` counter written by
    `trace_limit_cycle_branch` and nothing else. A branch that never turns — orders 3 and
    7 here — is stable over its whole range and is drawn solid all the way to Re_max.

    `rho_conv` deliberately does NOT affect the line style. It is a statement about the
    expansion's radius of convergence, not about the stability of the orbit; mixing the
    two drew the never-folding branches dashed from Re ~ 53, which reads as "unstable"
    and is simply wrong. It is reported in the title instead.

    The y-limit comes from the STABLE segments and the DNS points only. Past its turning
    point a branch can climb to several times the physical amplitude; letting that set
    the axis flattens every stable curve onto the x-axis.
    """
    for col, stem, ylabel in ((7, "lift_vs_Re", "max |lift|"),
                              (6, "tke_vs_Re", "period-averaged TKE")):
        plt.figure(figsize=(4.4, 4), dpi=150)
        stable_max = []

        for o in sorted(set(arr[:, 0].astype(int)))[::-1]:
            sel = arr[arr[:, 0] == o]
            stable = (sel[:, 8] == 0)          # fold index 0 = before any turning point
            n_ok = int(np.argmin(stable)) if not stable.all() else len(stable)
            if n_ok > 0:
                plt.plot(sel[:n_ok, 2], sel[:n_ok, col], ls="-", lw=2,
                         color=COLORS.get(o, "C0"), label=f"order {o}")
                stable_max.append(sel[:n_ok, col].max())
            if n_ok < len(sel):
                # Overlap by one point so the solid and dashed segments join up.
                plt.plot(sel[max(n_ok - 1, 0):, 2], sel[max(n_ok - 1, 0):, col],
                         ls="--", lw=1.5, color=COLORS.get(o, "C0"),
                         label=None if n_ok > 0 else f"order {o}")

        if fom_rows:
            fr = np.array([(r, l, t) for r, l, t, c in fom_rows if c])
            if fr.size:
                plt.plot(fr[:, 0], fr[:, 1] if col == 7 else fr[:, 2], marker="o", ms=6,
                         lw=0, markeredgewidth=2, markeredgecolor="k",
                         markerfacecolor="None", label="DNS")
                stable_max.append(max(l if col == 7 else t for _r, l, t, c in fom_rows if c))

        # Trivial (base-flow) branch: stable up to Re_c, unstable after — same convention.
        plt.plot([0, 48.9844], [0.0, 0.0], color="k", ls="-", lw=2)
        plt.plot([48.9844, 100], [0.0, 0.0], color="k", ls="--", lw=2)

        ymax = max(max(stable_max) * 1.15, 0.02 if col == 6 else 0.016) if stable_max \
            else (0.02 if col == 6 else 0.016)
        plt.ylim([-0.001, 0.02])#ymax])
        plt.xlim([RE_MIN, RE_MAX])
        plt.xlabel("Re")
        plt.ylabel(ylabel)
        title = tag if rho_conv is None else f"{tag}   (ρ_conv ≈ {rho_conv:.2f})"
        plt.title(title, fontsize=9)
        plt.legend(fontsize=7, frameon=False)
        # bbox_inches="tight" so the y-label is not clipped out of the saved PNG.
        # (tight_layout is left off deliberately — it would change the axes geometry.)
        fname = f"{stem}_{tag}.png"
        plt.savefig(out / fname, dpi=200, bbox_inches="tight")
        # plt.show() on a non-interactive backend blocks forever waiting for a window
        # that never opens, which stalls any scripted run.
        if matplotlib.is_interactive() or matplotlib.get_backend().lower() not in (
                "agg", "pdf", "ps", "svg", "template"):
            plt.show()
        else:
            plt.close(plt.gcf())
        print(f"wrote {out / fname}")


def main():
    run_dirs = sorted(p for p in (HERE / "results").glob("Re*_ord*")
                      if any((p / "data").glob("rom_branch_ord*.csv")))
    if not run_dirs:
        sys.exit("no rom_branch_ord*.csv found — run the notebook first")

    # The order comparison is only meaningful across truncations of ONE run: orders
    # 3/5/7/9 of a single order-9 solve share a mesh, so their differences are TRUNCATION
    # error. A FAST run (14 436 DOFs) beside a FULL one (57 860) would be drawn as if it
    # were another truncation when it is a different DISCRETISATION.
    sizes = {d: run_n_free(d) for d in run_dirs}
    known = {n for n in sizes.values() if n is not None}
    if len(known) > 1:
        keep = max(known)
        dropped = [d for d in run_dirs if sizes[d] is not None and sizes[d] != keep]
        run_dirs = [d for d in run_dirs if sizes[d] is None or sizes[d] == keep]
        print(f"mesh mismatch: keeping the {keep}-DOF run(s); dropped "
              + ", ".join(f"{d.name} ({sizes[d]})" for d in dropped))

    # The DNS reference comes from direct time integration, so it is independent of any
    # DPIM run and often outlives it. Look it up across ALL result dirs, not just those
    # holding ROM branches — tying the two together silently dropped it from these plots.
    fom_dirs = sorted(p for p in (HERE / "results").glob("Re*_ord*")
                      if (p / "data" / "fom_reference.csv").exists())
    fom_rows = [r for d in fom_dirs for r in load_fom_reference(d)]
    print(f"DNS reference: {len(fom_rows)} points"
          + (f" from {', '.join(d.name for d in fom_dirs)}" if fom_rows else " (none)"))

    out = HERE / "results" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    hdr = "order,eta,Re,rho,omega,T,avg_TKE,max_abs_lift,fold"

    import matplotlib
    import matplotlib.pyplot as plt

    for run_dir in run_dirs:
        rows = process_run(run_dir)
        if not rows:
            continue
        arr = np.array(rows)
        tag = run_dir.name
        rho_conv = load_convergence(run_dir / "data")
        np.savetxt(out / f"comparison_{tag}.csv", arr, delimiter=",",
                   header=hdr, comments="", fmt="%.10e")
        orders = sorted({int(r[0]) for r in rows})
        print(f"{tag}: orders {orders}"
              + (f", rho_conv {rho_conv:.2f}" if rho_conv else ", no rho_conv recorded"))
        make_figures(arr, fom_rows, out, tag, rho_conv, plt, matplotlib)


if __name__ == "__main__":
    main()
