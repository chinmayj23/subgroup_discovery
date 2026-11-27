"""Sensitivity to lambda, and SEAL versus the predictive clustering tree at matched
rule complexity, on the four public panels.

One randomized SEAL search per panel records every candidate's (rho, complexity),
so the SEAL frontier (the lowest rho reachable at each rule length) and the whole
lambda path come from a single run. The predictive clustering tree frontier is
traced over its depth and leaf-size grid. We report, per panel, the lowest cost at
lambda = 0.005, the coherence of each method when held to the other's rule length,
and the smallest lambda at which SEAL reaches the lower cost. The figure is written
to figures/frontier.pdf. CMIE is proprietary and server-side; it is not included.

Run: python sensitivity.py [n_candidates]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from seal import build_distance, search, diversity_ranking
from run_panels import load, PARAMS
from baselines import within_ratio, pct_complexity, cluster_then_explain, emm

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
N_CAND = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
PANELS_RUN = ["worldbank", "co2", "covid", "epa"]
TITLE = {"worldbank": "World Bank", "co2": "CO2", "covid": "COVID-19", "epa": "EPA"}
LAM = 0.005


def envelope(points):
    pts = [(c, r) for c, r in points if np.isfinite(r)]
    return {b: min(r for c, r in pts if c <= b) for b in range(max(c for c, _ in pts) + 1)
            if any(c <= b for c, _ in pts)}


def cost_opt(points, lam):
    return min(((c, r) for c, r in points if np.isfinite(r)), key=lambda cr: cr[1] + lam * cr[0])


def pct_points(X, rep, dist, gm):
    Xv = X.to_numpy(); N = len(X); pts = []
    for depth in (1, 2, 3, 4, 5, 6):
        for frac in (0.02, 0.05, 0.10):
            leaf = max(int(frac * N), 5)
            reg = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(Xv, rep)
            lab = reg.apply(Xv)
            pts.append((pct_complexity(reg, list(X.columns)), float(within_ratio(lab, dist, gm))))
    return pts


def crossover(seal_pts, pct_pts):
    for i in range(0, 201):
        lam = 0.0005 * i
        cs = cost_opt(seal_pts, lam); cp = cost_opt(pct_pts, lam)
        if cs[1] + lam * cs[0] <= cp[1] + lam * cp[0] + 1e-9:
            return round(lam, 4)
    return None


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    print(f"{'panel':10s} | {'SEAL cost(rho,c)':20s} | {'PCT@SEAL_c':10s} | {'SEAL@PCT_c':10s} | lambda*")
    print("-" * 78)
    for ax, ds in zip(axes.ravel(), PANELS_RUN):
        traj, X = load(ds)
        rep, dist, gm = build_distance(traj)
        records, best, trees, all_labels = search(
            rep, dist, gm, X, n_candidates=N_CAND, k_range=PARAMS["k_range"],
            tree_depths=PARAMS["tree_depths"], leaf_fractions=PARAMS["leaf_fractions"],
            lam=LAM, seed=PARAMS["seed"])
        seal_pts = [(r["complexity"], r["distance_ratio"]) for r in records]
        pcts = pct_points(X, rep, dist, gm)
        c_s, rho_s = cost_opt(seal_pts, LAM)
        c_p, rho_p = cost_opt(pcts, LAM)
        pct_at_seal = min([r for c, r in pcts if c <= c_s], default=float("nan"))
        seal_at_pct = min([r for c, r in seal_pts if c <= c_p and np.isfinite(r)], default=float("nan"))
        lam_star = crossover(seal_pts, pcts)
        print(f"{ds:10s} | {rho_s + LAM * c_s:.3f} ({rho_s:.3f},{c_s:3d})   | "
              f"{pct_at_seal:10.3f} | {seal_at_pct:10.3f} | {lam_star}")

        senv, penv = envelope(seal_pts), envelope(pcts)
        rho_cte, c_cte, _ = cluster_then_explain(X, dist, gm, rep)
        rho_emm, c_emm, _ = emm(X, traj, dist, gm, K=best["n_groups"] - 1)
        xs = sorted(senv); ax.plot(xs, [senv[c] for c in xs], "-", color="#0072B2", lw=2, label="SEAL")
        xp = sorted(penv); ax.plot(xp, [penv[c] for c in xp], "--o", color="#D55E00", ms=3, lw=1.3, label="PCT")
        ax.plot(c_cte, rho_cte, "s", color="#009E73", ms=5, label="Cluster-then-Explain")
        ax.plot(c_emm, rho_emm, "^", color="#CC79A7", ms=5, label="EMM")
        ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
        ax.set_title(TITLE[ds], fontsize=10); ax.set_xlabel("propositions c", fontsize=9)
        ax.set_ylabel(r"coherence $\rho$", fontsize=9); ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes.ravel()[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "frontier.pdf")
    print(f"\n-> figures/frontier.pdf  (n_candidates={N_CAND})")


if __name__ == "__main__":
    main()
