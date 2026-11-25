"""Quantitative validation of the discovered subgroups.

Three analyses, all on the already-discovered segmentations:
  1. Coherence permutation test. Is the within-group trajectory distance ratio
     far below what random groupings of the same sizes achieve? A large z and a
     tiny p say the rule-induced partition captures real trajectory structure.
  2. Outcome-stratification test. Do the groups differ on the target itself
     (Kruskal-Wallis on each entity's mean target level)? The rules are built on
     descriptors, not the target, so a significant difference says the
     descriptors genuinely stratify the outcome.
  3. Effect sizes. How far does each subgroup's mean target sit from the
     population mean, in population standard deviations?

Results are written to paper/analysis_results.json and printed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "seal_improved_runs"
PUBLIC = ["co2", "covid", "worldbank", "epa"]

# fixed permutation seed stream (Math.random is banned in workflows, plain numpy here is fine)
RNG = np.random.default_rng(0)


def load(ds):
    traj = pd.read_parquet(RUNS / ds / "trajectories.parquet")
    mem = pd.read_csv(RUNS / ds / "membership.csv").set_index("entity")["subgroup"]
    lab = mem.reindex(traj.index).to_numpy()
    return traj, lab


def within_ratio(dist, labels):
    n = len(labels)
    gm = dist[np.triu_indices(n, k=1)].mean()
    blocks = []
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if len(idx) > 1:
            blocks.append(dist[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)])
    return np.concatenate(blocks).mean() / gm


def coherence_test(traj, lab, n_perm=2000):
    cum = np.cumsum(traj.to_numpy(), axis=1)
    dist = squareform(pdist(cum))
    r_obs = within_ratio(dist, lab)
    null = np.empty(n_perm)
    lab_c = lab.copy()
    for i in range(n_perm):
        RNG.shuffle(lab_c)
        null[i] = within_ratio(dist, lab_c)
    z = (null.mean() - r_obs) / null.std()
    p = (np.sum(null <= r_obs) + 1) / (n_perm + 1)
    return dict(ratio=float(r_obs), null_mean=float(null.mean()),
                null_std=float(null.std()), z=float(z), p=float(p))


def outcome_test(traj, lab):
    level = traj.mean(axis=1).to_numpy()  # each entity's mean target level
    groups = [level[lab == g] for g in np.unique(lab)]
    H, p = kruskal(*groups)
    return dict(H=float(H), p=float(p), n_groups=len(groups))


def effect_sizes(traj, lab):
    level = traj.mean(axis=1).to_numpy()
    mu, sd = level.mean(), level.std()
    out = {}
    for g in np.unique(lab):
        d = (level[lab == g].mean() - mu) / sd
        out[("rest" if g == 0 else f"G{int(g)}")] = dict(
            n=int((lab == g).sum()), cohen_d=float(d),
            mean_level=float(level[lab == g].mean()))
    return out


def main():
    results = {}
    for ds in PUBLIC:
        traj, lab = load(ds)
        results[ds] = {
            "coherence": coherence_test(traj, lab),
            "outcome": outcome_test(traj, lab),
            "effects": effect_sizes(traj, lab),
        }
        c, o = results[ds]["coherence"], results[ds]["outcome"]
        print(f"[{ds}] ratio={c['ratio']:.3f} vs null {c['null_mean']:.3f}"
              f" (z={c['z']:.1f}, p<{max(c['p'],1e-4):.4f}) | "
              f"outcome Kruskal H={o['H']:.1f} p<{max(o['p'],1e-12):.1e}")
    (Path(__file__).parent / "analysis_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote analysis_results.json")


if __name__ == "__main__":
    main()
