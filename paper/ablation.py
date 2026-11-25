"""Feature-engineering ablation.

Question: do the engineered temporal descriptors buy anything over a naive
point-in-time snapshot of the same covariates?

For each dataset we run the identical SEAL search twice, changing only the
descriptor matrix:
  * snapshot: each numeric covariate contributes only its last observed value
    (the recent level), plus one-hot categoricals. This is the usual
    subgroup-discovery vocabulary.
  * temporal: the full engineered set (trend, recent level, and the worst
    sustained shortfall as severity, length, recency), plus one-hot categoricals.

We compare the best within-group distance ratio (lower is tighter) and the rule
complexity at that best cost. Lower ratio at equal or lower complexity means the
temporal features let a simple rule carve out tighter trajectory groups.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cmie_sd_run"))
sys.path.insert(0, str(ROOT / "seal_improved_runs"))
from seal_method import run_seal                         # noqa: E402
import run_improved_seal as R                            # noqa: E402

PARAMS = {**R.SEAL_PARAMS, "n_candidates": 5000}
DATASETS = ["co2", "worldbank", "covid", "epa"]


def snapshot_X(prep, target_col, entities):
    """Point-in-time baseline: last observed value of each numeric covariate, plus
    one-hot categoricals. No trend, no episode features."""
    prep = prep.copy()
    prep["time"] = R._numeric_time(prep["time"])
    drop = {"entity", "time", target_col}
    cov = [c for c in prep.columns if c not in drop]
    num = [c for c in cov if pd.api.types.is_numeric_dtype(prep[c])]
    cat = [c for c in cov if c not in num]
    prep = prep.sort_values("time")
    last = prep.groupby("entity")[num].last().reindex(entities)
    parts = [last]
    for c in cat:
        v = prep.groupby("entity")[c].first().reindex(entities).astype("string")
        if v.nunique(dropna=True) <= 40:
            parts.append(pd.get_dummies(v, prefix=c, prefix_sep="_", dtype=float))
    X = pd.concat(parts, axis=1).reindex(entities).replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def best_of(traj, X):
    res = run_seal(traj.loc[X.index].sort_index(), X.sort_index(), PARAMS)
    b = res["best"]
    return dict(ratio=float(b["distance_ratio"]), complexity=int(b["complexity"]),
                n_groups=int(b["n_groups"]), n_descriptors=int(X.shape[1]))


def main():
    out = {}
    for ds in DATASETS:
        prep_rel, target_col, _, _ = R.DATASETS[ds]
        prep = pd.read_csv(ROOT / "seal_repo" / "data" / prep_rel)
        traj = R.build_traj(ROOT / "seal_repo" / "results" / ds / "part4_target_trajectories.csv")
        ents = traj.index.to_numpy()
        Xs = snapshot_X(prep, target_col, ents)
        Xt = R.build_X(prep, target_col, ents)
        common = traj.index.intersection(Xs.index).intersection(Xt.index)
        traj = traj.loc[common]
        snap = best_of(traj, Xs.loc[common])
        temp = best_of(traj, Xt.loc[common])
        out[ds] = {"snapshot": snap, "temporal": temp}
        print(f"[{ds}] snapshot ratio={snap['ratio']:.3f} (cx={snap['complexity']}, "
              f"d={snap['n_descriptors']})  ->  temporal ratio={temp['ratio']:.3f} "
              f"(cx={temp['complexity']}, d={temp['n_descriptors']})  "
              f"improvement={snap['ratio']-temp['ratio']:+.3f}")
    (Path(__file__).parent / "ablation_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote ablation_results.json")


if __name__ == "__main__":
    main()
