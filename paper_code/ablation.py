"""Ablation for the no-split, descriptive use of SEAL.
(1) Out-of-sample coherence: fit SEAL on a random 70% of entities, apply the
    learned tree to the held-out 30%, and compare the within-group ratio rho on
    the held-out entities to the in-sample one. Close values mean the rules
    describe unseen entities about as well, so the in-sample fit is not inflating
    coherence.
(2) Seed stability: rerun the randomized search under different seeds and report
    the mean pairwise adjusted Rand index between the resulting partitions.
Four public panels; reduced search budget, which is enough for a robustness check."""
import sys, time, warnings
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings("ignore")
sys.path.insert(0, "paper_code")
from seal import run_seal, pyramid_features
from baselines import within_ratio
from run_panels import load, PANELS

def dist_gm(traj):
    rep = pyramid_features(traj.to_numpy().astype(float))
    d = squareform(pdist(rep))
    return d, float(d[np.triu_indices(len(traj), k=1)].mean())

BUDGET = 4000
def params(seed):
    return dict(n_candidates=BUDGET, k_range=[2, 8], tree_depths=[1, 2, 3, 4],
               leaf_fractions=[0.02, 0.3], lambda_complexity=0.005,
               diversity_pool=200, ari_max=0.8, seed=seed)

N_SPLITS, N_SEEDS = 3, 3
t0 = time.time()
print(f"panel      | rho_in  rho_out | ARI_seed")
print("-" * 44)
for ds in PANELS:
    traj, X = load(ds)
    N = len(traj)
    # (1) out-of-sample
    ins, outs = [], []
    for s in range(N_SPLITS):
        rng = np.random.default_rng(100 + s)
        perm = rng.permutation(N)
        cut = int(0.7 * N)
        tr, te = np.sort(perm[:cut]), np.sort(perm[cut:])
        traj_tr, X_tr = traj.iloc[tr], X.iloc[tr]
        traj_te, X_te = traj.iloc[te], X.iloc[te]
        res = run_seal(traj_tr, X_tr, params(0))
        tree = res["best"]["tree"]
        d_tr, gm_tr = dist_gm(traj_tr)
        d_te, gm_te = dist_gm(traj_te)
        lab_tr = res["best"]["labels"]
        lab_te = tree.predict(X_te.to_numpy())
        ins.append(within_ratio(lab_tr, d_tr, gm_tr))
        outs.append(within_ratio(lab_te, d_te, gm_te))
    # (2) seed stability on the full panel
    labs = []
    for seed in range(N_SEEDS):
        res = run_seal(traj, X, params(seed))
        labs.append(np.asarray(res["best"]["labels"]))
    aris = [adjusted_rand_score(labs[i], labs[j])
            for i in range(N_SEEDS) for j in range(i + 1, N_SEEDS)]
    print(f"{ds:10s} | {np.mean(ins):6.3f}  {np.mean(outs):6.3f} | {np.mean(aris):6.3f}")
print("-" * 44)
print("elapsed %.0fs" % (time.time() - t0))
