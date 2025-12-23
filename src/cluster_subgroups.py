# cluster_subgroups.py

import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from src.methods import hist_kl, gaussian_kl, get_box_rule_from_mask


def _generate_cluster_candidates(X, Y, cfg):
    """
    Generate a large set of candidate subgroups via KMeans clustering
    with different seeds and different numbers of clusters (K).

    Returns:
        candidates: list of dicts with keys
            - indices: np.array of row indices
            - ns: size of subgroup
            - leaf_Y: standardized Y values in subgroup
            - base_score: size^alpha * KL(exceptionality)
            - size_term: (ns / n_total) ** alpha
        Ys: standardized Y (global)
        global_Y: alias of Ys
        n_total: number of samples
    """
    X = np.asarray(X)
    Y = np.asarray(Y).ravel()
    n, d = X.shape

    # Standardize 
    # scaler_x = StandardScaler()
    # Xs = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    Ys = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()

    global_Y = Ys
    n_total = len(Ys)

    alpha = getattr(cfg, "alpha", 0.3)
    use_histogram_kl = getattr(cfg, "use_histogram_kl", True)
    k_values = getattr(cfg, "cluster_k_values", [2, 3, 4])
    n_seeds = getattr(cfg, "n_cluster_seeds", 100)
    base_seed = getattr(cfg, "cluster_seed", 0)
    min_k = getattr(cfg, "y_min_clusters", 2)
    max_k = getattr(cfg, "y_max_clusters", 50)
    min_abs = getattr(cfg, "min_cluster_size", 20)
    min_frac = getattr(cfg, "min_cluster_frac", 0.02)
    min_size = max(min_abs, int(min_frac * n_total))
    Z = linkage(Ys.reshape(-1, 1), method=getattr(cfg, "y_linkage", "ward"))
    candidates = []

    for k in range(min_k, max_k + 1):
        # cluster_labels in {1, 2, ..., k}
        cluster_labels = fcluster(Z, t=k, criterion="maxclust")
        unique_labels = np.unique(cluster_labels)

        for lab in unique_labels:
            mask = (cluster_labels == lab)
            ns = int(mask.sum())
            if ns < min_size:
                continue

            indices = np.where(mask)[0]
            leaf_Y = Ys[indices]

            # Exceptionality: KL(P(Y | cluster) || P(Y)) * size^alpha
            if use_histogram_kl:
                kl_exc = hist_kl(leaf_Y, global_Y)
            else:
                mu_s, std_s = float(leaf_Y.mean()), float(leaf_Y.std())
                mu_g, std_g = float(global_Y.mean()), float(global_Y.std()) + 1e-12
                kl_exc = gaussian_kl(mu_s, std_s, mu_g, std_g)

            size_term = (ns / n_total) ** alpha
            base_score = size_term * kl_exc

            candidates.append(
                dict(
                    indices=indices,
                    ns=ns,
                    leaf_Y=leaf_Y,
                    base_score=base_score,
                    size_term=size_term,
                )
            )

    return candidates, Ys, global_Y, n_total


def discover_subgroups_by_clustering(X, Y, cfg, feature_names=None, n_subgroups=None):
    """
    High-level API for Module 1.

    Steps:
        1. Generate many clustering candidates with different seeds & K.
        2. Score each candidate with size-corrected KL.
        3. Greedy selection with diversity regularizer (same γ and λ as SYFLOW). :contentReference[oaicite:8]{index=8}
        4. For each selected subgroup, create:
           - boolean mask over rows
           - human-readable box rule in original feature space
           - score of the objective

    Returns:
        masks: list of (n,) boolean arrays
        rules: list of strings
        scores: list of floats
    """
    X = np.asarray(X)
    Y = np.asarray(Y).ravel()
    n, d = X.shape

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(d)]

    if n_subgroups is None:
        n_subgroups = getattr(cfg, "n_subgroups", 5)

    candidates, Ys, global_Y, n_total = _generate_cluster_candidates(X, Y, cfg)
    if len(candidates) == 0:
        return [], [], []

    selected_masks = []
    selected_rules = []
    selected_scores = []
    priors_Ys = []

    use_histogram_kl = getattr(cfg, "use_histogram_kl", True)
    alpha = getattr(cfg, "alpha", 0.3)
    lambd = getattr(cfg, "lambd", 2.0)

    # Greedy selection with diversity, mirroring run_tree_syflow_corrected 
    for k in range(n_subgroups):
        best = None
        best_val = -np.inf

        for c in candidates:
            if c.get("selected", False):
                continue

            # Diversity term: average KL to already selected subgroups
            if len(priors_Ys) == 0:
                diversity = 0.0
            else:
                s_div = 0.0
                for priorY in priors_Ys:
                    if use_histogram_kl:
                        s_div += hist_kl(c["leaf_Y"], priorY)
                    else:
                        mu_s, std_s = float(c["leaf_Y"].mean()), float(c["leaf_Y"].std())
                        mu_p, std_p = float(priorY.mean()), float(priorY.std())
                        s_div += gaussian_kl(mu_s, std_s, mu_p, std_p)
                diversity = s_div / len(priors_Ys)

            weighted_div = c["size_term"] * diversity
            total_score = c["base_score"] + lambd * weighted_div

            if total_score > best_val:
                best_val = total_score
                best = c

        if best is None:
            break

        best["selected"] = True
        indices = best["indices"]
        mask = np.isin(np.arange(n), indices)

        rule_text, _ = get_box_rule_from_mask(
            X, mask, feature_names=feature_names, ignore_if_full=True
        )

        selected_masks.append(mask)
        selected_rules.append(rule_text)
        selected_scores.append(best_val)

        priors_Ys.append(best["leaf_Y"])

    return selected_masks, selected_rules, selected_scores


def build_subgroup_column(masks, n_samples, fill_value=-1):
    """
    Turn list of boolean masks into a single subgroup_id array of length n_samples.
    A row gets the index of the FIRST subgroup that covers it, otherwise 'fill_value'.
    """
    subgroup_id = np.full(shape=(n_samples,), fill_value=fill_value, dtype=int)
    for k, mask in enumerate(masks):
        subgroup_id[mask] = k
    return subgroup_id
