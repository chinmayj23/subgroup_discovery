# src/forest_explainer.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text


def _forest_complexity(model, X_val, y_val):
    """
    Approximate 'number of questions' as the average shortest path length
    across trees for validation samples with y_val == 1.
    """
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val).ravel()
    pos_idx = np.where(y_val == 1)[0]
    if pos_idx.size == 0:
        return np.inf

    path_lengths = []

    for idx in pos_idx:
        x_row = X_val[idx : idx + 1]
        tree_lengths = []
        for est in model.estimators_:
            node_indicator = est.decision_path(x_row)
            length = node_indicator.indptr[1] - node_indicator.indptr[0]
            tree_lengths.append(length)
        path_lengths.append(min(tree_lengths))

    return float(np.mean(path_lengths))


def train_forest_for_subgroup(X, subgroup_mask, cfg):
    """
    Train many candidate forests to describe a single subgroup (one-vs-rest),
    and pick the best according to:
        1. high validation accuracy
        2. low average path length ('questions') on positives.
    """
    X = np.asarray(X)
    subgroup_mask = np.asarray(subgroup_mask).astype(bool)
    y = subgroup_mask.astype(int)

    if y.sum() < getattr(cfg, "rf_min_positive", 20):
        return None, []

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=getattr(cfg, "rf_test_size", 0.3),
        stratify=y,
        random_state=getattr(cfg, "rf_base_seed", 123),
    )

    n_seeds = getattr(cfg, "rf_n_seeds", 50)
    n_estimators = getattr(cfg, "rf_n_estimators", 20)
    max_depth = getattr(cfg, "rf_max_depth", 4)
    min_samples_leaf = getattr(cfg, "rf_min_samples_leaf", 20)
    max_features = getattr(cfg, "rf_max_features", "sqrt")
    base_seed = getattr(cfg, "rf_base_seed", 123)

    candidates = []

    for s in range(n_seeds):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=base_seed + s,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        complexity = _forest_complexity(model, X_val, y_val)

        candidates.append(
            dict(
                model=model,
                acc=float(acc),
                complexity=float(complexity),
            )
        )

    if not candidates:
        return None, []

    accs = np.array([c["acc"] for c in candidates])
    q = getattr(cfg, "rf_accuracy_quantile", 0.8)
    thr = np.quantile(accs, q)
    filtered = [c for c in candidates if c["acc"] >= thr]
    if not filtered:
        filtered = candidates

    best = min(filtered, key=lambda c: (c["complexity"], -c["acc"]))
    return best["model"], candidates


def forest_to_rules(model, feature_names, max_trees=2):
    """
    Convert the first few trees in the forest into readable text rules.
    """
    rules = []
    for i, est in enumerate(model.estimators_[:max_trees]):
        text = export_text(est, feature_names=feature_names)
        rules.append(f"Tree {i}:\n{text}")
    return rules


def evaluate_forest(model, X, subgroup_mask):
    """
    Convenience: compute accuracy and complexity on all samples.
    """
    X = np.asarray(X)
    y = np.asarray(subgroup_mask).astype(int)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    comp = _forest_complexity(model, X, y)
    return acc, comp
