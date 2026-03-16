import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from SD_longitudinal.src.forest import extract_rules
from interpretable_sd.utils.metrics import classification_metrics
from interpretable_sd.utils.text_rules import render_natural_language_rules
from interpretable_sd.utils.tree_display import (
    compute_leaf_target_stats,
    save_constant_box_tree_plot,
)


def _fit_pruned_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    max_depth: int,
    min_samples_leaf: int,
) -> Tuple[DecisionTreeClassifier, Dict]:
    try:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
        )
    except ValueError:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
        )

    base_tree = DecisionTreeClassifier(
        random_state=seed,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    path = base_tree.cost_complexity_pruning_path(X_fit, y_fit)
    alphas = np.unique(np.round(path.ccp_alphas, 8))
    if alphas.size == 0:
        alphas = np.array([0.0], dtype=float)

    best = None
    best_info = None
    for alpha in alphas:
        clf = DecisionTreeClassifier(
            random_state=seed,
            ccp_alpha=float(alpha),
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        clf.fit(X_fit, y_fit)
        pred = clf.predict(X_val)
        acc = float(accuracy_score(y_val, pred))
        n_nodes = int(clf.tree_.node_count)
        info = {"val_accuracy": acc, "ccp_alpha": float(alpha), "n_nodes": n_nodes}
        if best is None:
            best, best_info = clf, info
            continue
        if acc > best_info["val_accuracy"] or (
            np.isclose(acc, best_info["val_accuracy"]) and n_nodes < best_info["n_nodes"]
        ):
            best, best_info = clf, info

    final_tree = DecisionTreeClassifier(
        random_state=seed,
        ccp_alpha=float(best_info["ccp_alpha"]),
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    final_tree.fit(X_train, y_train)
    return final_tree, best_info


def run_tree_paper_style_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    target_train: np.ndarray,
    meta_train,
    out_dir: Path,
    cfg: Dict,
) -> Dict:
    print(
        "[Baseline][tree_paper_style] start "
        f"train={len(y_train)} test={len(y_test)} features={len(feature_names)} "
        f"max_depth={cfg.get('max_depth', 5)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    max_depth = int(cfg.get("max_depth", 5))
    min_samples_leaf = int(cfg.get("min_samples_leaf", 15))

    tree, selection_info = _fit_pruned_tree(
        X_train,
        y_train,
        seed=seed,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    train_prob = tree.predict_proba(X_train)[:, 1] if len(np.unique(y_train)) > 1 else None
    test_prob = tree.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else None

    metrics = {
        "train": classification_metrics(y_train, train_pred, train_prob),
        "test": classification_metrics(y_test, test_pred, test_prob),
        "selection": selection_info,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    leaf_stats = compute_leaf_target_stats(
        tree,
        X_train,
        target_train,
        span_start_years=None if meta_train is None or "segment_start_year" not in meta_train.columns else meta_train["segment_start_year"].to_numpy(),
        span_end_years=None if meta_train is None or "segment_end_year" not in meta_train.columns else meta_train["segment_end_year"].to_numpy(),
        span_lengths=None if meta_train is None or "segment_length" not in meta_train.columns else meta_train["segment_length"].to_numpy(),
    )
    rules = extract_rules(tree, feature_names=feature_names, question_map=None, leaf_stats=leaf_stats)
    (out_dir / "rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    nl = render_natural_language_rules(
        "tree_paper_style",
        rules,
        "Tree-paper-style baseline: single pruned decision tree chosen by validation accuracy and compactness over segment-level samples.",
    )
    (out_dir / "rules_natural_language.txt").write_text(nl, encoding="utf-8")

    save_constant_box_tree_plot(
        tree,
        feature_names,
        out_dir / "tree",
        title=None,
        leaf_stats=leaf_stats,
        positive_class=1,
        positive_label="interesting",
        negative_label="not interesting",
    )

    print(
        "[Baseline][tree_paper_style] done | "
        f"train_acc={metrics['train'].get('accuracy', 0.0):.4f} "
        f"test_acc={metrics['test'].get('accuracy', 0.0):.4f} "
        f"n_rules={len(rules)} ccp_alpha={selection_info.get('ccp_alpha')}"
    )

    return {
        "metrics": metrics,
        "n_rules": int(len(rules)),
    }
