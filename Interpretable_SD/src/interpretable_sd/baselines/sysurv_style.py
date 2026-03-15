import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from SD_longitudinal.src.forest import extract_rules
from interpretable_sd.utils.metrics import classification_metrics
from interpretable_sd.utils.text_rules import render_natural_language_rules
from interpretable_sd.utils.tree_display import (
    compute_leaf_target_stats,
    save_constant_box_tree_plot,
)


def _save_namer_tree(
    tree: DecisionTreeClassifier,
    X_train: np.ndarray,
    y_train_like: np.ndarray,
    feature_names: List[str],
    target_train: np.ndarray,
    meta_train,
    out_dir: Path,
) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = float((tree.predict(X_train) == y_train_like).mean())
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

    save_constant_box_tree_plot(
        tree,
        feature_names,
        out_dir / "naming_tree",
        title=f"Sysurv-style naming tree | train acc={acc:.3f}",
        leaf_stats=leaf_stats,
        positive_class=1,
        positive_label="interesting",
        negative_label="not interesting",
    )
    return rules


def run_sysurv_style_baseline(
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
        "[Baseline][sysurv_style] start "
        f"train={len(y_train)} test={len(y_test)} features={len(feature_names)} "
        f"n_estimators={cfg.get('n_estimators', 200)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_train = np.unique(y_train)
    if unique_train.size < 2:
        constant_class = int(unique_train[0]) if unique_train.size == 1 else 0
        train_pred = np.full(len(y_train), constant_class, dtype=int)
        test_pred = np.full(len(y_test), constant_class, dtype=int)
        metrics = {
            "train": classification_metrics(y_train, train_pred, None),
            "test": classification_metrics(y_test, test_pred, None),
            "warning": (
                "Skipped sysurv-style scorer fit because train labels contain a single class. "
                "Returned constant-class baseline metrics."
            ),
            "constant_class": constant_class,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (out_dir / "rules_natural_language.txt").write_text(
            "Method: sysurv_style\nSingle-class train labels; no scorer/namer rules generated.",
            encoding="utf-8",
        )
        print(
            "[Baseline][sysurv_style] skipped scorer fit due to single-class train labels | "
            f"class={constant_class}"
        )
        return {
            "metrics": metrics,
            "n_rules": 0,
        }

    scorer = GradientBoostingClassifier(
        n_estimators=int(cfg.get("n_estimators", 200)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_depth=int(cfg.get("max_depth", 3)),
        random_state=int(cfg.get("seed", 42)),
    )
    scorer.fit(X_train, y_train)

    train_prob = scorer.predict_proba(X_train)[:, 1]
    test_prob = scorer.predict_proba(X_test)[:, 1]
    threshold_quantile = float(cfg.get("risk_quantile", 0.8))
    threshold = float(np.quantile(train_prob, threshold_quantile))
    train_pred = (train_prob >= threshold).astype(int)
    test_pred = (test_prob >= threshold).astype(int)
    threshold_mode = "quantile"

    # If the configured quantile produces a constant scorer target, fall back to
    # matching the observed train prevalence so the naming tree remains meaningful.
    if np.unique(train_pred).size < 2 and len(np.unique(y_train)) >= 2:
        prevalence_quantile = float(np.clip(1.0 - float(np.mean(y_train)), 0.0, 0.999))
        threshold_quantile = prevalence_quantile
        threshold = float(np.quantile(train_prob, threshold_quantile))
        train_pred = (train_prob >= threshold).astype(int)
        test_pred = (test_prob >= threshold).astype(int)
        threshold_mode = "match_train_prevalence"
        print(
            "[Baseline][sysurv_style] threshold fallback | "
            f"mode={threshold_mode} q={threshold_quantile:.4f}"
        )

    # Naming stage: learn an interpretable tree over the scorer decision.
    namer = DecisionTreeClassifier(
        max_depth=int(cfg.get("namer_max_depth", 3)),
        min_samples_leaf=int(cfg.get("namer_min_samples_leaf", 15)),
        random_state=int(cfg.get("seed", 42)),
    )
    namer.fit(X_train, train_pred)
    rules = _save_namer_tree(
        namer,
        X_train,
        train_pred,
        feature_names,
        target_train,
        meta_train,
        out_dir / "naming",
    )

    context_note = (
        "Sysurv-style baseline: a non-linear risk scorer first detects high-risk segments; "
        "a shallow naming tree then explains when those interesting spans occur."
    )
    nl = render_natural_language_rules("sysurv_style", rules, context_note)
    (out_dir / "rules_natural_language.txt").write_text(nl, encoding="utf-8")

    metrics = {
        "train": classification_metrics(y_train, train_pred, train_prob),
        "test": classification_metrics(y_test, test_pred, test_prob),
        "threshold_mode": threshold_mode,
        "risk_threshold_quantile": threshold_quantile,
        "risk_threshold": threshold,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        "[Baseline][sysurv_style] done | "
        f"train_acc={metrics['train'].get('accuracy', 0.0):.4f} "
        f"test_acc={metrics['test'].get('accuracy', 0.0):.4f} "
        f"n_rules={len(rules)}"
    )

    return {
        "metrics": metrics,
        "n_rules": int(len(rules)),
    }
