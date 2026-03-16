import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from SD_longitudinal.src.common import LABEL_COLUMN
from SD_longitudinal.src.plotting import save_forest_accuracy_plot
from SD_longitudinal.src.forest import (
    extract_rules,
    forest_predict,
    rebuild_forest,
    run_forest_search,
    tree_to_dict,
)
from interpretable_sd.utils.metrics import classification_metrics
from interpretable_sd.utils.text_rules import render_natural_language_rules
from interpretable_sd.utils.tree_display import (
    compute_leaf_target_stats,
    save_constant_box_tree_plot,
)


def _forest_score_matrix(trees: List[DecisionTreeClassifier], X: np.ndarray) -> np.ndarray:
    if not trees:
        return np.zeros(len(X), dtype=float)
    pred_matrix = np.vstack([t.predict(X).astype(float) for t in trees])
    return pred_matrix.mean(axis=0)


def _save_tree_bundle(
    trees: List[DecisionTreeClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    target_train: np.ndarray,
    meta_train: Optional[pd.DataFrame],
    out_dir: Path,
) -> List[dict]:
    all_positive_rules: List[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    for idx, tree in enumerate(trees):
        acc = float((tree.predict(X_train) == y_train).mean())
        leaf_stats = compute_leaf_target_stats(
            tree,
            X_train,
            target_train,
            span_start_years=None if meta_train is None or "segment_start_year" not in meta_train.columns else meta_train["segment_start_year"].to_numpy(),
            span_end_years=None if meta_train is None or "segment_end_year" not in meta_train.columns else meta_train["segment_end_year"].to_numpy(),
            span_lengths=None if meta_train is None or "segment_length" not in meta_train.columns else meta_train["segment_length"].to_numpy(),
        )
        tree_payload = tree_to_dict(
            tree,
            feature_names=feature_names,
            question_map=None,
            leaf_stats=leaf_stats,
            global_stats=None,
        )
        tree_json = out_dir / f"tree_{idx:02d}_acc_{acc:.4f}.json"
        tree_json.write_text(json.dumps(tree_payload, indent=2), encoding="utf-8")

        pos_rules = extract_rules(
            tree,
            feature_names=feature_names,
            question_map=None,
            leaf_stats=leaf_stats,
        )
        all_positive_rules.extend(pos_rules)
        (out_dir / f"tree_{idx:02d}_rules.json").write_text(
            json.dumps(pos_rules, indent=2),
            encoding="utf-8",
        )

        save_constant_box_tree_plot(
            tree,
            feature_names,
            out_dir / f"tree_{idx:02d}_acc_{acc:.4f}",
            title=None,
            leaf_stats=leaf_stats,
            positive_class=1,
            positive_label="interesting",
            negative_label="not interesting",
        )

    return all_positive_rules


def run_rf_rule_module(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    target_train: np.ndarray,
    meta_train: Optional[pd.DataFrame],
    out_dir: Path,
    cfg: Dict,
) -> Dict:
    print(
        "[Rules][RF] start "
        f"train={len(y_train)} test={len(y_test)} features={len(feature_names)} "
        f"n_seeds={cfg.get('n_seeds', 400)}"
    )
    forest_result = run_forest_search(
        X_train,
        y_train,
        n_seeds=int(cfg.get("n_seeds", 400)),
        accuracy_quantile=float(cfg.get("accuracy_quantile", 0.99)),
        min_trees=int(cfg.get("min_trees", 1)),
        max_trees=int(cfg.get("max_trees", 10)),
        min_questions=int(cfg.get("min_questions", 2)),
        max_questions=int(cfg.get("max_questions", 10)),
        test_size=float(cfg.get("test_size", 0.2)),
        vote_rule=str(cfg.get("vote_rule", "any")),
        eval_mode=str(cfg.get("eval_mode", "holdout")),
        n_jobs=int(cfg.get("n_jobs", 1)),
        max_depth=int(cfg["max_depth"]) if cfg.get("max_depth") is not None else None,
    )

    trees, _ = rebuild_forest(
        X_train,
        y_train,
        forest_result.best_seed,
        int(cfg.get("min_trees", 1)),
        int(cfg.get("max_trees", 10)),
        int(cfg.get("min_questions", 2)),
        int(cfg.get("max_questions", 10)),
        max_depth=int(cfg["max_depth"]) if cfg.get("max_depth") is not None else None,
    )
    vote_rule = str(cfg.get("vote_rule", "any"))
    train_pred = forest_predict(trees, X_train, vote_rule=vote_rule)
    test_pred = forest_predict(trees, X_test, vote_rule=vote_rule)
    train_score = _forest_score_matrix(trees, X_train)
    test_score = _forest_score_matrix(trees, X_test)

    metrics = {
        "train": classification_metrics(y_train, train_pred, train_score),
        "test": classification_metrics(y_test, test_pred, test_score),
    }
    metrics["train"]["n_rows"] = int(len(y_train))
    metrics["test"]["n_rows"] = int(len(y_test))
    metrics["train"]["n_positive"] = int(np.sum(y_train))
    metrics["test"]["n_positive"] = int(np.sum(y_test))

    out_dir.mkdir(parents=True, exist_ok=True)
    forest_result.results_df.to_csv(out_dir / "forest_search_results.csv", index=False)
    (out_dir / "best_forest.json").write_text(
        json.dumps(forest_result.best_row.to_dict(), indent=2),
        encoding="utf-8",
    )
    save_forest_accuracy_plot(
        forest_result.results_df,
        out_dir / "accuracy_vs_questions",
        best_row=forest_result.best_row,
    )

    positive_rules = _save_tree_bundle(
        trees,
        X_train,
        y_train,
        feature_names,
        target_train,
        meta_train,
        out_dir / "trees",
    )
    (out_dir / "rules.json").write_text(json.dumps(positive_rules, indent=2), encoding="utf-8")

    context_note = (
        "Each sample corresponds to a contiguous same-interestingness segment within a subject; "
        "positive rules therefore describe when an interesting span occurs and report its typical years."
    )
    nl = render_natural_language_rules("random_forest_module", positive_rules, context_note)
    (out_dir / "rules_natural_language.txt").write_text(nl, encoding="utf-8")

    print(
        "[Rules][RF] done | "
        f"train_acc={metrics['train'].get('accuracy', 0.0):.4f} "
        f"test_acc={metrics['test'].get('accuracy', 0.0):.4f} "
        f"n_trees={len(trees)} n_rules={len(positive_rules)}"
    )

    return {
        "metrics": metrics,
        "n_trees": int(len(trees)),
        "n_rules": int(len(positive_rules)),
        "best_seed": int(forest_result.best_seed),
    }
