from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


@dataclass
class ForestSearchResult:
    results_df: pd.DataFrame
    best_seed: int
    best_row: pd.Series


def prune_pure_subtrees(tree: DecisionTreeClassifier) -> None:
    tree_ = tree.tree_

    def recurse(node_id: int):
        if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:
            value = tree_.value[node_id][0]
            class_index = int(np.argmax(value))
            return {class_index}

        left_id = tree_.children_left[node_id]
        right_id = tree_.children_right[node_id]

        left_classes = recurse(left_id)
        right_classes = recurse(right_id)

        union = left_classes | right_classes

        if len(union) == 1:
            tree_.children_left[node_id] = -1
            tree_.children_right[node_id] = -1
            tree_.feature[node_id] = _tree.TREE_UNDEFINED
            tree_.threshold[node_id] = -2.0
        return union

    recurse(0)


def count_questions(tree: DecisionTreeClassifier) -> int:
    tree_ = tree.tree_
    children_left = tree_.children_left
    children_right = tree_.children_right

    def dfs(node_id: int) -> int:
        if node_id == -1:
            return 0
        if children_left[node_id] == -1 and children_right[node_id] == -1:
            return 0
        return 1 + dfs(children_left[node_id]) + dfs(children_right[node_id])

    return int(dfs(0))


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    eval_mode: str,
):
    if eval_mode == "train":
        return X, X, y, y

    if len(X) < 10 or len(np.unique(y)) < 2:
        return X, X, y, y

    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def vote_predictions(pred_matrix: np.ndarray, vote_rule: str) -> np.ndarray:
    if pred_matrix.size == 0:
        return np.array([], dtype=int)

    if vote_rule == "majority":
        return (pred_matrix.mean(axis=0) >= 0.5).astype(int)
    return np.any(pred_matrix, axis=0).astype(int)


def forest_predict(
    trees: List[DecisionTreeClassifier],
    X: np.ndarray,
    vote_rule: str = "any",
) -> np.ndarray:
    if len(trees) == 0:
        return np.zeros(X.shape[0], dtype=int)
    pred_matrix = np.vstack([clf.predict(X).astype(bool) for clf in trees])
    return vote_predictions(pred_matrix, vote_rule)


def build_forest_metrics(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    min_trees: int,
    max_trees: int,
    min_questions: int,
    max_questions: int,
    test_size: float,
    vote_rule: str,
    eval_mode: str,
    max_depth: Optional[int] = None,
) -> dict:
    X_train, X_val, y_train, y_val = _split_data(X, y, seed, test_size, eval_mode)

    rng = np.random.RandomState(seed)
    n_trees = int(rng.randint(min_trees, max_trees + 1))
    max_questions = int(rng.randint(min_questions, max_questions + 1))

    trees: List[DecisionTreeClassifier] = []
    tree_question_counts: List[int] = []

    for _ in range(n_trees):
        n_train = X_train.shape[0]
        bootstrap_idx = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_train[bootstrap_idx]
        y_boot = y_train[bootstrap_idx]

        clf = DecisionTreeClassifier(
            max_leaf_nodes=max_questions + 1,
            max_depth=max_depth,
            random_state=int(rng.randint(0, 1_000_000)),
        )
        clf.fit(X_boot, y_boot)
        prune_pure_subtrees(clf)

        trees.append(clf)
        tree_question_counts.append(count_questions(clf))

    if len(trees) == 0:
        return {
            "seed": seed,
            "forest_accuracy": 0.0,
            "n_trees": 0,
            "total_questions": 0,
            "max_questions": max_questions,
        }

    pred_matrix = np.vstack([clf.predict(X_val).astype(bool) for clf in trees])
    forest_pred = vote_predictions(pred_matrix, vote_rule)

    forest_accuracy = float((forest_pred == y_val).mean())
    total_questions = int(np.sum(tree_question_counts))

    return {
        "seed": seed,
        "forest_accuracy": forest_accuracy,
        "n_trees": len(trees),
        "total_questions": total_questions,
        "max_questions": max_questions,
    }


def run_forest_search(
    X: np.ndarray,
    y: np.ndarray,
    n_seeds: int = 1000,
    accuracy_quantile: float = 0.99,
    min_trees: int = 1,
    max_trees: int = 10,
    min_questions: int = 2,
    max_questions: int = 10,
    test_size: float = 0.2,
    vote_rule: str = "any",
    eval_mode: str = "train",
    n_jobs: int = -1,
    max_depth: Optional[int] = None,
) -> ForestSearchResult:
    def _runner(seed: int) -> dict:
        return build_forest_metrics(
            X,
            y,
            seed,
            min_trees,
            max_trees,
            min_questions,
            max_questions,
            test_size,
            vote_rule,
            eval_mode,
            max_depth,
        )

    if JOBLIB_AVAILABLE and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(delayed(_runner)(s) for s in range(n_seeds))
    else:
        results = [_runner(s) for s in range(n_seeds)]

    results_df = pd.DataFrame(results)

    acc_threshold = results_df["forest_accuracy"].quantile(accuracy_quantile)
    top_df = results_df[results_df["forest_accuracy"] > acc_threshold]
    if top_df.empty:
        top_df = results_df[results_df["forest_accuracy"] >= acc_threshold]

    top_sorted = top_df.sort_values(
        by=["total_questions", "forest_accuracy", "seed"],
        ascending=[True, False, True],
    )

    best_row = top_sorted.iloc[0]
    best_seed = int(best_row["seed"])

    return ForestSearchResult(results_df=results_df, best_seed=best_seed, best_row=best_row)


def rebuild_forest(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    min_trees: int,
    max_trees: int,
    min_questions: int,
    max_questions: int,
    max_depth: Optional[int] = None,
) -> Tuple[List[DecisionTreeClassifier], int]:
    rng = np.random.RandomState(seed)
    n_trees = int(rng.randint(min_trees, max_trees + 1))
    max_questions = int(rng.randint(min_questions, max_questions + 1))

    trees: List[DecisionTreeClassifier] = []
    for _ in range(n_trees):
        idx = rng.choice(len(X), size=len(X), replace=True)
        clf = DecisionTreeClassifier(
            max_leaf_nodes=max_questions + 1,
            max_depth=max_depth,
            random_state=int(rng.randint(0, 1_000_000)),
        )
        clf.fit(X[idx], y[idx])
        prune_pure_subtrees(clf)
        trees.append(clf)

    return trees, max_questions


def tree_to_dict(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    question_map: Optional[Dict[str, str]] = None,
    leaf_stats: Optional[Dict[int, dict]] = None,
    global_stats: Optional[Dict[str, float]] = None,
) -> dict:
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    def _question_text(feature_name: str, thresh: float) -> str:
        base = feature_name
        if question_map and feature_name in question_map:
            base = question_map[feature_name]
        return f"{base} <= {thresh:.4f}"

    def recurse(node_id: int):
        if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:
            value = tree_.value[node_id][0]
            class_index = int(np.argmax(value))
            class_label = int(tree.classes_[class_index])
            leaf_payload = {
                "leaf_prediction": class_label,
                "class_counts": value.tolist(),
            }
            if leaf_stats and node_id in leaf_stats:
                leaf_payload["target_summary"] = leaf_stats[node_id]
            if global_stats:
                leaf_payload["global_target_summary"] = global_stats
            return leaf_payload

        feat_name = feature_names[feature[node_id]]
        thresh = float(threshold[node_id])

        return {
            "question": f"{feat_name} <= {thresh}",
            "question_text": _question_text(feat_name, thresh),
            "feature": feat_name,
            "threshold": thresh,
            "left": recurse(tree_.children_left[node_id]),
            "right": recurse(tree_.children_right[node_id]),
        }

    return recurse(0)


def extract_rules(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    question_map: Optional[Dict[str, str]] = None,
    leaf_stats: Optional[Dict[int, dict]] = None,
) -> List[dict]:
    tree_ = tree.tree_

    def _question_text(feature_name: str, thresh: float, direction: str) -> str:
        base = feature_name
        if question_map and feature_name in question_map:
            base = question_map[feature_name]
        op = "<=" if direction == "left" else ">"
        return f"{base} {op} {thresh:.4f}"

    rules: List[dict] = []

    def recurse(node_id: int, path: List[str]):
        if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:
            value = tree_.value[node_id][0]
            class_index = int(np.argmax(value))
            class_label = int(tree.classes_[class_index])
            if class_label == 1:
                payload = {
                    "leaf_id": int(node_id),
                    "prediction": int(class_label),
                    "n_samples": int(tree_.n_node_samples[node_id]),
                    "rule": list(path),
                }
                if leaf_stats and node_id in leaf_stats:
                    payload["target_summary"] = leaf_stats[node_id]
                rules.append(payload)
            return

        feat_name = feature_names[tree_.feature[node_id]]
        thresh = float(tree_.threshold[node_id])
        left_q = _question_text(feat_name, thresh, "left")
        right_q = _question_text(feat_name, thresh, "right")

        recurse(tree_.children_left[node_id], path + [left_q])
        recurse(tree_.children_right[node_id], path + [right_q])

    recurse(0, [])
    return rules
