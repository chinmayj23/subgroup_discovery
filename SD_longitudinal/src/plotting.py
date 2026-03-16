from pathlib import Path
from typing import Dict, List, Optional

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .forest import extract_rules, tree_to_dict


def _save_multi(fig, base_path: Path, dpi: int = 300) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")


def save_kde_plot(
    df: pd.DataFrame,
    target: str,
    label_column: str,
    output_base: Path,
    title: Optional[str] = None,
) -> None:
    if label_column not in df.columns or target not in df.columns:
        return

    target_series = pd.to_numeric(df[target], errors="coerce")
    interesting = df[df[label_column] == True]
    non_interesting = df[df[label_column] == False]

    fig = plt.figure(figsize=(12, 7))
    sns.kdeplot(
        target_series,
        color="blue",
        fill=True,
        alpha=0.25,
        label="Original",
        bw_adjust=1.1,
        common_norm=False,
    )

    non_interesting_target = pd.to_numeric(non_interesting[target], errors="coerce")
    interesting_target = pd.to_numeric(interesting[target], errors="coerce")

    if not non_interesting_target.empty:
        sns.kdeplot(
            non_interesting_target,
            color="green",
            fill=True,
            alpha=0.25,
            label="Non-Interesting",
            bw_adjust=1.1,
            common_norm=False,
        )
        sns.rugplot(non_interesting_target, color="green", height=0.02, alpha=0.15)

    if not interesting_target.empty:
        sns.kdeplot(
            interesting_target,
            color="red",
            fill=True,
            alpha=0.25,
            label="Interesting",
            bw_adjust=1.1,
            common_norm=False,
        )
        sns.rugplot(interesting_target, color="red", height=0.03, alpha=0.25)

    if title:
        plt.title(title)
    plt.xlabel(target)
    plt.ylabel("Density")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.tight_layout()

    _save_multi(fig, output_base)
    plt.close(fig)


def save_forest_accuracy_plot(
    results_df: pd.DataFrame,
    output_base: Path,
    best_row: Optional[pd.Series] = None,
    title: Optional[str] = None,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(
        results_df["total_questions"],
        results_df["forest_accuracy"],
        s=40,
        alpha=0.7,
        color="blue",
        label="All forests",
    )
    if best_row is not None:
        plt.scatter(
            [best_row["total_questions"]],
            [best_row["forest_accuracy"]],
            s=70,
            color="red",
            label="Selected forest",
            zorder=3,
        )
    plt.xlabel("Total Number of Questions in Forest", fontsize=12)
    plt.ylabel("Forest Accuracy", fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    _save_multi(fig, output_base)
    plt.close(fig)


def save_cutoff_accuracy_plot(
    results_df: pd.DataFrame,
    output_base: Path,
    title: Optional[str] = "Test Accuracy By Cut-Off Year",
) -> None:
    required = {"cutoff_year", "test_accuracy"}
    if results_df.empty or not required.issubset(results_df.columns):
        return

    df = results_df.sort_values("cutoff_year")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        df["cutoff_year"],
        df["test_accuracy"],
        marker="o",
        linewidth=2.0,
        color="tab:blue",
        label="Test accuracy",
    )
    if "train_accuracy" in df.columns:
        plt.plot(
            df["cutoff_year"],
            df["train_accuracy"],
            marker="s",
            linewidth=1.6,
            linestyle="--",
            color="tab:orange",
            label="Train accuracy",
        )
    plt.xlabel("Cut-Off Year", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    _save_multi(fig, output_base)
    plt.close(fig)


def _rewrite_tree_plot_labels(
    artists,
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    leaf_stats: Optional[Dict[int, dict]] = None,
) -> None:
    tree_ = tree.tree_
    valid_node_ids = set(range(tree_.node_count))

    for artist in artists:
        text = artist.get_text().strip()
        if text in {"True", "False"}:
            artist.set_text("")
            continue

        match = re.match(r"#(\d+)", text)
        if not match:
            continue
        node_id = int(match.group(1))
        if node_id not in valid_node_ids:
            continue

        is_leaf = (
            tree_.children_left[node_id] == -1
            and tree_.children_right[node_id] == -1
        )
        if is_leaf:
            value = tree_.value[node_id][0]
            class_index = int(np.argmax(value))
            pred = int(tree.classes_[class_index])
            if leaf_stats and node_id in leaf_stats:
                mean_val = leaf_stats[node_id].get("target_mean")
                if mean_val is not None and np.isfinite(mean_val):
                    artist.set_text(f"mean={float(mean_val):.4g}\npred={pred}")
                    continue
            artist.set_text(f"pred={pred}")
            continue

        feat_idx = int(tree_.feature[node_id])
        feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f"f{feat_idx}"
        thresh = float(tree_.threshold[node_id])
        artist.set_text(f"{feat_name} <= {thresh:.4g}")


def save_forest_tree_artifacts(
    trees: List[DecisionTreeClassifier],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    question_map: Optional[Dict[str, str]] = None,
    target_values: Optional[np.ndarray] = None,
    target_name: Optional[str] = None,
) -> None:
    if not trees:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, tree in enumerate(trees):
        acc = float((tree.predict(X) == y).mean())
        leaf_stats = None
        global_stats = None
        if target_values is not None and len(target_values) == len(X):
            leaf_ids = tree.apply(X)
            target_arr = np.asarray(target_values, dtype=float)
            global_median = float(np.nanmedian(target_arr))
            global_mean = float(np.nanmean(target_arr))
            global_stats = {
                "target_name": target_name,
                "target_median": global_median,
                "target_mean": global_mean,
            }
            leaf_stats = {}
            for leaf_id in np.unique(leaf_ids):
                mask = leaf_ids == leaf_id
                vals = target_arr[mask]
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                leaf_median = float(np.median(vals))
                leaf_mean = float(np.mean(vals))
                leaf_stats[int(leaf_id)] = {
                    "n_samples": int(vals.size),
                    "target_median": leaf_median,
                    "target_mean": leaf_mean,
                    "target_min": float(np.min(vals)),
                    "target_max": float(np.max(vals)),
                    "target_position": "high" if leaf_median > global_median else "low",
                }

        tree_dict = tree_to_dict(
            tree,
            feature_names=feature_names,
            question_map=question_map,
            leaf_stats=leaf_stats,
            global_stats=global_stats,
        )
        tree_json = out_dir / f"tree_{idx:02d}_acc_{acc:.4f}.json"
        tree_json.write_text(json.dumps(tree_dict, indent=2), encoding="utf-8")

        rules = extract_rules(
            tree,
            feature_names=feature_names,
            question_map=question_map,
            leaf_stats=leaf_stats,
        )
        rules_path = out_dir / f"tree_{idx:02d}_rules.json"
        rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")

        fig = plt.figure(figsize=(14, 7))
        artists = plot_tree(
            tree,
            feature_names=feature_names,
            class_names=["not interesting", "interesting"],
            filled=True,
            rounded=True,
            impurity=False,
            proportion=False,
            label="none",
            node_ids=True,
        )
        _rewrite_tree_plot_labels(artists, tree, feature_names, leaf_stats=leaf_stats)
        plt.title(f"Tree {idx}")
        plt.tight_layout()
        _save_multi(fig, out_dir / f"tree_{idx:02d}_acc_{acc:.4f}")
        plt.close(fig)
