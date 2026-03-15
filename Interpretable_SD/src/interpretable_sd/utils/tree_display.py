from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.tree import DecisionTreeClassifier


def compute_leaf_target_stats(
    tree: DecisionTreeClassifier,
    X: np.ndarray,
    target_values: np.ndarray,
    span_start_years: Optional[np.ndarray] = None,
    span_end_years: Optional[np.ndarray] = None,
    span_lengths: Optional[np.ndarray] = None,
) -> Dict[int, Dict]:
    leaf_ids = tree.apply(X)
    target_arr = np.asarray(target_values, dtype=float)
    start_arr = None if span_start_years is None else np.asarray(span_start_years, dtype=float)
    end_arr = None if span_end_years is None else np.asarray(span_end_years, dtype=float)
    length_arr = None if span_lengths is None else np.asarray(span_lengths, dtype=float)
    out: Dict[int, Dict] = {}
    for leaf_id in np.unique(leaf_ids):
        vals = target_arr[leaf_ids == leaf_id]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[int(leaf_id)] = {
            "n_samples": int(vals.size),
            "target_mean": float(np.mean(vals)),
            "target_median": float(np.median(vals)),
        }
        if start_arr is not None:
            starts = start_arr[leaf_ids == leaf_id]
            starts = starts[np.isfinite(starts)]
            if starts.size:
                out[int(leaf_id)].update(
                    {
                        "span_start_year_median": int(round(float(np.median(starts)))),
                        "span_start_year_min": int(np.min(starts)),
                        "span_start_year_max": int(np.max(starts)),
                    }
                )
        if end_arr is not None:
            ends = end_arr[leaf_ids == leaf_id]
            ends = ends[np.isfinite(ends)]
            if ends.size:
                out[int(leaf_id)].update(
                    {
                        "span_end_year_median": int(round(float(np.median(ends)))),
                        "span_end_year_min": int(np.min(ends)),
                        "span_end_year_max": int(np.max(ends)),
                    }
                )
        if length_arr is not None:
            lengths = length_arr[leaf_ids == leaf_id]
            lengths = lengths[np.isfinite(lengths)]
            if lengths.size:
                out[int(leaf_id)].update(
                    {
                        "span_length_mean": float(np.mean(lengths)),
                        "span_length_median": float(np.median(lengths)),
                    }
                )
    return out


def simplify_tree_plot_labels(
    artists,
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    leaf_stats: Optional[Dict[int, dict]] = None,
    positive_class: int = 1,
    positive_label: str = "interesting",
    negative_label: str = "not interesting",
) -> None:
    tree_ = tree.tree_
    for artist in artists:
        text = artist.get_text().strip()
        if text in {"True", "False"}:
            artist.set_text("")
            continue
        if not text.startswith("#"):
            continue
        try:
            node_id = int(text[1:])
        except ValueError:
            continue
        if node_id < 0 or node_id >= tree_.node_count:
            continue

        is_leaf = tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1
        if is_leaf:
            value = tree_.value[node_id][0]
            class_index = int(np.argmax(value))
            pred = int(tree.classes_[class_index])
            if pred == positive_class:
                if leaf_stats and node_id in leaf_stats and leaf_stats[node_id].get("target_mean") is not None:
                    artist.set_text(f"{positive_label}\nmean={float(leaf_stats[node_id]['target_mean']):.4g}")
                else:
                    artist.set_text(positive_label)
            else:
                artist.set_text(negative_label)
            continue

        feat_idx = int(tree_.feature[node_id])
        feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f"f{feat_idx}"
        thresh = float(tree_.threshold[node_id])
        artist.set_text(f"{feat_name} <= {thresh:.4g}")


def _compute_node_positions(tree: DecisionTreeClassifier) -> Dict[int, tuple]:
    tree_ = tree.tree_
    positions: Dict[int, tuple] = {}
    leaf_counter = {"v": 0}

    def recurse(node_id: int, depth: int):
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        is_leaf = left == -1 and right == -1
        if is_leaf:
            x = float(leaf_counter["v"])
            leaf_counter["v"] += 1
            positions[node_id] = (x, float(depth))
            return x
        x_left = recurse(left, depth + 1)
        x_right = recurse(right, depth + 1)
        x_here = 0.5 * (x_left + x_right)
        positions[node_id] = (x_here, float(depth))
        return x_here

    recurse(0, 0)
    return positions


def save_constant_box_tree_plot(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    output_base: Path,
    title: str,
    leaf_stats: Optional[Dict[int, dict]] = None,
    positive_class: int = 1,
    positive_label: str = "interesting",
    negative_label: str = "not interesting",
    figsize=(16, 9),
    box_width: float = 0.17,
    box_height: float = 0.09,
) -> None:
    tree_ = tree.tree_
    positions = _compute_node_positions(tree)
    if not positions:
        return

    xs = [v[0] for v in positions.values()]
    ys = [v[1] for v in positions.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0, y_max - y_min)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    def norm_xy(node_id: int):
        x_raw, y_raw = positions[node_id]
        x = 0.05 + 0.90 * ((x_raw - x_min) / x_span if x_span > 0 else 0.5)
        y = 0.95 - 0.90 * ((y_raw - y_min) / y_span if y_span > 0 else 0.0)
        return x, y

    # edges first
    for node_id in range(tree_.node_count):
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left != -1:
            x0, y0 = norm_xy(node_id)
            x1, y1 = norm_xy(left)
            ax.plot([x0, x1], [y0 - box_height / 2, y1 + box_height / 2], color="#777777", lw=1.2, transform=ax.transAxes)
        if right != -1:
            x0, y0 = norm_xy(node_id)
            x1, y1 = norm_xy(right)
            ax.plot([x0, x1], [y0 - box_height / 2, y1 + box_height / 2], color="#777777", lw=1.2, transform=ax.transAxes)

    # nodes
    for node_id in range(tree_.node_count):
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        is_leaf = left == -1 and right == -1
        x, y = norm_xy(node_id)

        if is_leaf:
            value = tree_.value[node_id][0]
            class_index = int(np.argmax(value))
            pred = int(tree.classes_[class_index])
            if pred == positive_class:
                fill = "#ffe6b3"
                if leaf_stats and node_id in leaf_stats and leaf_stats[node_id].get("target_mean") is not None:
                    text = f"{positive_label}\nmean={float(leaf_stats[node_id]['target_mean']):.4g}"
                else:
                    text = positive_label
            else:
                fill = "#e6e6e6"
                text = negative_label
        else:
            feat_idx = int(tree_.feature[node_id])
            feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f"f{feat_idx}"
            thresh = float(tree_.threshold[node_id])
            text = f"{feat_name} <= {thresh:.4g}"
            fill = "#dff0ff"

        rect = Rectangle(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor="#4d4d4d",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            clip_on=True,
        )

    plt.title(title)
    plt.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), dpi=260, bbox_inches="tight")
    plt.close(fig)
