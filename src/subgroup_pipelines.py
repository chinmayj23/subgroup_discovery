import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier, _tree


LABEL_COLUMN = "is_interesting_subgroup"


@dataclass
class ForestSearchResult:
    results_df: pd.DataFrame
    best_seed: int
    best_row: pd.Series


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freedman_diaconis_bins(data):
    data = np.asarray(data).ravel()
    if data.size < 2:
        return np.array([data.min(), data.max()])
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        nbins = max(1, int(np.sqrt(data.size)))
    else:
        h = 2 * iqr / (data.size ** (1 / 3))
        nbins = int(np.ceil((data.max() - data.min()) / h)) if h > 0 else 10
    nbins = max(1, nbins)
    return np.histogram_bin_edges(data, bins=nbins)


def hist_kl(p_samples, q_samples, global_bins, eps=1e-9):
    p, q = np.asarray(p_samples).ravel(), np.asarray(q_samples).ravel()
    p_hist, _ = np.histogram(p, bins=global_bins, density=True)
    q_hist, _ = np.histogram(q, bins=global_bins, density=True)
    p_prob = (p_hist + eps) / (p_hist.sum() + eps * len(p_hist))
    q_prob = (q_hist + eps) / (q_hist.sum() + eps * len(q_hist))
    return float(entropy(p_prob, q_prob))


def jaccard_similarity(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


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


def build_and_evaluate_forest(X, y, seed: int):
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]

    max_n_trees = rng.randint(1, 11)
    max_questions = rng.randint(2, 11)

    trees = []
    tree_accuracies = []
    tree_question_counts = []

    for _ in range(max_n_trees):
        bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[bootstrap_indices]
        y_boot = y[bootstrap_indices]

        tree_random_state = rng.randint(0, 1_000_000)
        clf = DecisionTreeClassifier(
            max_leaf_nodes=max_questions + 1,
            random_state=tree_random_state,
        )
        clf.fit(X_boot, y_boot)
        prune_pure_subtrees(clf)

        y_pred_tree = clf.predict(X)
        acc_tree = (y_pred_tree == y).mean()
        n_questions = count_questions(clf)

        trees.append(clf)
        tree_accuracies.append(acc_tree)
        tree_question_counts.append(n_questions)

    pred_matrix = np.vstack([clf.predict(X).astype(bool) for clf in trees])
    forest_pred = np.any(pred_matrix, axis=0).astype(int)

    forest_accuracy = (forest_pred == y).mean()
    total_questions = int(np.sum(tree_question_counts))

    return {
        "seed": seed,
        "forest_accuracy": forest_accuracy,
        "n_trees": len(trees),
        "total_questions": total_questions,
    }


def rebuild_forest_for_seed(X, y, seed: int):
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]

    max_n_trees = rng.randint(1, 11)
    max_questions = rng.randint(2, 11)

    trees = []

    for _ in range(max_n_trees):
        bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[bootstrap_indices]
        y_boot = y[bootstrap_indices]

        tree_random_state = rng.randint(0, 1_000_000)
        clf = DecisionTreeClassifier(
            max_leaf_nodes=max_questions + 1,
            random_state=tree_random_state,
        )
        clf.fit(X_boot, y_boot)
        prune_pure_subtrees(clf)
        trees.append(clf)

    return trees, max_questions


def run_forest_search(X, y, n_seeds=1000, accuracy_quantile=0.99) -> ForestSearchResult:
    results = []
    for s in range(n_seeds):
        metrics = build_and_evaluate_forest(X, y, s)
        results.append(metrics)

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


def tree_to_dict(tree: DecisionTreeClassifier, feature_names):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    def recurse(node_id: int):
        if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:
            value = tree_.value[node_id][0]
            class_index = int(np.argmax(value))
            class_label = int(tree.classes_[class_index])
            return {
                "leaf_prediction": class_label,
                "class_counts": value.tolist(),
            }

        feat_name = feature_names[feature[node_id]]
        thresh = float(threshold[node_id])

        return {
            "question": f"{feat_name} <= {thresh}",
            "feature": feat_name,
            "threshold": thresh,
            "left": recurse(tree_.children_left[node_id]),
            "right": recurse(tree_.children_right[node_id]),
        }

    return recurse(0)


def save_kde_plot(df, target, label_column, output_path):
    if label_column not in df.columns:
        return
    if target not in df.columns:
        return

    target_series = pd.to_numeric(df[target], errors="coerce")
    interesting = df[df[label_column] == True]
    non_interesting = df[df[label_column] == False]

    plt.figure(figsize=(12, 7))
    sns.kdeplot(
        target_series,
        color="blue",
        fill=True,
        alpha=0.25,
        label="Original",
        bw_adjust=1.1,
        common_norm=False,
    )
    sns.kdeplot(
        pd.to_numeric(non_interesting[target], errors="coerce"),
        color="green",
        fill=True,
        alpha=0.25,
        label="Non-Interesting",
        bw_adjust=1.1,
        common_norm=False,
    )
    sns.kdeplot(
        pd.to_numeric(interesting[target], errors="coerce"),
        color="red",
        fill=True,
        alpha=0.25,
        label="Interesting",
        bw_adjust=1.1,
        common_norm=False,
    )

    sns.rugplot(
        pd.to_numeric(non_interesting[target], errors="coerce"),
        color="green",
        height=0.02,
        alpha=0.15,
    )
    sns.rugplot(
        pd.to_numeric(interesting[target], errors="coerce"),
        color="red",
        height=0.03,
        alpha=0.25,
    )

    plt.title(f"KDE Plots of {target} with Point Distributions")
    plt.xlabel(target)
    plt.ylabel("Density")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_forest_tree_artifacts(trees, X, y, feature_names, out_dir):
    if not trees:
        return

    from sklearn.tree import plot_tree

    out_dir = Path(out_dir)
    for idx, tree in enumerate(trees):
        acc = float((tree.predict(X) == y).mean())
        tree_dict = tree_to_dict(tree, feature_names=feature_names)
        tree_json = out_dir / f"tree_{idx:02d}_acc_{acc:.4f}.json"
        tree_json.write_text(json.dumps(tree_dict, indent=2), encoding="utf-8")

        plt.figure(figsize=(12, 6))
        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=["not interesting", "interesting"],
            filled=True,
            rounded=True,
            impurity=False,
        )
        plt.title(f"Tree {idx} (acc={acc:.4f})")
        plt.tight_layout()
        plt.savefig(out_dir / f"tree_{idx:02d}_acc_{acc:.4f}.png", dpi=300)
        plt.close()


def load_dataset(path, target, drop_columns=None, sep=","):
    df = pd.read_csv(path, sep=sep)
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    return df


def load_dataset_from_config(dataset_cfg):
    if "path" in dataset_cfg:
        return load_dataset(
            dataset_cfg["path"],
            dataset_cfg["target"],
            drop_columns=dataset_cfg.get("drop_columns"),
            sep=dataset_cfg.get("sep", ","),
        )

    source = dataset_cfg.get("source")
    if source == "sklearn":
        return load_sklearn_dataset(dataset_cfg)
    if source == "statsmodels":
        return load_statsmodels_dataset(dataset_cfg)
    if source == "openml":
        return load_openml_dataset(dataset_cfg)

    raise ValueError("Dataset config must include 'path' or a supported 'source'")


def load_sklearn_dataset(dataset_cfg):
    from sklearn import datasets as sk_datasets

    name = dataset_cfg["dataset"]
    loader_map = {
        "diabetes": ("load_diabetes", {}),
        "wine": ("load_wine", {}),
        "breast_cancer": ("load_breast_cancer", {}),
        "california_housing": ("fetch_california_housing", {"as_frame": True}),
    }
    if name not in loader_map:
        raise ValueError(f"Unsupported sklearn dataset '{name}'")

    loader_name, loader_kwargs = loader_map[name]
    loader = getattr(sk_datasets, loader_name)
    loader_kwargs = dict(loader_kwargs)
    loader_kwargs.setdefault("as_frame", True)
    dataset = loader(**loader_kwargs)

    if hasattr(dataset, "frame") and dataset.frame is not None:
        df = dataset.frame.copy()
    else:
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df["target"] = dataset.target

    target = dataset_cfg.get("target")
    if target is None:
        target_names = getattr(dataset, "target_names", None)
        if isinstance(target_names, (list, tuple)) and target_names:
            target = target_names[0]
        elif isinstance(target_names, str):
            target = target_names
        else:
            target = "target"

    if target not in df.columns:
        df[target] = dataset.target

    df[target] = pd.to_numeric(df[target], errors="coerce")
    return df


def load_statsmodels_dataset(dataset_cfg):
    import statsmodels.api as sm

    name = dataset_cfg["dataset"]
    dataset = getattr(sm.datasets, name, None)
    if dataset is None:
        raise ValueError(f"Unsupported statsmodels dataset '{name}'")

    data = dataset.load_pandas().data.copy()
    target = dataset_cfg.get("target")
    if target and target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in statsmodels dataset '{name}'")
    if target:
        data[target] = pd.to_numeric(data[target], errors="coerce")
    return data


def load_openml_dataset(dataset_cfg):
    from sklearn.datasets import fetch_openml

    name = dataset_cfg.get("dataset")
    data_id = dataset_cfg.get("data_id")
    if not name and not data_id:
        raise ValueError("OpenML datasets require 'dataset' or 'data_id'")

    dataset = fetch_openml(name=name, data_id=data_id, as_frame=True)
    df = dataset.frame.copy()
    target = dataset_cfg.get("target")
    if target is None:
        target_names = dataset.target_names
        if isinstance(target_names, (list, tuple)) and target_names:
            target = target_names[0]
        else:
            target = target_names
    if target and target not in df.columns:
        df[target] = dataset.target
    if target:
        df[target] = pd.to_numeric(df[target], errors="coerce")
    return df


def prepare_xy(df, target, label_column=LABEL_COLUMN):
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    y = df[label_column].astype(int)
    drop_cols = [target, label_column]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric feature columns available after filtering")
    X = numeric_df

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]

    return X.values, y.values, list(X.columns)


def label_original_method(df, target, high_value_quantile=0.94):
    y = df[target].values.reshape(-1, 1)
    Z = linkage(y, method="single")
    heights = Z[:, 2]
    diffs = np.diff(heights)
    if len(diffs) > 0:
        threshold = heights[np.argmax(diffs)]
    else:
        threshold = np.max(heights) if len(heights) > 0 else 0

    clusters = fcluster(Z, t=threshold, criterion="distance")
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    largest_cluster_label = unique_clusters[np.argmax(counts)]

    interesting_mask = clusters != largest_cluster_label
    high_value_cutoff = df[target].quantile(high_value_quantile)
    interesting_mask |= df[target] >= high_value_cutoff

    labeled_df = df.copy()
    labeled_df[LABEL_COLUMN] = interesting_mask
    return labeled_df


def label_syflow_method(
    df,
    target,
    n_seeds=1000,
    beta=0.5,
    lambd_div=2.0,
    top_percentile=0.01,
    max_overlap=0.95,
    min_subgroup_size=50,
    max_subgroup_frac=0.90,
):
    y_full = df[target].values
    n_total = len(y_full)

    global_bins = freedman_diaconis_bins(y_full)
    candidates = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(n_total, size=n_total, replace=True)
        y_boot = y_full[sample_idx].reshape(-1, 1)

        Z = linkage(y_boot, method="single")
        heights = Z[:, 2]
        diffs = np.diff(heights)
        threshold = heights[np.argmax(diffs)] if len(diffs) > 0 else 0.0
        if threshold == 0:
            continue

        boot_clusters = fcluster(Z, t=threshold, criterion="distance")
        unique_labels = np.unique(boot_clusters)

        for label in unique_labels:
            cluster_vals = y_boot[boot_clusters == label]
            if len(cluster_vals) == 0:
                continue
            c_min, c_max = cluster_vals.min(), cluster_vals.max()

            mask = (y_full >= c_min) & (y_full <= c_max)
            n_sub = int(mask.sum())

            if n_sub < min_subgroup_size or n_sub > (n_total * max_subgroup_frac):
                continue

            sub_y = y_full[mask]
            kl_exc = hist_kl(sub_y, y_full, global_bins)
            size_term = (n_sub / n_total) ** beta
            quality = size_term * kl_exc

            candidates.append(
                {
                    "seed": seed,
                    "mask": mask,
                    "y_values": sub_y,
                    "size_term": size_term,
                    "quality": quality,
                    "range": (c_min, c_max),
                    "selected": False,
                }
            )

    if len(candidates) == 0:
        labeled_df = df.copy()
        labeled_df[LABEL_COLUMN] = False
        return labeled_df

    n_select = int(len(candidates) * top_percentile)
    n_select = max(1, n_select)

    selected_candidates = []

    for _ in range(n_select):
        best_candidate = None
        best_score = -np.inf

        for cand in candidates:
            if cand["selected"]:
                continue

            is_duplicate = False
            for sel in selected_candidates:
                if jaccard_similarity(cand["mask"], sel["mask"]) > max_overlap:
                    is_duplicate = True
                    break
            if is_duplicate:
                cand["selected"] = True
                continue

            diversity = 0.0
            if len(selected_candidates) > 0:
                for sel in selected_candidates:
                    diversity += hist_kl(cand["y_values"], sel["y_values"], global_bins)
                diversity /= len(selected_candidates)

            total_score = cand["quality"] + (lambd_div * cand["size_term"] * diversity)
            if total_score > best_score:
                best_score = total_score
                best_candidate = cand

        if best_candidate is None:
            break

        best_candidate["selected"] = True
        selected_candidates.append(best_candidate)

    final_mask = np.zeros(n_total, dtype=bool)
    for cand in selected_candidates:
        final_mask |= cand["mask"]

    labeled_df = df.copy()
    labeled_df[LABEL_COLUMN] = final_mask
    return labeled_df


def save_outputs(base_dir, dataset_name, pipeline_name, labeled_df, forest_result=None):
    out_dir = Path(base_dir) / dataset_name / pipeline_name
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = out_dir / "labeled.csv"
    labeled_df.to_csv(labeled_path, index=False)

    if forest_result is not None:
        forest_result.results_df.to_csv(out_dir / "forest_results.csv", index=False)
        best_summary = forest_result.best_row.to_dict()
        summary_path = out_dir / "best_forest.json"
        summary_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    return out_dir


def save_forest_accuracy_plot(results_df, output_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        results_df["total_questions"],
        results_df["forest_accuracy"],
        s=40,
        alpha=0.7,
    )
    plt.xlabel("Total Number of Questions in Forest", fontsize=12)
    plt.ylabel("Forest Accuracy", fontsize=12)
    plt.title("Forest Accuracy vs Total Questions Across Seeds", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_pipelines_for_dataset(
    dataset_cfg,
    pipeline_cfg,
    forest_cfg,
    output_base_dir,
    run_forest=True,
):
    df = load_dataset_from_config(dataset_cfg)

    results = {}
    for pipeline_name, settings in pipeline_cfg.items():
        if pipeline_name == "original":
            labeled_df = label_original_method(
                df,
                dataset_cfg["target"],
                high_value_quantile=settings.get("high_value_quantile", 0.94),
            )
        elif pipeline_name == "syflow":
            labeled_df = label_syflow_method(
                df,
                dataset_cfg["target"],
                n_seeds=settings.get("n_seeds", 1000),
                beta=settings.get("beta", 0.5),
                lambd_div=settings.get("lambd_div", 2.0),
                top_percentile=settings.get("top_percentile", 0.01),
                max_overlap=settings.get("max_overlap", 0.95),
                min_subgroup_size=settings.get("min_subgroup_size", 50),
                max_subgroup_frac=settings.get("max_subgroup_frac", 0.90),
            )
        else:
            raise ValueError(f"Unknown pipeline '{pipeline_name}'")

        forest_result = None
        trees = None
        feature_names = None
        X = None
        y = None
        if run_forest:
            X, y, feature_names = prepare_xy(labeled_df, dataset_cfg["target"])
            forest_result = run_forest_search(
                X,
                y,
                n_seeds=forest_cfg.get("n_seeds", 1000),
                accuracy_quantile=forest_cfg.get("accuracy_quantile", 0.99),
            )
            trees, _ = rebuild_forest_for_seed(X, y, forest_result.best_seed)

        out_dir = save_outputs(
            output_base_dir, dataset_cfg["name"], pipeline_name, labeled_df, forest_result
        )
        save_kde_plot(
            labeled_df,
            dataset_cfg["target"],
            LABEL_COLUMN,
            Path(out_dir) / "kde_plot.png",
        )
        if forest_result is not None:
            save_forest_accuracy_plot(
                forest_result.results_df,
                Path(out_dir) / "accuracy_vs_questions.png",
            )
        if run_forest and trees is not None and feature_names is not None:
            save_forest_tree_artifacts(trees, X, y, feature_names, out_dir)
        results[pipeline_name] = out_dir

    return results
