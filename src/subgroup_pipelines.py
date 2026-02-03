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
from sklearn.model_selection import train_test_split  # <--- Essential for valid accuracy
from joblib import Parallel, delayed

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
        if node_id == -1: return 0
        if children_left[node_id] == -1 and children_right[node_id] == -1: return 0
        return 1 + dfs(children_left[node_id]) + dfs(children_right[node_id])
    return int(dfs(0))

def build_and_evaluate_forest(X, y, seed: int):
    # --- FIX 1: VALIDATION SPLIT ---
    # We train on 80%, Evaluate on 20%. 
    # This ensures accuracy is not 1.0 (overfitting).
    if len(X) < 10:
        X_train, X_val, y_train, y_val = X, X, y, y
    else:
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    rng = np.random.RandomState(seed)
    max_n_trees = rng.randint(1, 11)
    max_questions = rng.randint(2, 11)

    trees = []
    tree_question_counts = []

    for _ in range(max_n_trees):
        # Bootstrap from TRAIN set
        n_train = X_train.shape[0]
        bootstrap_indices = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_train[bootstrap_indices]
        y_boot = y_train[bootstrap_indices]

        clf = DecisionTreeClassifier(
            max_leaf_nodes=max_questions + 1,
            random_state=rng.randint(0, 1000000),
        )
        clf.fit(X_boot, y_boot)
        prune_pure_subtrees(clf)
        
        trees.append(clf)
        tree_question_counts.append(count_questions(clf))

    # Evaluate on VALIDATION set
    if len(trees) == 0: 
        return {"seed": seed, "forest_accuracy": 0, "n_trees": 0, "total_questions": 0}

    pred_matrix = np.vstack([clf.predict(X_val).astype(bool) for clf in trees])
    forest_pred = np.any(pred_matrix, axis=0).astype(int)
    
    forest_accuracy = (forest_pred == y_val).mean()
    total_questions = int(np.sum(tree_question_counts))

    return {
        "seed": seed,
        "forest_accuracy": forest_accuracy,
        "n_trees": len(trees),
        "total_questions": total_questions,
    }

def rebuild_forest_for_seed(X, y, seed: int):
    # Rebuilds the forest (structure only) for visualization
    rng = np.random.RandomState(seed)
    max_n_trees = rng.randint(1, 11)
    max_questions = rng.randint(2, 11)
    trees = []
    for _ in range(max_n_trees):
        idx = rng.choice(len(X), size=len(X), replace=True)
        clf = DecisionTreeClassifier(max_leaf_nodes=max_questions+1, random_state=rng.randint(0,1000000))
        clf.fit(X[idx], y[idx])
        prune_pure_subtrees(clf)
        trees.append(clf)
    return trees, max_questions

def run_forest_search(X, y, n_seeds=1000, accuracy_quantile=0.99) -> ForestSearchResult:
    results = Parallel(n_jobs=-1)(
        delayed(build_and_evaluate_forest)(X, y, s) for s in range(n_seeds)
    )
    results_df = pd.DataFrame(results)
    
    acc_threshold = results_df["forest_accuracy"].quantile(accuracy_quantile)
    top_df = results_df[results_df["forest_accuracy"] >= acc_threshold]
    if top_df.empty: 
        top_df = results_df 

    top_sorted = top_df.sort_values(
        by=["total_questions", "forest_accuracy", "seed"],
        ascending=[True, False, True],
    )

    best_row = top_sorted.iloc[0]
    best_seed = int(best_row["seed"])

    return ForestSearchResult(results_df=results_df, best_seed=best_seed, best_row=best_row)

def label_original_method(df, target, high_value_quantile=0.94):
    y_col = df[target]
    if isinstance(y_col, pd.DataFrame): y_col = y_col.iloc[:, 0]
    y_col = pd.to_numeric(y_col, errors="coerce")
    y = y_col.values.reshape(-1, 1)

    # 1. Clustering (Original Logic)
    try:
        Z = linkage(y, method="single")
        heights = Z[:, 2]
        diffs = np.diff(heights)
        threshold = heights[np.argmax(diffs)] if len(diffs) > 0 else 0
        clusters = fcluster(Z, t=threshold, criterion="distance")
        unique, counts = np.unique(clusters, return_counts=True)
        largest = unique[np.argmax(counts)]
        interesting_mask = clusters != largest
    except:
        interesting_mask = np.zeros(len(y), dtype=bool)

    # 2. Thresholding
    # --- FIX 2: SMART THRESHOLD ---
    # If the user asks for a very low quantile (e.g. 0.05), they mean "Low Values".
    # If they ask for high (e.g. 0.94), they mean "High Values".
    # This keeps the method signature original but makes it usable for Life Expectancy.
    cutoff = y_col.quantile(high_value_quantile)
    
    if high_value_quantile < 0.5:
        # User wants BOTTOM X%
        interesting_mask |= y_col <= cutoff
    else:
        # User wants TOP X% (Original behavior)
        interesting_mask |= y_col >= cutoff

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
    # --- EXACT ORIGINAL LOGIC (No changes) ---
    y_col = df[target]
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    y_col = pd.to_numeric(y_col, errors="coerce")
    y_full = y_col.values
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
        if threshold == 0: continue

        boot_clusters = fcluster(Z, t=threshold, criterion="distance")
        unique_labels = np.unique(boot_clusters)

        for label in unique_labels:
            cluster_vals = y_boot[boot_clusters == label]
            if len(cluster_vals) == 0: continue
            
            c_min, c_max = cluster_vals.min(), cluster_vals.max()
            mask = (y_full >= c_min) & (y_full <= c_max)
            n_sub = int(mask.sum())

            if n_sub < min_subgroup_size or n_sub > (n_total * max_subgroup_frac):
                continue

            sub_y = y_full[mask]
            kl_exc = hist_kl(sub_y, y_full, global_bins)
            size_term = (n_sub / n_total) ** beta
            quality = size_term * kl_exc

            candidates.append({
                "seed": seed, "mask": mask, "y_values": sub_y,
                "size_term": size_term, "quality": quality,
                "range": (c_min, c_max), "selected": False,
            })

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
            if cand["selected"]: continue
            is_duplicate = False
            for sel in selected_candidates:
                if jaccard_similarity(cand["mask"], sel["mask"]) > max_overlap:
                    is_duplicate = True; break
            if is_duplicate: cand["selected"] = True; continue

            diversity = 0.0
            if len(selected_candidates) > 0:
                for sel in selected_candidates:
                    diversity += hist_kl(cand["y_values"], sel["y_values"], global_bins)
                diversity /= len(selected_candidates)

            total_score = cand["quality"] + (lambd_div * cand["size_term"] * diversity)
            if total_score > best_score:
                best_score = total_score; best_candidate = cand

        if best_candidate is None: break
        best_candidate["selected"] = True
        selected_candidates.append(best_candidate)

    final_mask = np.zeros(n_total, dtype=bool)
    for cand in selected_candidates:
        final_mask |= cand["mask"]

    labeled_df = df.copy()
    labeled_df[LABEL_COLUMN] = final_mask
    return labeled_df

def prepare_xy(df, target, label_column=LABEL_COLUMN):
    if label_column not in df.columns: raise ValueError(f"Label column '{label_column}' not found")
    if target not in df.columns: raise ValueError(f"Target column '{target}' not found")

    y = df[label_column].astype(int)
    feature_df = df.drop(columns=[target, label_column], errors='ignore')
    
    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty: return np.array([]), np.array([]), []
    
    X = numeric_df
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    return X.values, y.values, list(X.columns)

def save_outputs(base_dir, dataset_name, pipeline_name, labeled_df, forest_result=None):
    out_dir = Path(base_dir) / dataset_name / pipeline_name
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(out_dir / "labeled.csv", index=False)
    if forest_result is not None:
        forest_result.results_df.to_csv(out_dir / "forest_results.csv", index=False)
        if forest_result.best_row is not None:
            best_summary = forest_result.best_row.to_dict()
            (out_dir / "best_forest.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    return out_dir

def tree_to_dict(tree: DecisionTreeClassifier, feature_names):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    def recurse(node_id: int):
        if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:
            value = tree_.value[node_id][0]
            class_index = int(np.argmax(value))
            return {"leaf_prediction": int(tree.classes_[class_index]), "class_counts": value.tolist()}
        feat_name = feature_names[feature[node_id]]
        thresh = float(threshold[node_id])
        return {"question": f"{feat_name} <= {thresh}", "feature": feat_name, "threshold": thresh,
                "left": recurse(tree_.children_left[node_id]), "right": recurse(tree_.children_right[node_id])}
    return recurse(0)

def save_forest_tree_artifacts(trees, X, y, feature_names, out_dir):
    if not trees: return
    from sklearn.tree import plot_tree
    out_dir = Path(out_dir)
    for idx, tree in enumerate(trees):
        acc = float((tree.predict(X) == y).mean())
        tree_dict = tree_to_dict(tree, feature_names=feature_names)
        (out_dir / f"tree_{idx:02d}_acc_{acc:.4f}.json").write_text(json.dumps(tree_dict, indent=2), encoding="utf-8")
        plt.figure(figsize=(12, 6))
        plot_tree(tree, feature_names=feature_names, class_names=["not interesting", "interesting"], filled=True, rounded=True)
        plt.title(f"Tree {idx} (acc={acc:.4f})")
        plt.tight_layout(); plt.savefig(out_dir / f"tree_{idx:02d}_acc_{acc:.4f}.png", dpi=300); plt.close()

def save_kde_plot(df, target, label_column, output_path, title_override=None):
    if label_column not in df.columns or target not in df.columns: return
    plt.figure(figsize=(12, 7))
    sns.kdeplot(pd.to_numeric(df[target], errors="coerce"), color="blue", fill=True, alpha=0.25, label="Original", bw_adjust=1.1)
    
    int_mask = df[label_column] == True
    if int_mask.any():
        sns.kdeplot(pd.to_numeric(df.loc[int_mask, target]), color="red", fill=True, alpha=0.25, label="Interesting")
    if (~int_mask).any():
        sns.kdeplot(pd.to_numeric(df.loc[~int_mask, target]), color="green", fill=True, alpha=0.25, label="Non-Interesting")
        
    plt.title(title_override or f"KDE of {target}"); plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=300); plt.close()

def save_forest_accuracy_plot(results_df, output_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df["total_questions"], results_df["forest_accuracy"], s=40, alpha=0.7)
    plt.xlabel("Total Questions"); plt.ylabel("Accuracy"); plt.title("Forest Accuracy vs Questions")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()