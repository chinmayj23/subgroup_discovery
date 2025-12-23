import torch
from .syflow import *
from sklearn.preprocessing import StandardScaler
import pysubgroup as ps
from collections import namedtuple
import pandas as pd
import numpy as np
import warnings
import time
import argparse
# from .RSD.rulelist_class import MDLRuleList, reduce

##MY ADDITION - TREE BASED METHODS
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import _tree
from scipy.stats import entropy
import math
##

warnings.filterwarnings("ignore")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#       device = "mps"
verbose = False
    
def run_method(method,X,Y,alpha,config,n_subgroups,feature_names):
    subgroups = 123
    rules = 123
    if method == "syflow":
        subgroups, rules = run_syflow(X,Y,config,n_subgroups,feature_names)
    elif method == "sd-mean":
        subgroups, rules = run_sd_mean(X,Y,config,alpha,n_subgroups,feature_names)
    elif method == "sd-kl":
        subgroups, rules = run_sd_kl(X,Y,config,alpha,n_subgroups,feature_names)
    elif method == "rsd":
        subgroups, rules = run_rsd(X,Y,config,alpha,n_subgroups,feature_names)
    # === MY ADDITON - TREE BASED SYFLOW ===
    elif method == "tree_syflow":
        subgroups, rules = run_tree_syflow_corrected(X, Y, config, n_subgroups, feature_names)
    elif method == "tree_classifier":
        subgroups, rules = run_tree_classifier_subgroups(X, Y, config, n_subgroups, feature_names)
    # ======================
    elif method == "bh":
        pass
        #subgroups, rules = run_bh(X,Y,config,alpha,n_subgroups,feature_names)
    return subgroups, rules

def run_syflow(X, Y, config, n_subgroups, feature_names, return_flows=False):
    cut_points = torch.zeros((X.shape[1],2))
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    Y = scaler_y.fit_transform(Y)
    X_tensor = torch.tensor(X,dtype=torch.float64)
    Y_tensor = torch.tensor(Y,dtype=torch.float64)

    subgroups = []
    priors = []
    rules = []
    pop_flow = None
    print("Logging Hyperparameters")
    print("Alpha:",config.alpha)
    print("Lambda:",config.lambd)
    print("Temperature:",config.temperature,"\n")
    for n in range(n_subgroups):
        print("Discovering Subgroup #{}".format(n+1))
        for i in range(X.shape[1]):
            cut_points[i,0] = torch.quantile(X_tensor[:,i],0)
            cut_points[i,1] = torch.quantile(X_tensor[:,i],1)
        cut_points = torch.sort(cut_points,dim=1)[0]
        classifier = And_Finder(cut_points,temperature=config.temperature,use_weights=config.use_weights,bin_deviation=config.bin_deviation)
        flows, classifier = syflow(X_tensor,Y_tensor,classifier,flow_population=pop_flow,subgroup_priors=priors,
                                            pop_train_epochs=config.pop_train_epochs,subgroup_train_epochs=config.subgroup_train_epochs,final_fit_epochs=config.final_fit_epochs,
                                        device=device,verbose=verbose,lr_flow=config.lr_flow,alpha=config.alpha,
                                    lr_classifier=config.lr_classifier,lambd=config.lambd,config=config)
        pop_flow = flows[0]
        priors.append(flows[1])
        classifier = classifier.to(torch.device("cpu"))
        subgroup = torch.argmax(classifier(X_tensor),dim=1).detach().numpy()==1
        subgroups.append(subgroup)
        rules.append(classifier.get_rules(cut_points,scaler=scaler_x,feature_names=feature_names,X=X))
    if return_flows:
        return subgroups, rules, pop_flow, scaler_y,classifier
    return subgroups, rules

def run_sd_mean(X,Y,config,alpha,n_subgroups,feature_names):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]
    data = pd.concat([X,Y],axis=1)
    target = ps.NumericTarget("Y")
    search_space = ps.create_selectors(data, ignore=["Y"],nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask (
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=ps.StandardQFNumeric(alpha))

    result = ps.BeamSearch(beam_width=config.beam_width,beam_width_adaptive=False).execute(task)
    result.to_dataframe()
    subgroups = []
    rules = []
    for i in range(n_subgroups):
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            # parse this: "X0>=0.80" or "X0<0.80"
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                #if isinstance(val,str):
                val = convert(data,var, val)
                conditions.append((var,val,val))
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
            conditions.append((var,low,high))
            
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<=high))
        subgroups.append(subgroup_member)
    return subgroups, rules

def convert(df,var, val):
     if val.replace(".", "").isnumeric():
          return float(val)
     else:
          if str(df.dtypes[var]) == 'bool':
               return float(val=='True')
          
def run_sd_kl(X,Y,config,alpha,n_subgroups,feature_names):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]
    data = pd.concat([X,Y],axis=1)
    
    target = ps.NumericTarget("Y")
    search_space = ps.create_selectors(data, ignore=["Y"],nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask (
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=QF_WKL(alpha))

    result = ps.BeamSearch(beam_width=config.beam_width,beam_width_adaptive=False).execute(task)
    result.to_dataframe()
    subgroups = []
    rules = []
    for i in range(n_subgroups):
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            # parse this: "X0>=0.80" or "X0<0.80"
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                val = val = convert(data,var, val)
                conditions.append((var,val,val))
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
            conditions.append((var,low,high))
            
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<=high))
        subgroups.append(subgroup_member)
    return subgroups, rules


# KL divergence quality measure assuming normal distribution

class QF_WKL(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('StandardQFNumeric_parameters', ('size_sg', 'mean', "std", 'estimate'))

    def __init__(self, a, invert=False, estimator='sum'):
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size_sg', 'mean')
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, data, target):
        self.all_target_values = data[target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        data_size = len(data)
        std = np.std(self.all_target_values)
        self.dataset_statistics = QF_WKL.tpl(data_size, target_mean, std,None)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        size_sg = statistics.size_sg
        mean_sg = statistics.mean
        std_sg = statistics.std + 0.0000001
        mean_dataset = dataset.mean
        std_dataset = dataset.std
        if size_sg < 2:
            return 0
        kl = np.log2(std_dataset/std_sg) + (std_sg**2+(mean_sg-mean_dataset)**2)/(2*std_dataset**2)
        w = size_sg/dataset.size_sg
        return w**(self.a)*kl

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_values), data)
        sg_mean = 0
        sg_target_values = 0
        sg_std = 0
        if sg_size > 1:
            sg_target_values = self.all_target_values[cover_arr]
            sg_mean = np.mean(sg_target_values)
            sg_std = np.std(sg_target_values)
        return QF_WKL.tpl(sg_size, sg_mean, sg_std, None)
    
def run_rsd(X,Y,config,alpha,n_subgroups,feature_names):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()


    X = scaler_x.fit_transform(X)
    Y = scaler_y.fit_transform(Y)

    X = pd.DataFrame(X)
    X.columns = feature_names
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    target_model = "gaussian"
    task = "discovery"
    model = MDLRuleList(task = task, target_model = target_model,max_rules=n_subgroups, n_cutpoints=config.ncutpoints)
    model.fit(X, Y)
    
    subgroups = []
    for subgroup in model._rulelist.subgroups:
        subgroup_member = reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        subgroups.append(subgroup_member)
    rules = model._rulelist.get_rules()
    return subgroups, rules

def replace_feature_names(rule,feature_names):
    # replace X0<=... with feature_names[0]<=...
    for i in reversed(range(len(feature_names))):
        if "X"+str(i) in rule:
            rule = rule.replace("X"+str(i),feature_names[i])
    return rule

# ==============================================================================
# TREE SYFLOW IMPLEMENTATION
# A simplified baseline using Decision Trees + Syflow's Scoring Metric
# ==============================================================================

## =============================================================================
# They estimate the probablity density of the target using normalizing flows.
# We can either use a gaussian density or the freedman daiconis method they suggest in their evaluation (Appendix E)

# def freedman_diaconis_bins(data):
#     """Auto-selects bin width for histograms."""
#     data = np.asarray(data).ravel()
#     if data.size < 2: return np.array([data.min(), data.max()])
#     q75, q25 = np.percentile(data, [75, 25])
#     iqr = q75 - q25
#     if iqr == 0:
#         nbins = max(1, int(np.sqrt(data.size)))
#         return np.histogram_bin_edges(data, bins=nbins)
#     h = 2 * iqr / (data.size ** (1/3))
#     if h <= 0 or np.isnan(h) or np.isinf(h):
#         nbins = max(1, int(np.sqrt(data.size)))
#         return np.histogram_bin_edges(data, bins=nbins)
#     nbins = int(math.ceil((data.max() - data.min()) / h))
#     nbins = max(1, nbins)
#     return np.histogram_bin_edges(data, bins=nbins)

# def hist_kl(p_samples, q_samples, eps=1e-12):
#     """Calculates KL Divergence using histograms (Shape-aware)."""
#     p, q = np.asarray(p_samples).ravel(), np.asarray(q_samples).ravel()
#     if p.size == 0 or q.size == 0: return 0.0
#     edges = freedman_diaconis_bins(np.concatenate([p, q]))
#     p_hist, _ = np.histogram(p, bins=edges, density=True)
#     q_hist, _ = np.histogram(q, bins=edges, density=True)
#     p_hist += eps; q_hist += eps
#     return float(entropy(p_hist / p_hist.sum(), q_hist / q_hist.sum()))


# def run_tree_syflow(X, Y, config, n_subgroups, feature_names):
#     """
#     1. Fits a Decision Tree to generate candidate subgroups (Leaves).
#     2. Selects the top N subgroups using the exact SYFLOW objective:
#        Maximize: (Size^gamma * KL_Exceptionality) + (Lambda * KL_Diversity)
#     """
#     # 1. Prepare Data
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()
#     X_scaled = scaler_x.fit_transform(X)
#     Y_scaled = scaler_y.fit_transform(Y).flatten()
    
#     # Detect feature types for formatting (Boolean vs Continuous)
#     feature_types = {}
#     for i in range(X.shape[1]):
#         unique_vals = np.unique(X[:, i])
#         if len(unique_vals) <= 2: feature_types[i] = 'boolean'
#         elif np.all(np.mod(X[:, i], 1) == 0): feature_types[i] = 'integer'
#         else: feature_types[i] = 'continuous'

#     # 2. Global Statistics (Population)
#     pop_mean = np.mean(Y_scaled)
#     pop_std = np.std(Y_scaled) + 1e-6
#     n_total = len(Y_scaled)

#     # 3. Fit ONE large tree to get candidates
#     # We use a slightly deeper tree to get a good pool of candidates
#     clf = DecisionTreeRegressor(max_depth=config.tree_depth, 
#                                 min_samples_leaf=config.min_samples)
#     clf.fit(X_scaled, Y_scaled)
    
#     # 4. Extract All Candidates (Leaves)
#     leaf_ids = clf.apply(X_scaled)
#     unique_leaves = np.unique(leaf_ids)
    
#     candidates = []
#     for leaf in unique_leaves:
#         mask = (leaf_ids == leaf)
#         n_sg = np.sum(mask)
#         if n_sg < config.min_samples: continue

#         # Calculate Subgroup Stats (Gaussian Assumption)
#         sg_data = Y_scaled[mask]
#         sg_mean = np.mean(sg_data)
#         sg_std = np.std(sg_data) + 1e-6
        
#         # 1. Exceptionality Score (KL || Population)
#         # --- OPTION A: Gaussian Score ---
#         # kl_exc = np.log(pop_std / sg_std) + \
#         #          (sg_std**2 + (sg_mean - pop_mean)**2) / (2 * pop_std**2) - 0.5
                 
#         # --- OPTION B: Histogram Score (Robust, Commented) ---
#         kl_exc = hist_kl(sg_data, Y_scaled)
        
#         # 2. Size Correction
#         size_term = (n_sg / n_total) ** config.alpha
        
#         candidates.append({
#             'mask': mask,
#             'mean': sg_mean,
#             'std': sg_std,
#             'data': sg_data,
#             'base_score': size_term * kl_exc, # This is the "Weighted KL"
#             'rule': get_rule_for_leaf(clf.tree_, leaf, feature_names, scaler_x, feature_types),
#             'id': leaf
#         })

#     # 5. Iterative Selection Loop (The SYFLOW Diversity Logic)
#     final_subgroups = []
#     final_rules = []
#     selected_stats = [] # Stores (mean, std) of found subgroups to calculate diversity
#     selected_data = []
#     selected_ids = []

#     print(f"Selecting {n_subgroups} subgroups from {len(candidates)} candidates...")

#     for k in range(n_subgroups):
#         best_candidate = None
#         best_obj_val = -np.inf
        
#         # Evaluate every candidate against the current list of selected subgroups
#         for cand in candidates:
#             # Skip if already selected
#             # if cand['id'] in [r['id'] for r in final_rules]: 
#             #     continue
#             if cand['id'] in selected_ids: 
#                 continue
            
#             # Calculate Diversity (KL || Priors)
#             # Syflow maximizes: Base_Score + Lambda * Sum(KL(Current || Prior))
#             # It wants the new group to be DIFFERENT from previous ones.
#             diversity_score = 0
#             # --- OPTION A: Gaussian Diversity ---
#             # if len(selected_stats) > 0:
#             #     for (p_mean, p_std) in selected_stats:
#             #         kl_div = np.log(p_std / cand['std']) + \
#             #                  (cand['std']**2 + (cand['mean'] - p_mean)**2) / (2 * p_std**2) - 0.5
#             #         diversity_score += kl_div
#             #     diversity_score /= len(selected_stats)
            
#             # --- OPTION B: Histogram Diversity  ---
#             if len(selected_data) > 0:
#                 for p_data in selected_data:
#                     diversity_score += hist_kl(cand['data'], p_data)
#                 diversity_score /= len(selected_data)

#             # Final Syflow Objective
#             # Note: lambda is in config.lambd
#             # weighted_diversity = cand['size_term'] *diversity_score
#             weighted_diversity = diversity_score
#             total_score = cand['base_score'] + (config.lambd * weighted_diversity)
            
#             if total_score > best_obj_val:
#                 best_obj_val = total_score
#                 best_candidate = cand
        
#         # Check if we found anything
#         if best_candidate is None:
#             break
            
#         # Add to final list
#         final_subgroups.append(best_candidate['mask'])
#         final_rules.append(best_candidate['rule']) # Store ID to avoid re-selecting
#         # Add stats to "Priors" for next iteration

#         selected_stats.append((best_candidate['mean'], best_candidate['std']))
#         selected_data.append(best_candidate['data'])
        
#         # # Mark as selected (hacky way using dict ID in list)
#         # # Better: remove from candidates list or keep separate ID list
#         # # Here we just modify the final_rules check above.
#         # final_rules[-1] = {'id': best_candidate['id'], 'rule': best_candidate['rule']} 
#         selected_ids.append(best_candidate['id'])

#     # Clean up rules list to just strings
#     # final_rules = [r['rule'] for r in final_rules]

#     return final_subgroups, final_rules

# # ==============================================================================
# # HELPER: Rule Formatting
# # ==============================================================================

# def get_rule_for_leaf(tree_, leaf_id, feature_names, scaler, feature_types):
#     path = []
#     def find_path(curr_node, current_path):
#         if curr_node == leaf_id: return current_path
#         if tree_.children_left[curr_node] != _tree.TREE_LEAF:
#             p = find_path(tree_.children_left[curr_node], current_path + [(curr_node, 'left')])
#             if p: return p
#             p = find_path(tree_.children_right[curr_node], current_path + [(curr_node, 'right')])
#             if p: return p
#         return None

#     raw_path = find_path(0, [])
#     if not raw_path: return "Global Population"

#     rules = []
#     for node_idx, direction in raw_path:
#         feat_idx = tree_.feature[node_idx]
#         threshold = tree_.threshold[node_idx]
        
#         # Un-scale
#         if scaler:
#             dummy_row = np.zeros((1, scaler.n_features_in_))
#             dummy_row[0, feat_idx] = threshold
#             unscaled_thresh = scaler.inverse_transform(dummy_row)[0, feat_idx]
#         else:
#             unscaled_thresh = threshold

#         name = feature_names[feat_idx]
#         f_type = feature_types.get(feat_idx, 'continuous')

#         if f_type == 'boolean':
#             if direction == 'left': rules.append(f"¬{name}")
#             else: rules.append(f"{name}")
#         elif f_type == 'integer':
#             if direction == 'left':
#                 val = int(np.floor(unscaled_thresh))
#                 rules.append(f"{name} <= {val}")
#             else:
#                 val = int(np.ceil(unscaled_thresh))
#                 rules.append(f"{name} >= {val}")
#         else:
#             if direction == 'left': rules.append(f"{name} <= {unscaled_thresh:.2f}")
#             else: rules.append(f"{name} > {unscaled_thresh:.2f}")

#     return " ∧ ".join(rules)


###CHAT GPT IMPROVED VERSION###
def freedman_diaconis_bins(data):
    data = np.asarray(data).ravel()
    if data.size < 2:
        return np.array([data.min(), data.max()])
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        nbins = max(1, int(np.sqrt(data.size)))
        return np.histogram_bin_edges(data, bins=nbins)
    h = 2 * iqr / (data.size ** (1/3))
    if h <= 0 or np.isnan(h) or np.isinf(h):
        nbins = max(1, int(np.sqrt(data.size)))
        return np.histogram_bin_edges(data, bins=nbins)
    nbins = int(math.ceil((data.max() - data.min()) / h))
    nbins = max(1, nbins)
    return np.histogram_bin_edges(data, bins=nbins)

def hist_kl(p_samples, q_samples, eps=1e-12):
    p = np.asarray(p_samples).ravel()
    q = np.asarray(q_samples).ravel()
    if p.size == 0 or q.size == 0:
        return 0.0
    edges = freedman_diaconis_bins(np.concatenate([p, q]))
    p_hist, _ = np.histogram(p, bins=edges, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_prob = p_hist / p_hist.sum()
    q_prob = q_hist / q_hist.sum()
    return float(entropy(p_prob, q_prob))  # nat log

def gaussian_kl(p_mean, p_std, q_mean, q_std):
    """KL(N(p_mean,p_std^2) || N(q_mean,q_std^2)) (natural log)"""
    # add eps safety
    eps = 1e-12
    p_std = max(p_std, eps)
    q_std = max(q_std, eps)
    term = math.log(q_std / p_std) + (p_std**2 + (p_mean - q_mean)**2) / (2 * q_std**2) - 0.5
    return term

def get_box_rule_from_mask(X_orig, mask, feature_names=None, ignore_if_full=True, tol=1e-12):
    """
    Collapse leaf mask into one interval per feature (a 'box').
    - X_orig: original, unscaled numpy X (n, d)
    - mask: boolean array selecting samples in leaf
    - ignore_if_full: if interval equals full data range, skip that feature
    Returns: rule_text (string) and used_features indices
    """
    n, d = X_orig.shape
    mins = X_orig[mask].min(axis=0)
    maxs = X_orig[mask].max(axis=0)
    overall_min = X_orig.min(axis=0)
    overall_max = X_orig.max(axis=0)
    parts = []
    used = []
    for j in range(d):
        lo = mins[j]
        hi = maxs[j]
        # if all samples same value, treat as equality/boolean when appropriate
        if np.isclose(lo, hi, atol=tol):
            # single value — report equality
            if not (np.isclose(lo, overall_min[j]) and np.isclose(hi, overall_max[j]) and ignore_if_full):
                name = feature_names[j] if feature_names else f"X{j}"
                parts.append(f"{name} = {lo:.4g}")
                used.append(j)
            continue
        # otherwise interval
        if ignore_if_full and (lo <= overall_min[j] + tol and hi >= overall_max[j] - tol):
            # covers whole range, skip
            continue
        name = feature_names[j] if feature_names else f"X{j}"
        parts.append(f"{lo:.4g} < {name} < {hi:.4g}")
        used.append(j)
    if len(parts) == 0:
        return "TRUE", []
    return " ∧ ".join(parts), used

def run_tree_syflow_corrected(X, Y, config, n_subgroups, feature_names=None,
                              use_histogram_kl=True, removal_mode=False, verbose=False):
    """
    Corrected tree-based SYFLOW baseline with:
      - size-corrected KL scoring
      - size-weighted diversity regularizer (same size-term)
      - collapsing leaf -> one-interval-per-feature (box)
    Args:
        X: numpy array (n,d)
        Y: numpy array (n,) or (n,1)
        config: object with attributes:
            - tree_depth
            - min_samples
            - alpha (gamma in paper)
            - lambd (regularizer weight)
        n_subgroups: number to select
        use_histogram_kl: if True use hist_kl, else use gaussian_kl (fast)
        removal_mode: if True remove selected samples, else keep all (use regularizer)
    Returns:
        final_subgroups: list of boolean masks (length K)
        final_rules: list of human-readable box rules
        final_scores: numeric objective values
    """
    X = np.asarray(X)
    Y = np.asarray(Y).ravel()
    n, d = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(d)]

    # standardize (same as original SYFLOW setup)
    scaler_x = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    Ys = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()

    # global distribution stats
    global_Y = Ys
    global_mean = float(np.mean(global_Y))
    global_std = float(np.std(global_Y)) + 1e-12

    # fit a single tree to obtain leaves (candidate pool)
    clf = DecisionTreeRegressor(max_depth=getattr(config, "tree_depth", 6),
                                min_samples_leaf=getattr(config, "min_samples", max(10, n//200)),
                                random_state=getattr(config, "seed", 0))
    clf.fit(Xs, Ys)
    leaf_ids = clf.apply(Xs)
    unique_leaves = np.unique(leaf_ids)

    # build candidate list
    n_total = len(Ys)
    candidates = []
    for leaf in unique_leaves:
        mask_scaled = (leaf_ids == leaf)
        ns = mask_scaled.sum()
        if ns < getattr(config, "min_samples", 2):
            continue
        leaf_indices = np.where(mask_scaled)[0]
        leaf_Y = Ys[leaf_indices]
        # compute KL exceptionality
        if use_histogram_kl:
            kl_exc = hist_kl(leaf_Y, global_Y)
        else:
            kl_exc = gaussian_kl(np.mean(leaf_Y), np.std(leaf_Y), global_mean, global_std)
        size_term = (ns / n_total) ** getattr(config, "alpha", 0.5)  # alpha==gamma
        base_score = size_term * kl_exc
        # store data to compute diversity later (use original Y units for rule printing)
        candidates.append({
            'leaf': leaf,
            'indices': leaf_indices,
            'ns': ns,
            'leaf_Y': leaf_Y,
            'base_score': base_score,
            'size_term': size_term
        })

    if len(candidates) == 0:
        return [], [], []

    # iterative selection with regularizer
    selected_masks = []
    selected_rules = []
    selected_scores = []
    priors_Ys = []  # store Y arrays of selected subgroups (in standardized space)
    candidates_by_leaf = {c['leaf']: c for c in candidates}

    for k in range(n_subgroups):
        best = None
        best_val = -np.inf
        for c in candidates:
            leaf = c['leaf']
            if 'selected' in c and c['selected']:
                continue
            # diversity: average KL between candidate leaf and priors
            if len(priors_Ys) == 0:
                diversity = 0.0
            else:
                s = 0.0
                for priorY in priors_Ys:
                    if use_histogram_kl:
                        s += hist_kl(c['leaf_Y'], priorY)
                    else:
                        s += gaussian_kl(np.mean(c['leaf_Y']), np.std(c['leaf_Y']), np.mean(priorY), np.std(priorY))
                diversity = s / len(priors_Ys)
            # weight diversity by same size term
            weighted_div = c['size_term'] * diversity
            total_score = c['base_score'] + getattr(config, "lambd", 1.0) * weighted_div
            if total_score > best_val:
                best_val = total_score
                best = c

        if best is None:
            break

        # mark selected
        best['selected'] = True
        selected_indices = best['indices']
        # build a box-rule in ORIGINAL feature space
        rule_text, used_features = get_box_rule_from_mask(X, np.isin(np.arange(n), selected_indices),
                                                          feature_names=feature_names, ignore_if_full=True)
        selected_masks.append(np.isin(np.arange(n), selected_indices))
        selected_rules.append(rule_text)
        selected_scores.append(best_val)

        # add prior
        priors_Ys.append(best['leaf_Y'])

        # if removal_mode, remove those candidates that overlap strongly with the selected set
        if removal_mode:
            # remove samples from global pool: remove from each candidate indices list
            sel_set = set(selected_indices.tolist())
            new_candidates = []
            for c in candidates:
                if 'selected' in c and c['selected']:
                    continue
                # compute remaining indices
                other_idx = [i for i in c['indices'] if i not in sel_set]
                if len(other_idx) < getattr(config, "min_samples", 2):
                    c['selected'] = True
                    continue
                # update leaf_Y and ns and size_term and base_score
                c['indices'] = np.array(other_idx, dtype=int)
                c['ns'] = len(other_idx)
                c['leaf_Y'] = Ys[c['indices']]
                c['size_term'] = (c['ns'] / n_total) ** getattr(config, "alpha", 0.5)
                if use_histogram_kl:
                    c['base_score'] = c['size_term'] * hist_kl(c['leaf_Y'], global_Y)
                else:
                    c['base_score'] = c['size_term'] * gaussian_kl(np.mean(c['leaf_Y']), np.std(c['leaf_Y']), global_mean, global_std)
                new_candidates.append(c)
            candidates = new_candidates

    return selected_masks, selected_rules, selected_scores

def run_tree_classifier_subgroups(X, Y, config, n_subgroups, feature_names=None, return_scores=False):
    """
    Classification-based subgroup discovery:
      1) Mark "interesting" samples using a target quantile split.
      2) Train a decision tree classifier to predict interesting vs non-interesting.
      3) Rank leaves by purity * size_term and return the top subgroups.
    Args:
        X: numpy array (n, d)
        Y: numpy array (n,) or (n, 1)
        config: expects optional attrs:
            - interesting_quantile (default 0.8)
            - tree_depth (int), min_samples (int), alpha (float), seed (int)
        n_subgroups: number of subgroups to return
        return_scores: if True, also return the ranking scores
    """
    X = np.asarray(X)
    Y = np.asarray(Y).ravel()
    n, d = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(d)]

    # 1) Build binary labels for "interesting" target values
    q = getattr(config, "interesting_quantile", 0.8)
    if q <= 0 or q >= 1:
        q = 0.8
    threshold = np.quantile(Y, q)
    labels = (Y >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        # fallback to median split if quantile is degenerate
        threshold = np.median(Y)
        labels = (Y >= threshold).astype(int)

    # 2) Fit classifier on standardized features
    scaler_x = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    clf = DecisionTreeClassifier(
        max_depth=getattr(config, "tree_depth", 6),
        min_samples_leaf=getattr(config, "min_samples", 10),
        class_weight="balanced",
        random_state=getattr(config, "seed", 0),
    )
    clf.fit(Xs, labels)
    leaf_ids = clf.apply(Xs)
    unique_leaves = np.unique(leaf_ids)

    # 3) Score leaves by purity * size correction
    scored = []
    total = len(labels)
    alpha = getattr(config, "alpha", 0.5)
    min_samples = getattr(config, "min_samples", 10)
    for leaf in unique_leaves:
        mask = leaf_ids == leaf
        n_leaf = mask.sum()
        if n_leaf < min_samples:
            continue
        positive = labels[mask].sum()
        purity = positive / n_leaf
        size_term = (n_leaf / total) ** alpha
        score = purity * size_term
        rule_text, _ = get_box_rule_from_mask(X, mask, feature_names=feature_names, ignore_if_full=True)
        scored.append((score, mask, rule_text))

    if len(scored) == 0:
        return ([], [], []) if return_scores else ([], [])

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:n_subgroups]
    subgroups = [item[1] for item in top]
    rules = [item[2] for item in top]
    scores = [item[0] for item in top]
    if return_scores:
        return subgroups, rules, scores
    return subgroups, rules
