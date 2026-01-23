# config_subgroups.py

class TreeClusterForestConfig:
    """
    Config for clustering-based subgroup discovery + random-forest explanations.

    This is intentionally simple; you can tune these later.
    """

    def __init__(self):
        # ------------------------------------------------------------------
        # DATA
        # ------------------------------------------------------------------

        # self.datasets = ["california", "diabetes", "bike", "wine"]
        self.datasets = ['california']

        # ------------------------------------------------------------------
        # MODULE 1: clustering-based subgroup discovery
        # ------------------------------------------------------------------
        
        #random cluster seeds
        self.n_cluster_seeds = 1000

        # K values to cycle through when we generate clustering candidates
        self.cluster_k_values = [3, 4, 5, 6, 7]

        # minimum subgroup size (both absolute and relative)
        self.min_cluster_size = 10          # absolute minimum number of rows
        self.min_cluster_frac = 0.02        # 2% of dataset

        # how many final subgroups to keep
        self.n_subgroups = 5

        # SYFLOW-style size–exceptionality trade-off
        self.alpha = 0.3                    # γ in the paper
        self.lambd = 2.0                    # regularizer weight λ
        self.use_histogram_kl = True        # use histogram KL as in Appendix E 

        self.cluster_seed = 0               # base seed for clustering

        # ------------------------------------------------------------------
        # MODULE 2: random-forest explanation of each subgroup
        # ------------------------------------------------------------------
        self.rf_n_seeds = 50                # number of candidate forests
        self.rf_n_estimators = 20
        self.rf_max_depth = 5
        self.rf_min_samples_leaf = 5
        self.rf_max_features = "sqrt"

        self.rf_test_size = 0.3             # validation split
        self.rf_accuracy_quantile = 0.8     # keep top 20% by accuracy
        self.rf_base_seed = 123             # base seed for RFs

        # to enforce a minimum positive support when training
        self.rf_min_positive = 20

        self.y_min_clusters = 2
        self.y_max_clusters = 50
        self.y_linkage = "ward"  # or "average" / "complete"
