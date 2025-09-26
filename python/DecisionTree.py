class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def calc_gini_impurity(self, y):
        if len(y) == 0:
            return 0
        
        class_counts = y.value_counts()
        total_samples = len(y)
        gini = 1.0

        for count in class_counts:
            probability = count / total_samples
            gini -= probability * probability

        return gini
    
    def get_split_thresholds(self, feature_values):
        try:
            # Determine if the feature is categorical
            if isinstance(feature_values.iloc[0], str):
                return list(feature_values.unique())

            unique_values = sorted(feature_values.unique())

            if len(unique_values) <= 1:
                return []  # No splits possible

            return [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]

        except Exception as e:
            print(f"Error in get_split_thresholds: {e}")
            return []
        
    def find_best_split(self, X, y, features_to_consider):
        best_feature = None
        best_threshold = None
        best_impurity_reduction = 0

        for feature_index in features_to_consider:
            feature_values = X.iloc[:, feature_index]
            thresholds = self.get_split_thresholds(feature_values)

            for threshold in thresholds:
                if isinstance(threshold, str):
                    left_mask = feature_values == threshold
                else:
                    left_mask = feature_values <= threshold

                right_mask = ~left_mask

                # Skip if split doesn't actually separate data
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                total_samples = len(y)
                left_weight = len(left_y) / total_samples
                right_weight = len(right_y) / total_samples

                current_impurity = self.calc_gini_impurity(y)
                left_impurity = self.calc_gini_impurity(left_y)
                right_impurity = self.calc_gini_impurity(right_y)

                weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
                impurity_reduction = current_impurity - weighted_impurity

                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_feature = feature_index
                    best_threshold = threshold

                self.feature = best_feature
                self.threshold = best_threshold

        return best_feature, best_threshold, best_impurity_reduction
    
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, num_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        if self.num_features is None:
            self.num_features = self.n_features  # Use all features if not specified

        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = y.nunique()

        # Stopping criteria
        if (depth >= self.max_depth or 
            num_samples < self.min_samples_split or 
            unique_classes == 1):
            leaf_value = y.mode()[0]  # Most common class