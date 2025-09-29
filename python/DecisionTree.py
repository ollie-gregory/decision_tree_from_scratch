import pandas as pd

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
                return []

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

        return best_feature, best_threshold, best_impurity_reduction

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth=0):
        n_features = X.shape[1]
        
        # Create a leaf node if stopping criteria are met
        if (self._should_stop_splitting(X, y, depth)):
            leaf_value = self._get_leaf_value(y)
            return DecisionNode(value=leaf_value)
        
        # Find the best split
        features_to_consider = list(range(n_features))
        node = DecisionNode()
        best_feature, best_threshold, best_impurity_reduction = node.find_best_split(
            X, y, features_to_consider
        )
        
        # If no good split found or impurity reduction is too small, create leaf
        if (best_feature is None or 
            best_impurity_reduction < self.min_impurity_decrease):
            leaf_value = self._get_leaf_value(y)
            return DecisionNode(value=leaf_value)
        
        # Create the split
        feature_values = X.iloc[:, best_feature]
        if isinstance(best_threshold, str):
            left_mask = feature_values == best_threshold
        else:
            left_mask = feature_values <= best_threshold
        
        right_mask = ~left_mask
        
        # Check minimum samples per leaf
        if (left_mask.sum() < self.min_samples_leaf or 
            right_mask.sum() < self.min_samples_leaf):
            leaf_value = self._get_leaf_value(y)
            return DecisionNode(value=leaf_value)
        
        # Split the data
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        # Create and return the internal node
        return DecisionNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _should_stop_splitting(self, X, y, depth):
        # Check max depth
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        
        # Check minimum samples to split
        if len(y) < self.min_samples_split:
            return True
        
        # Check if all samples have the same class
        if len(y.unique()) == 1:
            return True
        
        return False
    
    def _get_leaf_value(self, y):
        return y.mode().iloc[0]  # Most frequent class
    
    def predict(self, X):
        return [self._predict_sample(sample) for _, sample in X.iterrows()]
    
    def _predict_sample(self, sample):
        node = self.root
        
        while node.value is None:  # While not a leaf node
            if isinstance(node.threshold, str):
                # Categorical feature
                if sample.iloc[node.feature] == node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                # Numerical feature
                if sample.iloc[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
        
        return node.value
    
    def print_tree(self, node=None, depth=0, prefix="Root: "):
        if node is None:
            node = self.root
        
        if node.value is not None:  # Leaf node
            print("  " * depth + prefix + f"Predict: {node.value}")
        else:  # Internal node
            feature_name = f"Feature_{node.feature}"
            if isinstance(node.threshold, str):
                condition = f"{feature_name} == '{node.threshold}'"
            else:
                condition = f"{feature_name} <= {node.threshold:.3f}"
            
            print("  " * depth + prefix + condition)
            
            if node.left:
                self.print_tree(node.left, depth + 1, "├─ True: ")
            if node.right:
                self.print_tree(node.right, depth + 1, "└─ False: ")