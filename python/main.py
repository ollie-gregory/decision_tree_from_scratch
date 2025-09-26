import pandas as pd

data = pd.read_csv('../loan_data.csv')

print(data.head())

def calc_gini_impurity(y):
    class_counts = y.value_counts()
    total_samples = len(y)
    gini = 1.0
    
    for count in class_counts:
        probability = count / total_samples
        gini -= probability * probability
    
    return gini

def get_split_thresholds(feature_values):

    try:
        # Determine if the feature is categorical
        if type(feature_values.iloc[0]) == str:
            return [feature for feature in feature_values.unique()]

        unique_values = sorted(feature_values.unique())

        if len(unique_values) <= 1:
            return [unique_values[0]]

        return [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]

    except Exception as e:
        print(f"Error in get_thresholds: {e}")
        return []

def find_best_split(X, y, features_to_consider):

    best_feature = None
    best_threshold = None
    best_impurity_reduction = 0
    
    for feature_index in features_to_consider:

        feature_values = X.iloc[:, feature_index]
        thresholds = get_split_thresholds(feature_values)
        
        for threshold in thresholds:

            if type(threshold) == str:
                left_mask = feature_values == threshold
            else:
                left_mask = feature_values <= threshold
            
            right_mask = ~left_mask
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            total_samples = len(y)
            left_weight = len(left_y) / total_samples
            right_weight = len(right_y) / total_samples
            
            current_impurity = calc_gini_impurity(y)
            left_impurity = calc_gini_impurity(left_y) if len(left_y) > 0 else 0
            right_impurity = calc_gini_impurity(right_y) if len(right_y) > 0 else 0
            
            weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
            impurity_reduction = current_impurity - weighted_impurity
            
            if impurity_reduction > best_impurity_reduction:
                best_impurity_reduction = impurity_reduction
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_impurity_reduction

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

best_feature, best_threshold, best_impurity_reduction = find_best_split(X, y, range(X.shape[1]))

print(f"Best Feature Index: {best_feature}")
print(f"Best Threshold: {best_threshold}")
print(f"Best Impurity Reduction: {best_impurity_reduction}")