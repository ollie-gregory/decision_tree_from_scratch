# CLASS DecisionNode:
    
#     // Constructor
#     CONSTRUCTOR():
#         feature = None          // Index of feature to split on (None for leaf)
#         threshold = None        // Threshold value for splitting (None for leaf)
#         left = None            // Reference to left child node (None for leaf)
#         right = None           // Reference to right child node (None for leaf)
#         prediction = None      // Class prediction (only set for leaf nodes)
#         samples = 0            // Number of training samples at this node
#         impurity = 0.0         // Impurity measure (Gini/entropy) at this node
    
#     // Core Methods
#     METHOD is_leaf():
#         RETURN left IS None AND right IS None
    
#     METHOD predict(sample):
#         IF is_leaf():
#             RETURN prediction
        
#         IF sample[feature] <= threshold:
#             RETURN left.predict(sample)
#         ELSE:
#             RETURN right.predict(sample)
    
#     METHOD set_as_leaf(class_prediction, num_samples, node_impurity):
#         prediction = class_prediction
#         samples = num_samples
#         impurity = node_impurity
#         feature = None
#         threshold = None
#         left = None
#         right = None
    
#     METHOD set_as_internal(split_feature, split_threshold, left_child, right_child, num_samples, node_impurity):
#         feature = split_feature
#         threshold = split_threshold
#         left = left_child
#         right = right_child
#         samples = num_samples
#         impurity = node_impurity
#         prediction = None
    
#     // Utility Methods
#     METHOD get_split_condition():
#         IF is_leaf():
#             RETURN "Leaf: predict " + str(prediction)
#         ELSE:
#             RETURN "feature_" + str(feature) + " <= " + str(threshold)
    
#     METHOD get_info():
#         info = "Samples: " + str(samples) + ", Impurity: " + str(impurity)
#         IF is_leaf():
#             info += ", Prediction: " + str(prediction)
#         ELSE:
#             info += ", Split: " + get_split_condition()
#         RETURN info
# 
#     // Split Finding Methods
#     METHOD find_best_split(X, y, features_to_consider):
#         best_feature = None
#         best_threshold = None
#         best_impurity_reduction = 0
        
#         FOR each feature_index IN features_to_consider:
#             FOR each threshold IN get_candidate_thresholds(X[:, feature_index]):
#                 impurity_reduction = calculate_impurity_reduction(X, y, feature_index, threshold)
                
#                 IF impurity_reduction > best_impurity_reduction:
#                     best_impurity_reduction = impurity_reduction
#                     best_feature = feature_index
#                     best_threshold = threshold
        
#         RETURN best_feature, best_threshold, best_impurity_reduction
    
#     METHOD get_candidate_thresholds(feature_values):
#         unique_values = get_unique_sorted(feature_values)
#         thresholds = []
        
#         FOR i FROM 0 TO length(unique_values) - 2:
#             threshold = (unique_values[i] + unique_values[i+1]) / 2
#             thresholds.append(threshold)
        
#         RETURN thresholds
    
#     METHOD calculate_impurity_reduction(X, y, feature_index, threshold):
#         // Split data based on threshold
#         left_mask = X[:, feature_index] <= threshold
#         right_mask = NOT left_mask
        
#         left_y = y[left_mask]
#         right_y = y[right_mask]
        
#         // Calculate weighted impurity after split
#         total_samples = length(y)
#         left_weight = length(left_y) / total_samples
#         right_weight = length(right_y) / total_samples
        
#         current_impurity = calculate_impurity(y)
#         left_impurity = calculate_impurity(left_y)
#         right_impurity = calculate_impurity(right_y)
        
#         weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
#         impurity_reduction = current_impurity - weighted_impurity
        
#         RETURN impurity_reduction
    
#     METHOD calculate_impurity(y):
#         // For Gini impurity
#         class_counts = count_classes(y)
#         total_samples = length(y)
#         gini = 1.0
        
#         FOR each count IN class_counts.values():
#             probability = count / total_samples
#             gini -= probability * probability
        
#         RETURN gini
    
#     // Main method to build this node
#     METHOD build_node(X, y, max_depth, min_samples_split, current_depth):
#         samples = length(y)
#         impurity = calculate_impurity(y)
        
#         // Base cases for leaf creation
#         IF current_depth >= max_depth OR samples < min_samples_split OR impurity == 0:
#             prediction = most_common_class(y)
#             set_as_leaf(prediction, samples, impurity)
#             RETURN
        
#         // Find best split
#         all_features = [0, 1, 2, ..., number_of_features - 1]
#         best_feature, best_threshold, best_gain = find_best_split(X, y, all_features)
        
#         // If no good split found, make leaf
#         IF best_gain <= 0:
#             prediction = most_common_class(y)
#             set_as_leaf(prediction, samples, impurity)
#             RETURN
        
#         // Split data and create children
#         left_mask = X[:, best_feature] <= best_threshold
#         right_mask = NOT left_mask
        
#         left_child = new DecisionNode()
#         right_child = new DecisionNode()
        
#         left_child.build_node(X[left_mask], y[left_mask], max_depth, min_samples_split, current_depth + 1)
#         right_child.build_node(X[right_mask], y[right_mask], max_depth, min_samples_split, current_depth + 1)
        
#         set_as_internal(best_feature, best_threshold, left_child, right_child, samples, impurity)