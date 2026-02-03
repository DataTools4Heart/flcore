import bz2
import os
import shutil
import urllib.request
from typing import Tuple
import json

import numpy as np
import openml
#import torch
from pathlib import Path
import pandas as pd

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from ucimlrepo import fetch_ucirepo
import pickle


from flcore.models.xgb.utils import TreeDataset, do_fl_partitioning, get_dataloader

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]

def calculate_preprocessing_params(subset_data, subset_target, n_features=None, feature_selection_method='mutual_info'):
    """
    Calculate preprocessing parameters based on a subset of data (reference center)
    
    Args:
        subset_data: DataFrame containing the subset data
        subset_target: Series containing the target variable
        n_features: Number of features to select (None for all features)
        feature_selection_method: Method for feature selection ('mutual_info', 'f_classif', 'random_forest')
        
    Returns:
        dict: Preprocessing parameters (imputation values, mean, std, label_encoders, feature_selector)
    """
    data_copy = subset_data.copy()
    target_copy = subset_target.copy()
    
    # Calculate imputation parameters
    imputation_params = {}
    label_encoders = {}
    
    for column in data_copy.columns:
        # Handle missing values
        if data_copy[column].isna().any():
            if data_copy[column].dtype in ['float64', 'int64']:
                imputation_params[column] = data_copy[column].median()
            else:
                imputation_params[column] = data_copy[column].mode()[0] if not data_copy[column].mode().empty else 0
        
        # Store label encoders for categorical variables
        if data_copy[column].dtype == 'object':
            le = LabelEncoder()
            # Fit on non-null values only
            non_null_data = data_copy[column].dropna()
            if len(non_null_data) > 0:
                # Add 'unknown' category for unseen labels
                classes = np.append(non_null_data.astype(str).unique(), 'unknown')
                le.fit(classes)
                label_encoders[column] = le
    
    # Calculate normalization parameters for ALL columns (after conversion to numerical)
    numeric_data = data_copy.copy()
    
    # Temporarily convert categorical to numerical for normalization parameter calculation
    for column in numeric_data.columns:
        if numeric_data[column].dtype == 'object':
            # Use simple integer encoding for parameter calculation
            numeric_data[column] = pd.Categorical(numeric_data[column]).codes
        # Handle missing values temporarily for parameter calculation
        if column in imputation_params:
            numeric_data[column].fillna(imputation_params[column], inplace=True)
    
    # Convert all to numeric
    numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
    
    # Calculate normalization parameters
    normalization_params = {
        'mean': numeric_data.mean().to_dict(),
        'std': numeric_data.std().to_dict()
    }
    
    # Handle zero standard deviation
    for col, std_val in normalization_params['std'].items():
        if std_val == 0 or np.isnan(std_val):
            normalization_params['std'][col] = 1.0
    
    # Feature Selection
    feature_selector = None
    selected_features = None
    feature_scores = None
    
    if n_features is not None:
        if n_features < len(numeric_data.columns):
            # Prepare data for feature selection
            X_temp = numeric_data.fillna(numeric_data.median())
            y_temp = target_copy
            
            # Handle any remaining NaN values
            X_temp = X_temp.fillna(0)
            
            if feature_selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X_temp.shape[1]))
            elif feature_selection_method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=min(n_features, X_temp.shape[1]))
            elif feature_selection_method == 'random_forest':
                # Use Random Forest feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_temp, y_temp)
                importances = rf.feature_importances_
                indices = np.argsort(importances)[::-1]
                selected_indices = indices[:min(n_features, len(indices))]
                
                # Create a custom selector object
                class CustomSelector:
                    def __init__(self, selected_indices, feature_names):
                        self.selected_indices = selected_indices
                        self.feature_names = feature_names
                        self.scores_ = importances
                        
                    def transform(self, X):
                        if isinstance(X, pd.DataFrame):
                            return X.iloc[:, self.selected_indices]
                        else:
                            return X[:, self.selected_indices]
                            
                    def get_support(self, indices=False):
                        if indices:
                            return self.selected_indices
                        else:
                            mask = np.zeros(len(self.feature_names), dtype=bool)
                            mask[self.selected_indices] = True
                            return mask
                
                selector = CustomSelector(selected_indices, numeric_data.columns.tolist())
                feature_scores = importances
            else:
                raise ValueError("feature_selection_method must be 'mutual_info', 'f_classif', or 'random_forest'")
            
            if feature_selection_method != 'random_forest':
                selector.fit(X_temp, y_temp)
                feature_scores = selector.scores_
            
            feature_selector = selector
            selected_features = numeric_data.columns[selector.get_support()].tolist()
            
            print(f"Feature selection: Selected {len(selected_features)} most informative features")
            if feature_scores is not None:
                # Print top feature scores
                feature_importance = pd.DataFrame({
                    'feature': numeric_data.columns,
                    'score': feature_scores
                }).sort_values('score', ascending=False)
                print("Top 5 features:")
                for i, (_, row) in enumerate(feature_importance.head().iterrows()):
                    print(f"  {i+1}. {row['feature']}: {row['score']:.4f}")
    
    return {
        'imputation': imputation_params,
        'normalization': normalization_params,
        'label_encoders': label_encoders,
        'feature_selector': feature_selector,
        'selected_features': selected_features,
        'n_features': n_features
    }

def apply_preprocessing(subset_data, preprocessing_params, normalization="global"):
    """
    Apply preprocessing to a subset using pre-calculated parameters from reference center
    
    Args:
        subset_data: DataFrame to preprocess
        preprocessing_params: dict from calculate_preprocessing_params
        
    Returns:
        tuple: (preprocessed_data, feature_names)
    """
    data_copy = subset_data.copy()
    
    # Step 1: Handle missing values using reference center parameters
    for column in data_copy.columns:
        if column in preprocessing_params['imputation']:
            missing_mask = data_copy[column].isna()
            if missing_mask.any():
                data_copy.loc[missing_mask, column] = preprocessing_params['imputation'][column]
    
    # Step 2: Convert all features to numerical using reference center label encoders
    for column in data_copy.columns:
        if column in preprocessing_params['label_encoders']:
            le = preprocessing_params['label_encoders'][column]
            # Convert to string and handle unseen labels
            encoded_values = []
            for val in data_copy[column]:
                if pd.isna(val):
                    encoded_values.append(-1)  # Special value for missing
                else:
                    str_val = str(val)
                    if str_val in le.classes_:
                        encoded_values.append(le.transform([str_val])[0])
                    else:
                        # Map unseen labels to 'unknown' class
                        encoded_values.append(le.transform(['unknown'])[0])
            data_copy[column] = encoded_values
        elif data_copy[column].dtype == 'object':
            # Fallback: use categorical codes for any remaining object columns
            data_copy[column] = pd.Categorical(data_copy[column]).codes
    
    # Ensure all data is numerical
    data_copy = data_copy.apply(pd.to_numeric, errors='coerce')
    
    # Step 3: Normalize ALL features using global parameters if enabled
    if normalization == "global":
        normalization_params = preprocessing_params['normalization']
        for column in data_copy.columns:
            if column in normalization_params['mean']:
                mean_val = normalization_params['mean'][column]
                std_val = normalization_params['std'][column]
                data_copy[column] = (data_copy[column] - mean_val) / std_val
        # print("Applied global normalization during preprocessing.")
    elif normalization == "local":
        # Calculate local normalization parameters
        local_mean = data_copy.mean()
        local_std = data_copy.std()
        for column in data_copy.columns:
            mean_val = local_mean[column]
            std_val = local_std[column] if local_std[column] != 0 else 1.0
            data_copy[column] = (data_copy[column] - mean_val) / std_val
        # print("Applied local normalization during preprocessing.")
    elif normalization is not None:
        raise ValueError("Data normalization method must be 'global', 'local', or None")
    
    # Step 4: Apply feature selection if enabled
    if preprocessing_params['feature_selector'] is not None:
        selector = preprocessing_params['feature_selector']
        data_copy = pd.DataFrame(selector.transform(data_copy), 
                               columns=preprocessing_params['selected_features'])
    
    return data_copy, data_copy.columns.tolist()

def partition_data_dirichlet(labels, num_centers, alpha=1.0):
    """
    Partition data among centers using Dirichlet distribution
    """
    unique_labels = np.unique(labels)
    n_samples = len(labels)
    n_classes = len(unique_labels)

    if not alpha:
        alpha = -1.0

    if alpha <= 0:
        # IID partitioning
        shuffled_indices = np.random.permutation(n_samples)
        center_indices = np.array_split(shuffled_indices, num_centers)
        center_indices = [indices.tolist() for indices in center_indices]
        # check lengths of each center
        center_lengths = [len(indices) for indices in center_indices]
        return center_indices
    
    # Create assignment matrix
    center_indices = [[] for _ in range(num_centers)]
    
    # For each class, distribute samples to centers using Dirichlet distribution
    for class_idx in unique_labels:
        class_mask = (labels == class_idx)
        class_indices = np.where(class_mask)[0]
        n_class_samples = len(class_indices)
        
        if n_class_samples > 0:
            # Generate Dirichlet distribution for this class
            proportions = np.random.dirichlet(np.repeat(alpha, num_centers))
            proportions = proportions / proportions.sum()
            
            # Calculate number of samples for each center
            center_samples = (proportions * n_class_samples).astype(int)
            
            # Adjust for rounding errors
            diff = n_class_samples - center_samples.sum()
            if diff > 0:
                center_samples[np.random.choice(num_centers, diff, replace=True)] += 1
            
            # Shuffle and assign indices
            np.random.shuffle(class_indices)
            ptr = 0
            for center_id in range(num_centers):
                if center_samples[center_id] > 0:
                    center_indices[center_id].extend(
                        class_indices[ptr:ptr + center_samples[center_id]]
                    )
                    ptr += center_samples[center_id]
    
    # Shuffle indices within each center
    for center_id in range(num_centers):
        np.random.shuffle(center_indices[center_id])
    
    return center_indices

def select_reference_center(all_center_data, method='largest'):
    """
    Select which center to use for calculating preprocessing parameters
    """
    if method == 'largest':
        center_sizes = [len(X) for X, y in all_center_data]
        reference_center_id = np.argmax(center_sizes)
        print(f"Selected largest center (ID: {reference_center_id}) with {center_sizes[reference_center_id]} samples")
        
    elif method == 'random':
        reference_center_id = np.random.randint(0, len(all_center_data))
        print(f"Selected random center (ID: {reference_center_id})")
    else:
        raise ValueError("Method must be 'largest' or 'random'")
    
    return reference_center_id

def aggregate_preprocessing_params(preprocessing_params_list, center_sizes, method='weighted_aggregate'):
    """
    Aggregate preprocessing parameters from multiple centers using weighted aggregation.
    
    Args:
        preprocessing_params_list: List of preprocessing parameter dictionaries from each center
        center_sizes: List of center sizes (number of samples)
        
    Returns:
        dict: Aggregated preprocessing parameters
    """
    if not preprocessing_params_list:
        raise ValueError("preprocessing_params_list cannot be empty")
    
    if "equal" in method:
        # Equal weights
        center_sizes = [1 for _ in center_sizes]
        print("Using equal weights for aggregation of preprocessing parameters.")
    
    total_size = sum(center_sizes)
    weights = [size / total_size for size in center_sizes]
    
    aggregated = {
        'imputation': {},
        'normalization': {'mean': {}, 'std': {}},
        'label_encoders': {},
        'feature_selector': None,
        'selected_features': [],
        'n_features': preprocessing_params_list[0]['n_features']  # Assume same for all
    }
    
    # Collect all columns
    all_columns = set()
    for params in preprocessing_params_list:
        all_columns.update(params['imputation'].keys())
        all_columns.update(params['normalization']['mean'].keys())
        all_columns.update(params['label_encoders'].keys())
    
    # Aggregate imputation
    for col in all_columns:
        numeric_values = []
        categorical_values = []
        weights_num = []
        weights_cat = []
        for params, weight in zip(preprocessing_params_list, weights):
            if col in params['imputation']:
                value = params['imputation'][col]
                if isinstance(value, (int, float)) and not pd.isna(value):
                    numeric_values.append(value)
                    weights_num.append(weight)
                else:
                    categorical_values.append(value)
                    weights_cat.append(weight)
        
        if numeric_values:
            # Weighted mean for numeric
            aggregated['imputation'][col] = sum(v * w for v, w in zip(numeric_values, weights_num)) / sum(weights_num)
        elif categorical_values:
            # Most frequent for categorical (simple mode)
            from collections import Counter
            counter = Counter(categorical_values)
            aggregated['imputation'][col] = counter.most_common(1)[0][0]
    
    # Aggregate normalization
    for col in all_columns:
        means = []
        stds = []
        weights_norm = []
        for params, weight in zip(preprocessing_params_list, weights):
            if col in params['normalization']['mean']:
                means.append(params['normalization']['mean'][col])
                stds.append(params['normalization']['std'][col])
                weights_norm.append(weight)
        
        if means:
            global_mean = sum(m * w for m, w in zip(means, weights_norm)) / sum(weights_norm)
            aggregated['normalization']['mean'][col] = global_mean
            
            # Calculate global std: sqrt( sum(w_i * var_i) + sum(w_i * (mean_i - global_mean)^2) )
            variances = [s ** 2 for s in stds]
            weighted_var_sum = sum(v * w for v, w in zip(variances, weights_norm))
            mean_diff_sq = [(m - global_mean) ** 2 for m in means]
            weighted_mean_var = sum(md * w for md, w in zip(mean_diff_sq, weights_norm))
            global_var = weighted_var_sum + weighted_mean_var
            global_std = np.sqrt(global_var) if global_var > 0 else 1.0
            aggregated['normalization']['std'][col] = global_std
    
    # For label_encoders, take from the largest center (simplest approach)
    max_size_idx = center_sizes.index(max(center_sizes))
    aggregated['label_encoders'] = preprocessing_params_list[max_size_idx]['label_encoders'].copy()
    
    # Aggregate selected_features by frequency
    if preprocessing_params_list[0]['selected_features']:
        from collections import Counter
        feature_counts = Counter()
        for params, weight in zip(preprocessing_params_list, weights):
            for feature in params['selected_features']:
                feature_counts[feature] += weight
        
        # Select top n_features most frequent
        n_features = aggregated['n_features']
        if n_features:
            selected = [feat for feat, _ in feature_counts.most_common(n_features)]
            aggregated['selected_features'] = selected
    
    return aggregated

def prepare_dataset_old(X, y, center_id, config, center_indices=None):
        """
        Load and preprocess raw dataset for federated learning with feature selection
        
        This function will extract the following config values:
            center_id: Identifier for the federated node
            num_centers: Total number of federated centers
            alpha: Dirichlet concentration parameter for data partitioning
            reference_method: How to select reference center ('largest' or 'random')
            global_preprocessing_params: Precomputed parameters (if None, will calculate)
            n_features: Number of features to select (None for all features)
            feature_selection_method: Method for feature selection
            
        Returns:
            tuple: X_train, y_train, X_test, y_test
        """

        num_centers = config.get("num_clients", 5)
        alpha = config.get("dirichlet_alpha", 1.0)
        reference_method = config.get("reference_center_method", "largest")
        global_preprocessing_params = None
        n_features = config.get("n_features", 20)
        feature_selection_method = config.get("feature_selection_method", "mutual_info")
        normalization_method = config.get("data_normalization", "global")

        np.random.seed(42)

        # Convert target to binary classification if needed
        if y.nunique() > 2:
            y_binary = (y > y.median()).astype(int)
        else:
            y_binary = y
        
        if not center_indices:
            # Partition data using Dirichlet distribution
            all_center_indices = partition_data_dirichlet(y_binary.values, num_centers, alpha)
        else:
            all_center_indices = center_indices

        # Get all center data for reference selection
        all_center_data = []
        for i in range(num_centers):
            if i < len(all_center_indices) and len(all_center_indices[i]) > 0:
                X_center = X.iloc[all_center_indices[i]]
                all_center_data.append((X_center, y_binary.iloc[all_center_indices[i]]))
            else:
                all_center_data.append((pd.DataFrame(), pd.Series()))
        
        # Calculate or use global preprocessing parameters
        if global_preprocessing_params is None:
            if aggregation_method == 'reference':
                # Select reference center and calculate parameters
                reference_center_id = select_reference_center(all_center_data, reference_method)
                X_reference = all_center_data[reference_center_id][0]
                y_reference = all_center_data[reference_center_id][1]
                
                if len(X_reference) == 0:
                    # Fallback: use full dataset if reference center is empty
                    X_reference = X
                    y_reference = y_binary
                    print("Warning: Reference center empty, using full dataset for preprocessing parameters")
                
                global_preprocessing_params = calculate_preprocessing_params(
                    X_reference, y_reference, n_features=n_features, feature_selection_method=feature_selection_method
                )
                print("Calculated global preprocessing parameters using reference center")
            elif aggregation_method == 'weighted_aggregate':
                # Calculate parameters for each center and aggregate
                preprocessing_params_list = []
                center_sizes = []
                for X_center, y_center in all_center_data:
                    if len(X_center) > 0:
                        params = calculate_preprocessing_params(
                            X_center, y_center, n_features=n_features, feature_selection_method=feature_selection_method
                        )
                        preprocessing_params_list.append(params)
                        center_sizes.append(len(X_center))
                
                if preprocessing_params_list:
                    global_preprocessing_params = aggregate_preprocessing_params(preprocessing_params_list, center_sizes)
                    print("Calculated global preprocessing parameters using weighted aggregation")
                else:
                    # Fallback
                    global_preprocessing_params = calculate_preprocessing_params(
                        X, y_binary, n_features=n_features, feature_selection_method=feature_selection_method
                    )
                    print("Warning: No valid centers, using full dataset for preprocessing parameters")
            else:
                raise ValueError("aggregation_method must be 'reference' or 'weighted_aggregate'")
        
        if center_id is not None:
            # Get indices for the requested center
            if center_id >= len(all_center_indices) or len(all_center_indices[center_id]) == 0:
                raise ValueError(f"Center ID {center_id} has no data assigned")
            
            center_indices = all_center_indices[center_id]
            X_center = X.iloc[center_indices].reset_index(drop=True)
            y_center = y.iloc[center_indices].reset_index(drop=True)
        else:
            # Use full dataset if no center_id specified
            X_center = X
            y_center = y

        # Split into train/test for this center
        if len(X_center) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_center, y_center, test_size=0.2, random_state=42, stratify=y_center
            )
        else:
            X_train, y_train = X_center, y_center
            X_test, y_test = X_center.iloc[:0], y_center.iloc[:0]
        
        # Apply GLOBAL preprocessing parameters to both train and test sets
        X_train_processed, feature_names = apply_preprocessing(X_train, global_preprocessing_params, normalization=normalization_method)
        X_test_processed, _ = apply_preprocessing(X_test, global_preprocessing_params, normalization=normalization_method)

        # shuffle the training data
        X_train_processed, y_train = shuffle(X_train_processed, y_train)

        return X_train_processed, y_train, X_test_processed, y_test

def prepare_dataset(X, y, center_id, config, center_indices=None):
    """
    Load and preprocess raw dataset for federated learning with feature selection
    
    This function will extract the following config values:
        center_id: Identifier for the federated node
        num_centers: Total number of federated centers
        alpha: Dirichlet concentration parameter for data partitioning
        reference_method: How to select reference center ('largest' or 'random')
        aggregation_method: How to aggregate preprocessing params ('reference' or 'weighted_aggregate')
        global_preprocessing_params: Precomputed parameters (if None, will calculate)
        n_features: Number of features to select (None for all features)
        feature_selection_method: Method for feature selection
        
    Returns:
        tuple: X_train, y_train, X_test, y_test
    """

    num_centers = config.get("num_clients", 5)
    alpha = config.get("dirichlet_alpha", 1.0)
    reference_method = config.get("reference_center_method", "largest")
    preprocessing_method = config.get("data_preprocessing_method", "reference")
    global_preprocessing_params = None
    n_features = config.get("n_features", 20)
    feature_selection_method = config.get("feature_selection_method", "mutual_info")
    normalization_method = config.get("data_normalization", "global")

    np.random.seed(42)

    # Convert target to binary classification if needed
    if y.nunique() > 2:
        y_binary = (y > y.median()).astype(int)
    else:
        y_binary = y
    
    if not center_indices:
        # Partition data using Dirichlet distribution
        all_center_indices = partition_data_dirichlet(y_binary.values, num_centers, alpha)
    else:
        all_center_indices = center_indices

    # Get all center data for reference selection
    all_center_data = []
    for i in range(num_centers):
        if i < len(all_center_indices) and len(all_center_indices[i]) > 0:
            X_center = X.iloc[all_center_indices[i]]
            all_center_data.append((X_center, y_binary.iloc[all_center_indices[i]]))
        else:
            all_center_data.append((pd.DataFrame(), pd.Series()))
    
    # Calculate or use global preprocessing parameters
    if global_preprocessing_params is None:
        if preprocessing_method == 'reference':
            # Select reference center and calculate parameters
            reference_center_id = select_reference_center(all_center_data, reference_method)
            X_reference = all_center_data[reference_center_id][0]
            y_reference = all_center_data[reference_center_id][1]
            
            if len(X_reference) == 0:
                # Fallback: use full dataset if reference center is empty
                X_reference = X
                y_reference = y_binary
                print("Warning: Reference center empty, using full dataset for preprocessing parameters")
            
            global_preprocessing_params = calculate_preprocessing_params(
                X_reference, y_reference, n_features=n_features, feature_selection_method=feature_selection_method
            )
        elif "aggregate" in preprocessing_method:
            # Calculate parameters for each center and aggregate
            preprocessing_params_list = []
            center_sizes = []
            for X_center, y_center in all_center_data:
                if len(X_center) > 0:
                    params = calculate_preprocessing_params(
                        X_center, y_center, n_features=n_features, feature_selection_method=feature_selection_method
                    )
                    preprocessing_params_list.append(params)
                    center_sizes.append(len(X_center))
            
            if preprocessing_params_list:
                global_preprocessing_params = aggregate_preprocessing_params(preprocessing_params_list, center_sizes, method=preprocessing_method)
            else:
                # Fallback
                global_preprocessing_params = calculate_preprocessing_params(
                    X, y_binary, n_features=n_features, feature_selection_method=feature_selection_method
                )
                print("Warning: No valid centers, using full dataset for preprocessing parameters")
        else:
            raise ValueError("aggregation_method must be 'reference', 'equal_aggregate' or 'weighted_aggregate'")
        
        print("Calculated global preprocessing parameters using", preprocessing_method)
    
    if center_id is not None:
        # Get indices for the requested center
        if center_id >= len(all_center_indices) or len(all_center_indices[center_id]) == 0:
            raise ValueError(f"Center ID {center_id} has no data assigned")
        
        center_indices = all_center_indices[center_id]
        X_center = X.iloc[center_indices].reset_index(drop=True)
        y_center = y.iloc[center_indices].reset_index(drop=True)
    else:
        # Use full dataset if no center_id specified
        X_center = X
        y_center = y

    # Split into train/test for this center
    if len(X_center) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_center, y_center, test_size=0.2, random_state=42, stratify=y_center
        )
    else:
        X_train, y_train = X_center, y_center
        X_test, y_test = X_center.iloc[:0], y_center.iloc[:0]
    
    # Apply GLOBAL preprocessing parameters to both train and test sets
    X_train_processed, feature_names = apply_preprocessing(X_train, global_preprocessing_params, normalization=normalization_method)
    X_test_processed, _ = apply_preprocessing(X_test, global_preprocessing_params, normalization=normalization_method)

    # shuffle the training data
    X_train_processed, y_train = shuffle(X_train_processed, y_train)

    return X_train_processed, y_train, X_test_processed, y_test
   
def load_mnist(center_id=None, num_splits=5):
    """Loads the MNIST dataset using OpenML.
    OpenML dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    # print(X.shape)
    # print(y.shape)
    # print(y[0])
    # First 60000 samples consist of the train set
    # x_train, y_train = X[:60000], y[:60000]
    # x_train, y_train = X[:1000], y[:1000]
    # # x_test, y_test = X[60000:], y[60000:]
    # x_test, y_test = X[1000:], y[1000:]
    x_train = X
    y_train = y

    if center_id != None:
        # Split the data
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            if i + 1 != center_id:
                continue
            x_train, y_train = X[train_index], y[train_index]
            x_train, x_test, y_train, y_test = train_test_split(
                x_train, y_train, test_size=0.2, random_state=42
            )
            print(f"Loaded subset of MNIST with fold {i+1} out of {num_splits}.")
    else:
        x_train, y_train = X[:60000], y[:60000]
        x_test, y_test = X[60000:], y[60000:]

    # y_train = np.array(np.array(y_train, dtype=bool), dtype=float)
    # y_test = np.array(np.array(y_test, dtype=bool), dtype=float)
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    return (x_train, y_train), (x_test, y_test)

def load_cvd(data_path, center_id, config) -> Dataset:
    id = center_id

    code_id = "f_eid"
    code_outcome = "Eval"

    data = pd.read_csv(os.path.join(data_path, "data_centerAll.csv"))
    X_data = data.drop([code_id, code_outcome], axis=1)
    y_data = data[code_outcome]

    X_train_processed, y_train, X_test_processed, y_test = prepare_dataset(X_data, y_data, center_id, config)

    return (X_train_processed, y_train), (X_test_processed, y_test)

def load_ukbb_cvd(data_path, center_id, config) -> Dataset:
    """
    Load UKBB CVD mortality dataset

    Args:
        data_path: Path to the dataset
        center_id: ID of the center to load
        config: Configuration dictionary

    """
    data_path = os.path.join(data_path, "CVDMortalityData.csv")
    data = pd.read_csv(data_path)

    center_key = 'f.54.0.0'
    patient_key = 'f.eid'
    label_key = 'label'

    #Create a list of lists for each center_key with row indexes from that center
    center_keys = sorted(list(data[center_key].unique()))
    # convert to list of ints
    center_keys = set(int(center) for center in center_keys)
    center_indices = []
    for center in center_keys:
        center_indices.append(data.loc[(data[center_key] == center)].index.tolist())

    X = data.drop([label_key, center_key, patient_key], axis=1)
    y = data[label_key]

    X_train, y_train, X_test, y_test = prepare_dataset(X, y, center_id, config, center_indices)

    # print("Center ", center_id, "with ", len(X_train), " samples, of which positive samples are ", len(X_train.loc[y_train == 1]))

    # center_names = ['Bristol', 'Newcastle', 'Oxford', 'Stockport (pilot)', 'Reading',
    #                 'Middlesborough', 'Leeds', 'Liverpool', 'Nottingham', 'Glasgow', 'Croydon',
    #                 'Hounslow', 'Barts', 'Edinburgh', 'Birmingham', 'Manchester', 'Cardiff',
    #                 'Stoke', 'Bury', 'Sheffield', 'Swansea', 'Wrexham']
    # center_keys = [2, 13, 15, 18, 16, 12, 9, 10, 14, 7, 5, 8, 0, 6, 1, 11, 4, 19, 3, 17, 20, 21]
    # center_dict = dict(zip(center_keys, center_names))
    # # sort dictionary and convert to list
    # center_dict = dict(sorted(center_dict.items()))
    # center_dict = list(center_dict.values())
    # print(center_dict)

    return (X_train, y_train), (X_test, y_test)

def load_kaggle_hf(data_path, center_id, config) -> Dataset:
    """
    Load Kaggle Heart Failure dataset for federated learning using prepare_dataset
    
    Args:
        data_path: Path to the dataset
        center_id: ID of the center (0: cleveland, 1: hungarian, 2: va, 3: switzerland, None: all)
        config: Configuration dictionary
        
    Returns:
        tuple: ((X_train, y_train), (X_test, y_test))
    """
    
    file_name = os.path.join(data_path, "kaggle_hf.csv")
    data = pd.read_csv(file_name)
    
    # Define centers
    centers = ['cleveland', 'hungarian', 'va', 'switzerland']
    
    # Map center_id to index
    center_id_mapped = None
    if center_id is not None:
        if center_id == 0:
            center_id_mapped = 0  # cleveland
        elif center_id == 1:
            center_id_mapped = 1  # hungarian
        elif center_id == 2:
            center_id_mapped = 2  # va
        elif center_id == 3:
            center_id_mapped = 3  # switzerland
        else:
            # print(f"Invalid center id: {center_id}", type(center_id))
            raise ValueError(f"Invalid center id: {center_id}")
    
    # Create center_indices
    center_indices = []
    for center in centers:
        indices = data.loc[data['data_center'] == center].index.tolist()
        center_indices.append(indices)
    
    # Prepare X and y
    X = data.drop(['HeartDisease', 'data_center'], axis=1)
    y = data['HeartDisease']
    
    X_train_processed, y_train, X_test_processed, y_test = prepare_dataset(X, y, center_id_mapped, config, center_indices)
    
    return (X_train_processed, y_train), (X_test_processed, y_test)

def load_libsvm(config, center_id=None, task_type="BINARY"):
    # ## Manually download and load the tabular dataset from LIBSVM data
    # Datasets can be downloaded from LIBSVM Data: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
    CLASSIFICATION_PATH = os.path.join("dataset", "binary_classification")
    REGRESSION_PATH = os.path.join("dataset", "regression")

    if not os.path.exists(CLASSIFICATION_PATH):
        os.makedirs(CLASSIFICATION_PATH)
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",
            f"{os.path.join(CLASSIFICATION_PATH, 'cod-rna')}",
        )
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t",
            f"{os.path.join(CLASSIFICATION_PATH, 'cod-rna.t')}",
        )
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r",
            f"{os.path.join(CLASSIFICATION_PATH, 'cod-rna.r')}",
        )
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2",
            f"{os.path.join(CLASSIFICATION_PATH, 'ijcnn1.t.bz2')}",
        )
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2",
            f"{os.path.join(CLASSIFICATION_PATH, 'ijcnn1.tr.bz2')}",
        )
        for filepath in os.listdir(CLASSIFICATION_PATH):
            if filepath[-3:] == "bz2":
                abs_filepath = os.path.join(CLASSIFICATION_PATH, filepath)
                with bz2.BZ2File(abs_filepath) as fr, open(
                    abs_filepath[:-4], "wb"
                ) as fw:
                    shutil.copyfileobj(fr, fw)

    if not os.path.exists(REGRESSION_PATH):
        os.makedirs(REGRESSION_PATH)
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001",
            f"{os.path.join(REGRESSION_PATH, 'eunite2001')}",
        )
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001.t",
            f"{os.path.join(REGRESSION_PATH, 'eunite2001.t')}",
        )
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2",
            f"{os.path.join(REGRESSION_PATH, 'YearPredictionMSD.bz2')}",
        )
        urllib.request.urlretrieve(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2",
            f"{os.path.join(REGRESSION_PATH, 'YearPredictionMSD.t.bz2')}",
        )
        for filepath in os.listdir(REGRESSION_PATH):
            if filepath[-3:] == "bz2":
                abs_filepath = os.path.join(REGRESSION_PATH, filepath)
                with bz2.BZ2File(abs_filepath) as fr, open(
                    abs_filepath[:-4], "wb"
                ) as fw:
                    shutil.copyfileobj(fr, fw)

    binary_train = ["cod-rna.t", "cod-rna", "ijcnn1.t"]
    binary_test = ["cod-rna.r", "cod-rna.t", "ijcnn1.tr"]
    reg_train = ["eunite2001", "YearPredictionMSD"]
    reg_test = ["eunite2001.t", "YearPredictionMSD.t"]

    # Select the downloaded training and test dataset
    if task_type == "BINARY":
        dataset_path = "dataset/binary_classification/"
        train = binary_train[0]
        test = binary_test[0]
    elif task_type == "REG":
        dataset_path = "dataset/regression/"
        train = reg_train[0]
        test = reg_test[0]

    data_train = load_svmlight_file(dataset_path + train, zero_based=False)
    data_test = load_svmlight_file(dataset_path + test, zero_based=False)

    print("Task type selected is: " + task_type)
    print("Training dataset is: " + train)
    print("Test dataset is: " + test)

    X_train = data_train[0].toarray()
    y_train = data_train[1]
    X_test = data_test[0].toarray()
    y_test = data_test[1]

    if task_type == "BINARY":
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

    num_clients = config["num_clients"]

    if center_id != None:
        trainset = TreeDataset(
            np.array(X_train, copy=True), np.array(y_train, copy=True)
        )
        testset = TreeDataset(np.array(X_test, copy=True), np.array(y_test, copy=True))
        trainloaders, valloaders, testloader = do_fl_partitioning(
            trainset,
            testset,
            batch_size="whole",
            pool_size=num_clients,
            val_ratio=0.0,
        )
        X_train, y_train = [], []
        print(f"ID: {center_id}")
        for sample in trainloaders[center_id - 1]:
            X_train.extend(sample[0].numpy())
            y_train.extend(sample[1].numpy())
            # y_train.extend(sample[1].numpy()/2.0 + 0.5)

        # X_test, y_test = [], []
        # for sample in valloaders[center_id-1]:
        #     X_test.extend(sample[0].numpy())
        #     y_test.extend(sample[1].numpy()/2.0 + 0.5)

        # print(len(X_train))
        # print(len(y_train))
        # print(X_train[0])
        # print(y_train)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # print(X_train.shape)
        # print(y_train.shape)

    train_unique = np.unique(y_train, return_counts=True)
    test_unique = np.unique(y_test, return_counts=True)
    # print(np.unique(y_train, return_counts=True))
    # print(np.unique(y_test, return_counts=True))
    train_max_acc = train_unique[1][0] / len(y_train)
    test_max_acc = test_unique[1][0] / len(y_test)
    # print(train_max_acc)
    # print(test_max_acc)
    return (X_train, y_train), (X_test, y_test)

def std_normalize(col, mean, std):
    return (col - mean) / std

def iqr_normalize(col, Q1, Q2, Q3):
    return (col - Q2) / (Q3 - Q1)

def min_max_normalize(col, min_val, max_val):
    return (col - min_val) / (max_val - min_val)

def load_dt4h(config,id):
    metadata = Path(config['metadata_file'])
    with open(metadata, 'r') as file:
        metadata = json.load(file)

    data_file = Path(config['data_file'])
    dat = pd.read_parquet(data_file)

    dat_len = len(dat)
    # Numerical variables
    numeric_columns_non_zero = {}
    for feat in metadata["entries"][0]["featureSet"]["features"]:
        if feat["dataType"] == "NUMERIC" and feat["statistics"]["numOfNotNull"] != 0:
            # statistic keys = ['Q1', 'avg', 'min', 'Q2', 'max', 'Q3', 'numOfNotNull']
            numeric_columns_non_zero[feat["name"]] = (
                feat["statistics"]["Q1"],
                feat["statistics"]["avg"],
                feat["statistics"]["min"],
                feat["statistics"]["Q2"],
                feat["statistics"]["max"],
                feat["statistics"]["Q3"],
                feat["statistics"]["numOfNotNull"],
            )

    for col, (q1,avg,mini,q2,maxi,q3,numOfNotNull) in numeric_columns_non_zero.items():
        if col in dat.columns:
            if config["normalization_method"] == "IQR":
               dat[col] = iqr_normalize(dat[col], q1,q2,q3 )
            elif config["normalization_method"] == "STD":
                pass # no std found in data set
            elif config["normalization_method"] == "MIN_MAX":
               dat[col] = min_max_normalize(col, mini, maxi)
    tipos=[]
    map_variables = {}
    for feat in metadata["entries"][0]["featureSet"]["features"]:
        tipos.append(feat["dataType"])
        if feat["dataType"] == "NOMINAL" and feat["statistics"]["numOfNotNull"] != 0:
            num_cat = len(feat["statistics"]["valueset"])
            map_cat = {}
            for ind, cat in enumerate(feat["statistics"]["valueset"]):
                map_cat[cat] = ind
            map_variables[feat["name"]] = map_cat
    for col,mapa in map_variables.items():
        dat[col] = dat[col].map(mapa)
    
    dat[map_variables.keys()].dropna()
    
    tipos=[]
    map_variables = {}
    boolean_map = {np.bool_(False) :0, np.bool_(True):1, "False":0,"True":1}
    for feat in metadata["entries"][0]["featureSet"]["features"]:
        tipos.append(feat["dataType"])
        if feat["dataType"] == "BOOLEAN" and feat["statistics"]["numOfNotNull"] != 0:
            map_variables[feat["name"]] = boolean_map
    for col,mapa in map_variables.items():
        dat[col] = dat[col].map(boolean_map)
    
    dat[map_variables.keys()].dropna()

    """    # Print statistics
    for i in dat.keys():
        maxim = dat[i].max()
        minim = dat[i].min()
        mean = dat[i].mean()
        estd = dat[i].std()
        print(f"Column: {i}")
        print(f"  Maximum:          {maxim:10.2f}")
        print(f"  Minimum:          {minim:10.2f}")
        print(f"  Mean:             {mean:10.2f}")
        print(f"  Std dev:          {estd:10.2f}")
        print("-" * 40)
    """

    dat_shuffled = dat.sample(frac=1).reset_index(drop=True)

    target_labels = config["target_label"]
    train_labels = config["train_labels"]
    data_train = dat_shuffled[train_labels] #.to_numpy()
    data_target = dat_shuffled[target_labels] #.to_numpy()

    X_train = data_train[:int(dat_len*config["train_size"])]
    y_train = data_target[:int(dat_len*config["train_size"]):].iloc[:, 0]

    X_test = data_train[int(dat_len*config["train_size"]):]
    y_test = data_target[int(dat_len*config["train_size"]):].iloc[:, 0]
    return (X_train, y_train), (X_test, y_test)

def load_diabetes(center_id, config):
    """
    Load and preprocess diabetes dataset for federated learning with feature selection
    
    Args:
        center_id: Identifier for the federated node
        num_centers: Total number of federated centers
        alpha: Dirichlet concentration parameter for data partitioning
        reference_method: How to select reference center ('largest' or 'random')
        global_preprocessing_params: Precomputed parameters (if None, will calculate)
        n_features: Number of features to select (None for all features)
        feature_selection_method: Method for feature selection
        
    Returns:
        tuple: ((X_train, y_train), (X_test, y_test), preprocessing_params)
    """

    dataset_file = "dataset/cdc_diabetes_health_indicators.pkl"
    if os.path.exists(dataset_file):
        # Load from pickle
        with open(dataset_file, 'rb') as f:
            cdc_diabetes_health_indicators = pickle.load(f)
    else:
        # Download the dataset
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891).data
        # save as pickle for faster loading next time
        dataset = {"features": cdc_diabetes_health_indicators.features, "targets": cdc_diabetes_health_indicators.targets}
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)

    # Get features and target
    X = cdc_diabetes_health_indicators['features']
    y = cdc_diabetes_health_indicators['targets']

    # convert y to a pandas Series for easier handling
    y = pd.Series(y.values.flatten())

    # # # # Use fraction of data for faster testing (optional)
    if not config['num_clients'] == 1:
        fraction = 1.0
        # Sample indices first, then select from both X and y
        sampled_indices = X.sample(frac=fraction, random_state=42).index
        X = X.loc[sampled_indices].reset_index(drop=True)
        y = y.loc[sampled_indices].reset_index(drop=True)
    
    X_train_processed, y_train, X_test_processed, y_test = prepare_dataset(X, y, center_id, config)

    return (X_train_processed, y_train), (X_test_processed, y_test)


def cvd_to_torch(config):
    pass
def mnist_to_torch(config):
    pass
def kaggle_to_torch(config):
    pass
def libsvm_to_torch(config):
    pass

"""
def custom_to_torch(config):
    data_file = config["data_file"]
    # Base function, modify according with konstantinos especifications:
    ext = data_file.split(".")[-1]
    nome = data_file.split("/")[-1].split(".")[0]
    if ext == "pqt" or ext == "parquet":
        dat = pd.read_parquet(data_file)
    elif ext == "csv":
        dat = pd.read_csv(data_file)
    keys = list(dat.keys())
    data_set = []
    for i in range(len(dat)):
        temp = {}
        for j in keys:
            temp[j] = dat.iloc[i][j]
        data_set.append(temp)
    # Maybe we have to add the path too
    torch.save(data_set,config["data_path"]+nome+".pt")
# x_train y x_test : (n_samples_train, n_features)
# y_train y y_test : (n_samples_train,)

def convert_dataset(config):
    if config["dataset"] == "mnist":
        mnist_to_torch(config["num_clients"])
    elif config["dataset"] == "cvd":
        cvd_to_torch(config["data_path"], id)
    elif config["dataset"] == "kaggle_hf":
        kaggle_to_torch(config["data_path"], id)
    elif config["dataset"] == "libsvm":
        libsvm_to_torch(config, id)
    elif config["dataset"] == "custom":
        custom_to_torch(config)
    else:
        raise ValueError("Invalid dataset name")
"""

def load_dataset(config, id=None):
    if config["dataset"] == "mnist":
        return load_mnist(id, config["num_clients"])
    elif config["dataset"] == "cvd":
        return load_cvd(config["data_path"], id, config)
    elif config["dataset"] == "ukbb_cvd":
        return load_ukbb_cvd(config["data_path"], id, config)
    elif config["dataset"] == "kaggle_hf":
        return load_kaggle_hf(config["data_path"], id, config)
    elif config["dataset"] == "diabetes":
        return load_diabetes(id, config)
    elif config["dataset"] == "libsvm":
        return load_libsvm(config, id)
    elif config["dataset"] == "dt4h_format":
        return load_dt4h(config, id)
    else:
        raise ValueError("Invalid dataset name")

def get_stratifiedPartitions(n_splits,test_size, random_state):
    sss = StratifiedShuffleSplit(n_splits=n_splits,test_size=test_size, random_state=random_state)
    return sss

def split_partitions(n_splits,test_size, random_state,X_data, y_data):
    sss = get_stratifiedPartitions(n_splits,test_size, random_state)
    splits_nested = (sss.split(X_data, y_data))
    return splits_nested
