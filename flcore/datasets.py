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
    
    if n_features is not None and n_features < len(numeric_data.columns):
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

def apply_preprocessing(subset_data, preprocessing_params):
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
    
    # Step 3: Normalize ALL features using reference center parameters
    normalization_params = preprocessing_params['normalization']
    for column in data_copy.columns:
        if column in normalization_params['mean']:
            mean_val = normalization_params['mean'][column]
            std_val = normalization_params['std'][column]
            data_copy[column] = (data_copy[column] - mean_val) / std_val
    
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


def load_cvd(data_path, center_id=None) -> Dataset:
    id = center_id
    if center_id == 1:
        file_name = data_path+'data_center1.csv'
    elif center_id == 2:
        file_name = data_path+'data_center2.csv'
    elif center_id == 3:
        file_name = data_path+'data_center3.csv'
    else:
        file_name = data_path+'data_center3.csv'
    
    if id == None:
        # id = 'All'
        data_centers = ['All']
    else:
        data_centers = [id]

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    test_index_list = []
    train_index_list = []

    for id in data_centers:
        # file_name = os.path.join(data_path, f"data_center{id}.csv")
        # file_name = os.path.join(data_path, file_name)

        code_id = "f_eid"
        code_outcome = "Eval"

        data = pd.read_csv(file_name)
        X_data = data.drop([code_id, code_outcome], axis=1)
        y_data = data[code_outcome]
        f_eid = data[code_id]

        # Split the data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=None)
        train_index, test_index = next(sss.split(X_data, y_data))
        X_test = X_data.iloc[test_index, :]
        X_train = X_data.iloc[train_index, :]
        y_test, y_train = y_data.iloc[test_index], y_data.iloc[train_index]
        # We save the names
        f_eid.iloc[test_index]
        f_eid.iloc[train_index]

        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        train_index_list.append(train_index)
        test_index_list.append(test_index)

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    X_test = pd.concat(X_test_list)
    y_test = pd.concat(y_test_list)
    train_index = np.concatenate(train_index_list)
    test_index = np.concatenate(test_index_list)

    # Verify set difference, data centers overlap
    # print(len(train_index.tolist()))
    # print(len(test_index.tolist()))
    # train_set = set(train_index.tolist())
    # test_set = set(test_index.tolist())
    # diff = train_set.intersection(test_set)
    # print(len(train_set))
    # print(len(test_set))
    # print( len(diff) )
    # print(f"SUBSET {id}")
    # train_unique = np.unique(y_train, return_counts=True)
    # test_unique = np.unique(y_test, return_counts=True)
    # train_max_acc = train_unique[1][0]/len(y_train)
    # test_max_acc = test_unique[1][0]/len(y_test)
    # print(np.unique(y_train, return_counts=True))
    # print(np.unique(y_test, return_counts=True))
    # print(train_max_acc)
    # print(test_max_acc)

    return (X_train, y_train), (X_test, y_test)

def load_ukbb_cvd(data_path, center_id, config) -> Dataset:

    seed = config["seed"]
    data_path = os.path.join(data_path, "CVDMortalityData.csv")
    data = pd.read_csv(data_path)

    # print(len(data))

    center_key = 'f.54.0.0'
    patient_key = 'f.eid'
    label_key = 'label'

    # center_id = None
    # center_id = 1
    preprocessing_data = data.loc[(data[center_key] == 1)]
    # center_id = None
    if center_id is not None:
        center_id = center_id
        if center_id == 19:
            center_id = 21
        elif center_id == 21:
            center_id = 19
        data = data.loc[(data[center_key] == center_id)]

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

    # xx

    # for i in range(0, 23):
    #     center_data = data.loc[(data[center_key] == i)]
    #     print(f'Center ID: {i} {center_dict[i]} with {len(center_data)} samples of which positive samples are {len(center_data.loc[center_data[label_key] == 1])})')
    # xx
    # features = data.drop([label_key, center_key, patient_key], axis=1)
    # target = data[label_key]

    # print(len(data))
    # print(features.head())
    # print(f'Center ID: {center_id} with {len(data)} samples of which positive samples are {len(data.loc[data[label_key] == 1])})')
    # print(target.head())

    def get_preprocessing_params(preprocessing_data):

        data = preprocessing_data
        features = data.drop([label_key, center_key, patient_key], axis=1)
        target = data[label_key]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = seed, stratify=target)

        n_features = 40
        fs = SelectKBest(f_classif, k=n_features).fit(X_train, y_train)
        index_features = fs.get_support()
        X_train = X_train.iloc[:, index_features]

        # print(X_train.head())

        # Get the unique values of the categorical features
        col = list(X_train.columns)
        categorical_features = []
        numerical_features = []
        for i in col:
            if len(X_train[i].unique()) > 24:
                numerical_features.append(i)
            # else:
                # categorical_features.append(i)

        transformers_dict = {}

        for i in categorical_features:
            transformers_dict[i] = OrdinalEncoder()
        for i in numerical_features:
            transformers_dict[i] = StandardScaler()
        
        # df1 = data.copy(deep = True)

        for feature in transformers_dict:
            transformers_dict[feature].fit(X_train[feature].values.reshape(-1, 1))

        return index_features, transformers_dict

    
    index_features, transformers_dict = get_preprocessing_params(preprocessing_data)

    def preprocess_data(data, index_features, column_transformer):
        # Scale the data using the precomputed parameters
        data = data.copy(deep = True)
        features = data.drop([label_key, center_key, patient_key], axis=1)
        features = features.iloc[:, index_features]
        target = data[label_key]

        for feature in column_transformer:
            features[feature] = column_transformer[feature].transform(features[feature].values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = seed, stratify=target)

        return X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = preprocess_data(data, index_features, transformers_dict)

    # print shapes of the data
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    # features = features.iloc[:, index_features]

    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = None, stratify=target)

    # print(features.head())

    print(f'Center ID: {center_id} with {len(data)} samples of which positive samples are {len(data.loc[data[label_key] == 1])})')
    

    return (X_train, y_train), (X_test, y_test)


def load_kaggle_hf(data_path, center_id, config) -> Dataset:
    id = center_id
    seed = config["seed"]
    
    if id == -1:
        id = 'switzerland'
    elif id == 1:
        id = 'hungarian'
    elif id == 2:
        id = 'va'
    elif id == 0:
        id = 'cleveland'
    elif id == None:
        pass
    else:
        raise ValueError(f"Invalid center id: {id}")

    # elif id == 5:
        # id = 'cleveland'

    file_name = os.path.join(data_path, "kaggle_hf.csv")
    data = pd.read_csv(file_name)

    scaling_data = data.loc[(data['data_center'] == 'hungarian')]
    # scaling_data = data

    if id is not None:
        data = data.loc[(data['data_center'] == id)]
    

    # print('Categorical Features :',*categorical_features)
    # print('Numerical Features :',*numerical_features)

    def get_preprocessing_params(data):

        # Get the unique values of the categorical features
        col = list(data.columns)
        categorical_features = []
        numerical_features = []
        for i in col:
            if len(data[i].unique()) > 6:
                numerical_features.append(i)
            else:
                categorical_features.append(i)

        transformers_dict = {}

        categorical_features.pop(categorical_features.index('HeartDisease'))
        if 'RestingBP' in numerical_features:
            numerical_features.pop(numerical_features.index('RestingBP'))
        elif 'RestingBP' in categorical_features:
            categorical_features.pop(categorical_features.index('RestingBP'))
        categorical_features.pop(categorical_features.index('RestingECG'))
        categorical_features.pop(categorical_features.index('data_center'))
        numerical_features.pop(numerical_features.index('Oldpeak'))
        min_max_scaling_features = ['Oldpeak']

        for i in categorical_features:
            transformers_dict[i] = OrdinalEncoder()
        for i in numerical_features:
            transformers_dict[i] = StandardScaler()
        for i in min_max_scaling_features:
            transformers_dict[i] = MinMaxScaler()
        
        df1 = data.copy(deep = True)

        target = df1['HeartDisease']
        X_train, X_test, y_train, y_test = train_test_split(df1, target, test_size = 0.20, random_state = seed)

        for feature in transformers_dict:
            if feature == 'ST_Slope':
                # Change value of last row to 'Down' to avoid error as it is missing in some splits
                X_train.loc[X_train.index[-1], feature] = 'Down'
                transformers_dict[feature].fit(X_train[feature].values.reshape(-1, 1))
            else:
                transformers_dict[feature].fit(X_train[feature].values.reshape(-1, 1))

        return transformers_dict
        
    
    def preprocess_data(data, column_transformer):
        # Scale the data using the precomputed parameters
        df1 = data.copy(deep = True)
        features = df1[df1.columns.drop(['HeartDisease','RestingBP','RestingECG', 'data_center'])]
        target = df1['HeartDisease']

        for feature in column_transformer:
            features.loc[:, feature] = column_transformer[feature].transform(features[feature].values.reshape(-1, 1))

        features = features.infer_objects()

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = seed, stratify=target)

        return (X_train, y_train), (X_test, y_test)
    

    preprocessing_params = get_preprocessing_params(scaling_data)

    (X_train, y_train), (X_test, y_test) = preprocess_data(data, preprocessing_params)

    return (X_train, y_train), (X_test, y_test)


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
    num_centers = config.get("num_clients", 5)
    alpha = config.get("dirichlet_alpha", 1.0)
    reference_method = config.get("reference_center_method", "largest")
    global_preprocessing_params = None
    n_features = config.get("n_features", 20)
    feature_selection_method = config.get("feature_selection_method", "mutual_info")

    # Load the dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
    
    # Get features and target
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets
    
    # convert y to a pandas Series for easier handling
    y = pd.Series(y.values.flatten())

    # Use fraction of data for faster testing (optional)
    fraction = 0.02
    X = X.sample(frac=fraction, random_state=42).reset_index(drop=True)
    y = y.loc[X.index].reset_index(drop=True)
    
    # Set random seed for reproducible partitioning
    np.random.seed(42)
    
    # Convert target to binary classification if needed
    if y.nunique() > 2:
        y_binary = (y > y.median()).astype(int)
    else:
        y_binary = y
    
    # Partition data using Dirichlet distribution
    all_center_indices = partition_data_dirichlet(y_binary.values, num_centers, alpha)

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
            X_reference, y_reference, n_features, feature_selection_method
        )
        print("Calculated new global preprocessing parameters with feature selection")
    
    if center_id:
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
    X_train_processed, feature_names = apply_preprocessing(X_train, global_preprocessing_params)
    X_test_processed, _ = apply_preprocessing(X_test, global_preprocessing_params)
    
    # Convert targets to numpy arrays
    # y_train_processed = y_train.values
    # y_test_processed = y_test.values
    
    # # Print center statistics
    # print(f"Center {center_id}/{num_centers} (alpha={alpha}):")
    # print(f"  Samples: {len(X_center)} (Train: {len(X_train_processed)}, Test: {len(X_test_processed)})")
    # print(f"  Features: {X_train_processed.shape[1]}/{len(X.columns)} selected")
    # print(f"  Data range: [{X_train_processed.min():.3f}, {X_train_processed.max():.3f}]")
    # print(f"  Normalized stats - Mean: {X_train_processed.mean():.4f}, Std: {X_train_processed.std():.4f}")
    
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
        return load_cvd(config["data_path"], id)
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
