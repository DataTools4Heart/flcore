import os
from typing import Tuple

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
import bz2
import os
import shutil
import urllib.request

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]


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
            if i+1 != center_id:
                continue
            x_train, y_train = X[train_index], y[train_index]
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            print(f"Loaded subset of MNIST with fold {i+1} out of {num_splits}.")
    else:
        x_train, y_train = X[:60000], y[:60000]
        x_test, y_test = X[60000:], y[60000:]

    # y_train = np.array(np.array(y_train, dtype=bool), dtype=float)
    # y_test = np.array(np.array(y_test, dtype=bool), dtype=float)

    return (x_train, y_train), (x_test, y_test)

def load_cvd(data_path, center_id=None) -> Dataset:

    id = center_id
    # match num_center:
    #     case -1:
    #         file_name = data_path+'data_centerAll.csv'
    #     case 1:
    #         file_name = data_path+'data_center1.csv'
    #     case 2:
    #         file_name = data_path+'data_center2.csv'
    #     case _:
    #         file_name = data_path+'data_center3.csv'
    # 
    if id == None:
        id = 'All'

    file_name = os.path.join(data_path,f'data_center{id}.csv')

    code_id = 'f_eid'
    code_outcome = 'Eval'

    
    data = pd.read_csv(file_name)
    X_data = data.drop([code_id, code_outcome], axis=1)
    y_data = data[code_outcome]
    f_eid = data[code_id]

    #Split the data
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
    train_index, test_index = next(sss.split(X_data, y_data))
    X_test = X_data.iloc[test_index, :]
    X_train = X_data.iloc[train_index, :]
    y_test, y_train = y_data.iloc[test_index], y_data.iloc[train_index]
    #We save the names
    f_eid.iloc[test_index]
    f_eid.iloc[train_index]

    
    return (X_train, y_train), (X_test, y_test)

def load_libsvm(center_id=None, task_type="BINARY"):
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
                with bz2.BZ2File(abs_filepath) as fr, open(abs_filepath[:-4], "wb") as fw:
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
                with bz2.BZ2File(abs_filepath) as fr, open(abs_filepath[:-4], "wb") as fw:
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

    return (X_train, y_train), (X_test, y_test)

def load_dataset(config, id=None):
    if config['dataset'] == 'mnist':
        return load_mnist(id, config['num_clients'])
    elif config['dataset'] == 'cvd':
        return load_cvd(config['data_path'], id)
    elif config['dataset'] == 'libsvm':
        return load_libsvm()
    else:
        raise ValueError('Invalid dataset name')
