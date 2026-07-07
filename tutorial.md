1. Node & Environment Variables

| Argument            | Type | Default        | Description                                |
| ------------------- | ---- | -------------- | ------------------------------------------ |
| `--node_name`       | str  | "./"           | Node identifier for certificate management |
| `--local_port`      | int  | 8081           | Local port for client-server communication |
| `--sandbox_path`    | str  | "/sandbox"     | Path to the client sandbox                 |
| `--certs_path`      | str  | "/certs"       | Path to SSL certificates                   |
| `--data_path`       | str  | "/data"        | Path to the dataset                        |
| `--production_mode` | str  | "True"         | Enable production mode (minimal logs)      |
| `--experiment_name` | str  | "experiment_1" | Folder to store experiment outputs         |
2. Dataset Variables

| Argument                 | Type      | Default           | Description                                   |
| ------------------------ | --------- | ----------------- | --------------------------------------------- |
| `--dataset`              | str       | "dt4h_format"     | Dataset loader type                           |
| `--data_id`              | str       | "data_id.parquet" | Dataset filename                              |
| `--normalization_method` | str       | "IQR"             | Normalization method: `IQR`, `MIN_MAX`, `STD` |
| `--train_labels`         | list[str] | None              | List of feature columns for training          |
| `--target_labels`        | list[str] | None              | List of target columns (labels)               |
| `--train_size`           | float     | 0.7               | Fraction for training dataset                 |
| `--validation_size`      | float     | 0.2               | Fraction for validation dataset               |
| `--test_size`            | float     | 0.1               | Fraction for testing dataset                  |

3. Training Variables

| Argument                        | Type  | Default     | Description                                        |
| ------------------------------- | ----- | ----------- | -------------------------------------------------- |
| `--num_rounds`                  | int   | 50          | Number of federated iterations                     |
| `--lr`                          | float | 1e-3        | Learning rate (when applicable)                    |
| `--checkpoint_selection_metric` | str   | "precision" | Metric used to select checkpoint models            |
| `--seed`                        | int   | 42          | Random seed                                        |
| `--num_clients`                 | int   | 1           | Number of clients in federation (informative only) |

4. General Model Variables

| Argument         | Type | Default         | Description                                                           |
| ---------------- | ---- | --------------- | --------------------------------------------------------------------- |
| `--model`        | str  | "random_forest" | Model type (`random_forest`, `xgb`, `nn`, `linear`, `cox`)            |
| `--n_feats`      | int  | 0               | Number of input features (can be inferred from dataset)               |
| `--n_out`        | int  | 0               | Number of output features (depends on target type)                    |
| `--task`         | str  | "None"          | Task type (`classification`, `regression`, `survival`)                |
| `--device`       | str  | "cpu"           | Device for training (`cpu`, `cuda`)                                   |
| `--local_epochs` | int  | 10              | Number of epochs per federated round                                  |
| `--batch_size`   | int  | 8               | Batch size for training                                               |
| `--penalty`      | str  | "none"          | Regularization penalty: `none`, `l1`, `l2`, `elasticnet`, `smooth l1` |
5. Model-Specific Variables
Linear Models / SVR

| Argument     | Type  | Default  | Description                                         |
| ------------ | ----- | -------- | --------------------------------------------------- |
| `--solver`   | str   | "saga"   | Optimization solver                                 |
| `--l1_ratio` | float | 0.5      | L1-L2 ratio for ElasticNet                          |
| `--max_iter` | int   | 100000   | Maximum optimizer iterations                        |
| `--tol`      | float | 0.001    | Tolerance for optimization                          |
| `--kernel`   | str   | "linear" | Kernel for SVR (`linear`, `poly`, `rbf`, `sigmoid`) |
| `--degree`   | int   | 3        | Polynomial degree (for `poly` kernel)               |
| `--gamma`    | str   | "scale"  | Gamma parameter (for `rbf` / `poly` kernels)        |
Random Forest
￼| Argument                 | Type | Default         | Description                    |
| ------------------------ | ---- | --------------- | ------------------------------ |
| `--balanced`             | str  | "True"          | Balanced class weights         |
| `--n_estimators`         | int  | 100             | Number of trees                |
| `--max_depth`            | int  | 2               | Maximum tree depth             |
| `--class_weight`         | str  | "balanced"      | Class weight strategy          |
| `--levelOfDetail`        | str  | "DecisionTree"  | Logging detail level           |
| `--regression_criterion` | str  | "squared_error" | Criterion for regression trees |

XGBoost
| Argument         | Type  | Default   | Description                                 |
| ---------------- | ----- | --------- | ------------------------------------------- |
| `--booster`      | str   | "gbtree"  | Booster type (`gbtree`, `gblinear`, `dart`) |
| `--tree_method`  | str   | "hist"    | Tree construction method (`exact`, `hist`)  |
| `--train_method` | str   | "bagging" | Training method (`bagging`, `cyclic`)       |
| `--eta`          | float | 0.1       | Learning rate                               |

Neural Networks
| Argument      | Type  | Default | Description                     |
| ------------- | ----- | ------- | ------------------------------- |
| `--dropout_p` | float | 0.0     | Monte Carlo dropout probability |
| `--T`         | int   | 20      | Number of MC dropout samples    |

￼
Survival

| Argument                       | Type  | Default | Description                        |
| ------------------------------ | ----- | ------- | ---------------------------------- |
| `--time_col`                   | str   | "time"  | Column containing survival times   |
| `--event_col`                  | str   | "event" | Column indicating event occurrence |
| `--negative_duration_strategy` | str   | "clip"  | Strategy for negative durations    |
| `--l1_penalty`                 | float | 0.0     | L1 regularization penalty          |

1. General Settings

| Argument                  | Type | Default    | Description                                    |
| ------------------------- | ---- | ---------- | ---------------------------------------------- |
| `--model`                 | str  | None       | Model type                                     |
| `--task`                  | str  | None       | Task type                                      |
| `--num_rounds`            | int  | 50         | Number of federated iterations                 |
| `--num_clients`           | int  | 1          | Total number of clients                        |
| `--min_fit_clients`       | int  | 0          | Minimum clients required to perform `fit`      |
| `--min_evaluate_clients`  | int  | 0          | Minimum clients required to perform evaluation |
| `--min_available_clients` | int  | 0          | Minimum clients required to start a round      |
| `--seed`                  | int  | 42         | Random seed                                    |
| `--sandbox_path`          | str  | "/sandbox" | Path to sandbox directory                      |
| `--local_port`            | int  | 8081       | Server listening port                          |
| `--production_mode`       | str  | "True"     | Production mode (minimal logs)                 |

2. Strategy Settings

| Argument                        | Type  | Default            | Description                          |
| ------------------------------- | ----- | ------------------ | ------------------------------------ |
| `--strategy`                    | str   | "FedAvg"           | Federated aggregation strategy       |
| `--smooth_method`               | str   | "EqualVoting"      | Weight smoothing method              |
| `--smoothing_strenght`          | float | 0.5                | Weight smoothing strength            |
| `--dropout_method`              | str   | None               | Dropout strategy for clients         |
| `--dropout_percentage`          | float | 0.0                | Ratio of dropout nodes               |
| `--checkpoint_selection_metric` | str   | "precision"        | Metric used for checkpoint selection |
| `--metrics_aggregation`         | str   | "weighted_average" | Aggregation method for metrics       |
| `--experiment_name`             | str   | "experiment_1"     | Directory for experiment outputs     |
