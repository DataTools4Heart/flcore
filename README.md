# FLCore
Library of Federated Learning models integrated within the DataTools4Heart project.

![CI status](https://gitlab.bsc.es/fl/flcore/badges/main/pipeline.svg)

## Currently implemented models
| Model | Aggregation method | Alias | Link |
|---|---|---|---|
|Logistic regression| FedAvg | `logistic_regression` |[flower.dev/docs/framework/quickstart-scikitlearn.html](https://flower.dev/docs/framework/quickstart-scikitlearn.html)|
|SGD Classifier| FedAvg |`lsvc` | [flower.dev/docs/framework/quickstart-scikitlearn.html](https://flower.dev/docs/framework/quickstart-scikitlearn.html) |
|Elastic Net| FedAvg |`elastic_net` | [flower.dev/docs/framework/quickstart-scikitlearn.html](https://flower.dev/docs/framework/quickstart-scikitlearn.html) |
|Random Forest| Custom |`random_forest` | [Random Forest Based on Federated Learning for Intrusion Detection](https://link.springer.com/chapter/10.1007/978-3-031-08333-4_11) |
|Weighted Random Forest| Custom |`weighted_random_forest` | [Random Forest Based on Federated Learning for Intrusion Detection](https://link.springer.com/chapter/10.1007/978-3-031-08333-4_11) |
|XGBoost| FedXgbNnAvg |`xgb` |[Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates](https://arxiv.org/abs/2304.07537)|
|Deep Learning | FedAvg |`nn` |[https://flower.dev/docs/framework/tutorial-quickstart-pytorch.html](https://flower.dev/docs/framework/tutorial-quickstart-pytorch.html)|

## Quickstart
Install necessary dependencies:
```
pip install -r requirements.txt
```
To start a federated training run:
```
python run.py
```
it will automatically start a server and client processes defined in `config.yaml`

### Step by step
Also, you can do it manually by running:
```
python server.py
```
And then, preferably in a separate shell window for clean output, start clients with their corresponding ids:
```
python client.py 1
```
```
python client.py 2
```

## Configuration file
The federated training parameters are defined in ```config.yaml```
The most important parameters are:
 - `num_clients` (number of clients needed in a federated training)
 - `num_rounds` (number of training rounds)
 - `model` (machine learning model with it's federated implementation)

 ## Data loader
To train on your own dataset add a loading method in the `datasets.py` file and a corresponding entry in the `load_dataset()` method.

#### Loading method
 ```python
 XY = Tuple[np.ndarray, np.ndarray]
 Dataset = Tuple[XY, XY]

 def load_my_dataset(data_path, center_id=None) -> Dataset:
 ```

 #### Note
 It is important to note that each client can only use it's subset of data corresponding to it's institution. When deployed in a real federated setting,
 each client will access the available data through the provided `data_path` in `config.yaml` file. To enable this behaviour in simulated setting,
 a dataset loading method should accept `center_id` argument in order to load only a specific part of a dataset and simulate distributed data scheme.



 ## Contributing
 To add a new model to the framework two methods need to be implemented:
 #### For server side:

 ```python
 def get_server_and_strategy(config, data = None) -> Tuple[Optional[flwr.server.Server], flwr.server.strategy.Strategy]:
 ```
 which returns Flower Server object (optional) and Flower Strategy object.

#### For client side:

 ```python
 def get_client(config, data) -> flwr.client.Client:
 ```
 This method should return the initialized client with data loaded specifically for this data center.

#### Contribution steps
After implementing the necessary methods follow the remaining steps:
1. Create a new branch in `flcore` repository
2. Copy your model package to `flcore/models` directory
3. Add cases for the new model in `server_selector.py` and `client_selector.py` modules in `flcore/` directory
4. Add the model to the available models table in `README.md` file
5. Open a Pull Request and wait for review
