# FLCore
Library of Federated Learning models integrated within the DataTools4Heart project.

## Currently implemented models
| Model | Aggregation method | Link |
|---|---|---|
|Logistic regression| FedAvg |[flower.dev/docs/framework/quickstart-scikitlearn.html](https://flower.dev/docs/framework/quickstart-scikitlearn.html)|
|XGBoost| FedXgbNnAvg |[Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates](https://arxiv.org/abs/2304.07537)|
|BNN| --- |[Bayesian Neural Networks with Pyro](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html)|

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
4. Add the model to the avaialable models table in `README.md` file
5. Open a Pull Request and wait for review
