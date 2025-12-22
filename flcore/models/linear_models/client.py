
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import time
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
import warnings
import flcore.models.linear_models.utils as utils
import flwr as fl
from sklearn.metrics import log_loss
from flcore.performance import measurements_metrics, get_metrics
from flcore.metrics import calculate_metrics
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error




# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, data,config):
        self.config = config
        self.node_name = config["node_name"]
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        # Create train and validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train,
                self.y_train,
                test_size=config["test_size"],
                random_state=config["seed"],
                stratify=self.y_train)

        # #Only use the standardScaler to the continous variables
        # scaled_features_train = StandardScaler().fit_transform(self.X_train.values)
        # scaled_features_train = pd.DataFrame(scaled_features_train, index=self.X_train.index, columns=self.X_train.columns)
        # self.X_train = scaled_features_train

        # #Only use the standardScaler to the continous variables. 
        # scaled_features = StandardScaler().fit_transform(self.X_test.values)
        # scaled_features_df = pd.DataFrame(scaled_features, index=self.X_test.index, columns=self.X_test.columns)
        # self.X_test = scaled_features_df

        self.model = utils.get_model(config)
        self.round_time = 0
        self.first_round = True
        self.personalize = True
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model, config)

    def get_parameters(self, config):  # type: ignore
        #compute the feature selection
        #We perform it from the one called by the server
        #at the begining to start the parameters
        # if(bool(config) == False):
        #         fs = SelectKBest(f_classif, k= self.n_features).fit(self.X_train, self.y_train)
        #         index_features = fs.get_support()
        #         self.model.features = index_features
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore

        utils.set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #To implement the center dropout, we need the execution time
            start_time = time.time()
            self.model.fit(self.X_train, self.y_train)
            # self.model.fit(self.X_train.loc[:, parameters[2].astype(bool)], self.y_train)
            # y_pred = self.model.predict(self.X_test.loc[:, parameters[2].astype(bool)])
            y_pred = self.model.predict(self.X_test)

            metrics = calculate_metrics(self.y_test, y_pred,config)
            print(f"Client {self.node_name} Evaluation just after local training: {metrics['balanced_accuracy']}")
            # Add 'personalized' to the metrics to identify them
            metrics = {f"personalized {key}": metrics[key] for key in metrics}
            self.round_time = (time.time() - start_time)
            metrics["running_time"] = self.round_time

        print(f"Training finished for round {config['server_round']}")

        if self.first_round:
            local_model = utils.get_model(config)
            utils.set_initial_params(self.model, config)
            local_model.fit(self.X_train, self.y_train)
            y_pred = local_model.predict(self.X_test)
            local_metrics = calculate_metrics(self.y_test, y_pred,config)
            #Add 'local' to the metrics to identify them
            local_metrics = {f"local {key}": local_metrics[key] for key in local_metrics}
            metrics.update(local_metrics)
            self.first_round = False

        return utils.get_model_parameters(self.model), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        utils.set_model_params(self.model, parameters)

        # Calculate validation set metrics
        pred = self.model.predict(self.X_val)
        if self.config["model"] == "logistic_regression": # buscar modelos compatibles
            y_pred = pred
        elif self.config["model"] == "linear_regression": # idem
            y_pred = pred[:,0]    
        print("CLIENT::EVALUATE::Y VAL, Y PRED", self.y_val, y_pred)
        metrics = calculate_metrics(self.y_val, y_pred, config)

        """
        if self.config["model"] == "logistic_regression": # buscar modelos compatibles
            loss = log_loss(
                self.y_test,
                self.model.predict_proba(self.X_test),
                labels=[0, 1]) # NECESITARIAMOS AÑADIR LAS DEMAS CLASES
        elif self.config["model"] == "linear_regression": # idem
            loss = mean_squared_error(self.y_test,
                self.model.predict(self.X_test))
        elif self.config["model"] in ["lsvc","svm"]:
            loss = 1.0
        elif config["model"] in ["svm", "svr"]:
            loss = mean_squared_error(self.y_test,
                    self.model.predict(self.X_test))
        else:
            pass
        """
# .............................................................................................
        if config["task"] == "classification":
            if config["n_out"] > 1: # Multivariable
                losses = []

                if hasattr(self.model, "predict_proba"):
                    y_score = self.model.predict_proba(self.X_test)

                    for m in range(self.y_test.shape[1]):
                        losses.append(
                            log_loss(
                                self.y_test[:, m],
                                y_score[:, m]
                            )
                        )
                else:
                    print("PREDICT PROBA NO DISPONIBLE")
                    """
                    for m in range(self.y_test.shape[1]):
                        losses.append(
                            1.0 - accuracy_score(
                                self.y_test[:, m],
                                y_pred[:, m]
                            )
                        )
                    """
            elif config["n_out"] == 1: # Binario
                if hasattr(self.model, "predict_proba"):
                    loss = log_loss(
                        self.y_test,
                        self.model.predict_proba(self.X_test)
                    )
                else:
                    print("PREDICT PROBA NO DISPONIBLE")
                    """
                    loss = 1.0 - accuracy_score(
                        self.y_test,
                        y_test_pred
                    )
                    """

        elif config["task"] == "regression":
            loss = mean_squared_error(self.y_test, y_pred)

        metrics["round_time [s]"] = self.round_time
        metrics["client_id"] = self.node_name

        print(f"Client {self.node_name} Evaluation after aggregated model: {metrics['balanced_accuracy']}")

        # Add validation metrics to the evaluation metrics with a prefix
        val_metrics = {f"val {key}": val_metrics[key] for key in val_metrics}
        metrics.update(val_metrics)


        return loss, len(y_pred),  metrics


def get_client(config,data) -> fl.client.Client:
    return MnistClient(data,config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())

