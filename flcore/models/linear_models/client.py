
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import time
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import flcore.models.linear_models.utils as utils
import flwr as fl
from sklearn.metrics import log_loss
from flcore.performance import measurements_metrics
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

from flcore.smpc_module import SMPClient


# Define Flower client
# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, data, client_id, config):
        self.client_id = client_id
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        # Only use the standardScaler to the continuous variables
        scaled_features_train = StandardScaler().fit_transform(self.X_train.values)
        scaled_features_train = pd.DataFrame(scaled_features_train, index=self.X_train.index, columns=self.X_train.columns)
        self.X_train = scaled_features_train

        # Only use the standardScaler to the continuous variables.
        scaled_features = StandardScaler().fit_transform(self.X_test.values)
        scaled_features_df = pd.DataFrame(scaled_features, index=self.X_test.index, columns=self.X_test.columns)
        self.X_test = scaled_features_df

        self.model_name = config['linear_models']['model_type']
        self.n_features = config['linear_models']['n_features']
        self.model = utils.get_model(self.model_name)
        self.model2 = utils.get_model(self.model_name)

        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model, self.n_features)
        utils.set_initial_params(self.model2, self.n_features)

        # Check if SMPC should be used
        self.use_smpc = config.get("smpc", {}).get("use_smpc", False)
        self.smpc_url = config.get("smpc", {}).get("smpc_url")
        print("use_smpc:", self.use_smpc)
        if self.use_smpc:
            # Initialize SMPClient only if use_smpc is True
            self.smpc_client = SMPClient(self.model, self.smpc_url, self.client_id)

        else:
            self.smpc_client = None

    def get_parameters(self, config):  # type: ignore
        # compute the feature selection
        # We perform it from the one called by the server
        # at the beginning to start the parameters

        print("calling get_parameters", self.use_smpc, self.smpc_client)
        if not bool(config):
            fs = SelectKBest(f_classif, k=self.n_features).fit(self.X_train, self.y_train)
            index_features = fs.get_support()
            self.model.features = index_features
            self.model2.features = index_features

        # Use SMPClient to share weights with the external SMPC server if enabled
        if self.use_smpc and self.smpc_client:
            if isinstance(self.model, SGDClassifier):
                weights = [self.model.coef_.ravel(), self.model.intercept_]
            else:
                # Handle other scikit-learn models or update accordingly
                weights = [self.model.coef_.ravel(), self.model.intercept_]

            weights = weights if weights else []
            self.smpc_client.share_weights(weights, config)
            return utils.get_model_parameters(self.model2)
        else:
            return utils.get_model_parameters(self.model)


    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # To implement the center dropout, we need the execution time
            start_time = time.time()
            self.model.fit(self.X_train.loc[:, parameters[2].astype(bool)], self.y_train)
            accuracy, specificity, sensitivity, balanced_accuracy, precision, F1_score = \
                measurements_metrics(self.model, self.X_test.loc[:, parameters[2].astype(bool)], self.y_test)
            print(f"Accuracy client in fit:  {accuracy}")
            print(f"Sensitivity client in fit:  {sensitivity}")
            print(f"Specificity client in fit:  {specificity}")
            print(f"Balanced_accuracy in fit:  {balanced_accuracy}")
            print(f"precision in fit:  {precision}")
            print(f"F1_score in fit:  {F1_score}")
            elapsed_time = (time.time() - start_time)

        # Use SMPClient to share weights with the external SMPC server after training if enabled
        if self.use_smpc and self.smpc_client:
            # Get coefficients and intercept from LogisticRegression model
            weights = [self.model.coef_.ravel(), self.model.intercept_]

            # Share weights using SMPClient
            self.smpc_client.share_weights(weights, config)

        # print(f"Training finished for round {config['server_round']}")
        return utils.get_model_parameters(self.model), len(self.X_train), {"running_time": elapsed_time}

    def evaluate(self, parameters, config):  # type: ignore
        weights = []

        utils.set_model_params(self.model, parameters)
        if isinstance(self.model, SGDClassifier):
            loss = 1.0
        else:
            loss = log_loss(
                self.y_test,
                self.model.predict_proba(self.X_test.loc[:, parameters[2].astype(bool)]),
            )
        accuracy = self.model.score(
            self.X_test.loc[:, parameters[2].astype(bool)], self.y_test
        )
        accuracy, specificity, sensitivity, balanced_accuracy, precision, F1_score = measurements_metrics(
            self.model,
            self.X_test.loc[:, parameters[2].astype(bool)],
            self.y_test,
        )

        # Use SMPClient to share weights with the external SMPC server during evaluation if enabled
        if self.use_smpc and self.smpc_client:
            if isinstance(self.model, SGDClassifier):
                weights = [self.model.coef_.ravel(), self.model.intercept_]

            self.smpc_client.share_weights(weights, config)

        print(f"Accuracy client in evaluate:  {accuracy}")
        print(f"Sensitivity client in evaluate:  {sensitivity}")
        print(f"Specificity client in evaluate:  {specificity}")
        print(f"Balanced_accuracy in evaluate:  {balanced_accuracy}")
        print(f"precision in evaluate:  {precision}")
        print(f"F1_score in evaluate:  {F1_score}")
        return loss, len(
            self.X_test.loc[:, parameters[2].astype(bool)]
        ), {"accuracy": float(accuracy), "sensitivity": float(sensitivity),
            "specificity": float(specificity)}


def get_client(config, data, client_id) -> fl.client.Client:
    return MnistClient(data, client_id, config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())

