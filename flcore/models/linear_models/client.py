
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import time
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import flcore.models.linear_models.utils as utils
import flwr as fl
from sklearn.metrics import log_loss
from flcore.performance import measurements_metrics, get_metrics
from flcore.metrics import calculate_metrics
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler




# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, data,client_id,config):
        self.client_id = client_id
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        
        # #Only use the standardScaler to the continous variables
        # scaled_features_train = StandardScaler().fit_transform(self.X_train.values)
        # scaled_features_train = pd.DataFrame(scaled_features_train, index=self.X_train.index, columns=self.X_train.columns)
        # self.X_train = scaled_features_train

        # #Only use the standardScaler to the continous variables. 
        # scaled_features = StandardScaler().fit_transform(self.X_test.values)
        # scaled_features_df = pd.DataFrame(scaled_features, index=self.X_test.index, columns=self.X_test.columns)
        # self.X_test = scaled_features_df

        self.model_name = config['model']
        self.n_features = config['linear_models']['n_features']
        self.model = utils.get_model(self.model_name)
        self.round_time = 0
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model,self.n_features)
    
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

            metrics = calculate_metrics(self.y_test, y_pred)

            self.round_time = (time.time() - start_time)

            metrics["running_time"] = self.round_time

        print(f"Training finished for round {config['server_round']}")
        return utils.get_model_parameters(self.model), len(self.X_train), metrics

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)
        y_pred = self.model.predict(self.X_test)
        # y_pred = self.model.predict(self.X_test.loc[:, parameters[2].astype(bool)])

        if(isinstance(self.model, SGDClassifier)):
            loss = 1.0
        else:
            loss = log_loss(self.y_test, self.model.predict_proba(self.X_test), labels=[0, 1])
       
        metrics = calculate_metrics(self.y_test, y_pred)
        metrics["round_time [s]"] = self.round_time
        metrics["client_id"] = self.client_id

        return loss, len(y_pred),  metrics


def get_client(config,data,client_id) -> fl.client.Client:
    return MnistClient(data,client_id,config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
