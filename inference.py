from pathlib import Path
import json
import joblib


def load_model(model_dir):

    model_dir = Path(model_dir)

    model_file = list(model_dir.glob("*_model.joblib"))[0]
    metadata_file = list(model_dir.glob("*_model_metadata.json"))[0]

    model = joblib.load(model_file)

    with open(metadata_file) as f:
        metadata = json.load(f)

    return model, metadata

import pandas as pd
import numpy as np


class InferenceEngine:

    def __init__(self, model, metadata, normalization_method="IQR"):

        self.model = model
        self.metadata = metadata
        self.normalization_method = normalization_method

        self.features_meta = metadata["features_meta"]
        self.outcomes_meta = metadata["outcomes_meta"]

        self.feature_names = metadata["feature_names"]
        self.target_names = metadata["target_names"]

        self.boolean_map = {False: 0, True: 1, "False": 0, "True": 1}

    def preprocess(self, df):

        dat = df.copy()

        # -------------------
        # FEATURES
        # -------------------

        for name, feat in self.features_meta.items():

            dtype = feat["dataType"]
            stats = feat.get("stats", {})

            if name not in dat.columns:
                continue

            # NUMERIC
            if dtype == "NUMERIC":

                if self.normalization_method == "IQR":

                    q1 = stats.get("q1")
                    q2 = stats.get("q2")
                    q3 = stats.get("q3")

                    if None not in (q1, q2, q3):
                        dat[name] = (dat[name] - q2) / (q3 - q1)

                elif self.normalization_method == "MIN_MAX":

                    mini = stats.get("min")
                    maxi = stats.get("max")

                    if None not in (mini, maxi):
                        dat[name] = (dat[name] - mini) / (maxi - mini)

            # NOMINAL
            elif dtype == "NOMINAL":

                value_set = stats.get("valueSet", [])

                if len(value_set) > 0:
                    cat_map = {cat: i for i, cat in enumerate(value_set)}
                    dat[name] = dat[name].map(cat_map)

            # BOOLEAN
            elif dtype == "BOOLEAN":

                dat[name] = dat[name].map(self.boolean_map)

        return dat[self.feature_names]

    def predict(self, df):

        X = self.preprocess(df)

        preds = self.model.predict(X)

        # -------------------
        # Interpret output
        # -------------------

        target = self.target_names[0]
        target_meta = self.outcomes_meta[target]

        dtype = target_meta["dataType"]

        # NOMINAL → map back category
        if dtype == "NOMINAL":

            value_set = target_meta["stats"].get("valueSet", [])

            if len(value_set) > 0:
                inv_map = {i: cat for i, cat in enumerate(value_set)}
                preds = [inv_map.get(p, p) for p in preds]

        # BOOLEAN
        elif dtype == "BOOLEAN":

            preds = [bool(p) for p in preds]

        return preds

#**** * * * * * *  *  *   *   *     *  *  * * * * *******  INPUT
model_path = "sandbox/model"
model, metadata = load_model(model_path)
new_data_path = "/home/yuca/DT4H/completo/flcore-main/dataset/bucarest_sintetico/synthetic_dt4h_dataset.csv"
#**** * * * * * *  *  *   *   *     *  *  * * * * *******  INPUT
engine = InferenceEngine(model, metadata)

df_new = pd.read_csv(new_data_path)

predictions = engine.predict(df_new)
print(predictions)