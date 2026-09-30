import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np

try:
    import sklearn
except Exception:
    sklearn = None

try:
    import imblearn
except Exception:
    imblearn = None

try:
    import torch
except Exception:
    # Captura también fallos de import distintos de ImportError (p. ej. un
    # OSError si la instalación de torch quedó corrupta/incompleta), para que
    # una instalación rota de una dependencia opcional no tumbe todo el módulo.
    torch = None

try:
    import xgboost
except Exception:
    xgboost = None


def _check_version(metadata, key, current_version, label):
    saved_version = metadata.get(key)
    if saved_version and current_version and saved_version != current_version:
        warnings.warn(
            f"Modelo guardado con {label} {saved_version}, entorno actual tiene "
            f"{current_version}. Las predicciones podrían no ser fiables.",
            RuntimeWarning,
        )


def _find_model_file(model_dir, model, task, prefer_final=True):
    model_dir = Path(model_dir)
    name_prefix = f"{model}_{task}" if model and task else ""
    extensions = [".pkl", ".pt", ".json", ".npz"]

    def get_round(filepath):
        match = re.search(r'_round_(\d+)', filepath.name)
        return int(match.group(1)) if match else -1

    if prefer_final:
        final_candidates = []
        for ext in extensions:
            final_candidates.extend(model_dir.glob(f"{name_prefix}*_model_final{ext}"))
        if final_candidates:
            return final_candidates[0]

    candidates = []
    for ext in extensions:
        candidates.extend(model_dir.glob(f"{name_prefix}*_model{ext}"))

    if not candidates:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    candidates.sort(key=get_round, reverse=True)
    return candidates[0]


def load_model(model_dir, model=None, task=None, prefer_final=True, mlp_cls=None):
    """
    mlp_cls: clase MCDropoutMLP a usar para reconstruir modelos .pt / .npz.
    Se pasa explícitamente en vez de importarla fija dentro de esta función,
    porque hay (al menos) dos implementaciones distintas en el proyecto
    (numpy puro vs torch.nn.Module real) con nombres que pueden colisionar.
    """
    model_dir = Path(model_dir)
    model_file = _find_model_file(model_dir, model, task, prefer_final=prefer_final)
    ext = model_file.suffix.lower()

    if ext == ".pkl":
        with open(model_file, "rb") as f:
            bundle = pickle.load(f)
        model_obj = bundle["model"]
        metadata = bundle["metadata"]

        if sklearn is not None:
            _check_version(metadata, "sklearn_version", sklearn.__version__, "scikit-learn")
        _check_version(
            metadata, "imblearn_version",
            getattr(imblearn, "__version__", None) if imblearn else None,
            "imbalanced-learn",
        )

    elif ext == ".pt":
        if torch is None:
            raise RuntimeError("Se necesita PyTorch instalado para cargar este modelo (.pt)")
        if mlp_cls is None:
            raise ValueError("Hay que pasar mlp_cls para reconstruir un modelo .pt")

        # weights_only=False explícito: desde PyTorch 2.6 el default cambió a
        # True, que solo permite un allowlist de tipos "seguros" y rechaza
        # cosas como metadata anidada arbitraria o torch.__version__ (que es
        # TorchVersion, no un str plano). El bundle es generado por nuestro
        # propio save_model, no viene de una fuente externa no confiable, así
        # que desactivarlo aquí es razonable en vez de allowlistear a mano.
        bundle = torch.load(model_file, map_location="cpu", weights_only=False)
        metadata = bundle["metadata"]

        n_feats = metadata.get("n_feats")
        n_out = metadata.get("n_out")
        task_meta = metadata.get("task", "classification")
        dropout_p = metadata.get("dropout_p")
        if dropout_p is None:
            raise ValueError(
                "metadata no trae 'dropout_p'; BasicNN no se puede reconstruir sin él. "
                "Añade \"dropout_p\": self.config[\"dropout_p\"] al diccionario metadata en save_model."
            )
        n_samples = metadata.get("mc_samples", 20)

        # BasicNN(n_feats, n_out, p=...) — posicionales + p, sin 'task' (no existe
        # como parámetro del constructor ni como atributo del modelo).
        base_model = mlp_cls(n_feats, n_out, p=dropout_p)
        base_model.load_state_dict(bundle["state_dict"])
        base_model.to("cpu")

        class NnWrapperTorch:
            """
            task/n_out se guardan aquí, no en el modelo: BasicNN no los expone
            como atributos. La activación de salida se decide igual que en
            FlowerClient.evaluate() (sigmoid para binario, softmax para
            multiclase, nada para regresión) — no se reutiliza
            BasicNN.predict_proba_mc porque ese método aplica softmax
            incondicionalmente, lo cual es incorrecto para el caso binario
            (entrenado con BCEWithLogitsLoss/sigmoid, no con softmax).
            """

            def __init__(self, m, task, n_out, n_samples=20):
                self.m = m
                self.task = task
                self.n_out = n_out
                self.n_samples = n_samples

            def _forward_samples(self, x_tensor):
                self.m.train()  # MC-Dropout: dropout activo en inferencia
                with torch.no_grad():
                    samples = [self.m(x_tensor).numpy() for _ in range(self.n_samples)]
                return np.stack(samples, axis=0)  # (T, B, n_out), logits crudos

            def predict(self, X):
                x = X.values if hasattr(X, "values") else X
                x_tensor = torch.as_tensor(x, dtype=torch.float32)
                logits_mean = self._forward_samples(x_tensor).mean(axis=0)

                if self.task == "regression":
                    return logits_mean

                if self.n_out == 1:
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_mean, -500, 500)))
                    return (probs[:, 0] > 0.5).astype(int)

                exp = np.exp(logits_mean - logits_mean.max(axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)
                return probs.argmax(axis=1)

            def predict_uncertainty(self, X):
                x = X.values if hasattr(X, "values") else X
                x_tensor = torch.as_tensor(x, dtype=torch.float32)
                return self._forward_samples(x_tensor).std(axis=0)

        model_obj = NnWrapperTorch(base_model, task_meta, n_out, n_samples=n_samples)
        _check_version(metadata, "torch_version", torch.__version__, "torch")

    elif ext == ".json":
        if xgboost is None:
            raise RuntimeError("Se necesita xgboost instalado para cargar este modelo (.json)")

        with open(model_file, "r") as f:
            bundle = json.load(f)
        metadata = bundle["metadata"]

        bst = xgboost.Booster()
        bst.load_model(bytearray(json.dumps(bundle["model"]), "utf-8"))

        class XgbWrapper:
            def __init__(self, b):
                self.b = b

            def predict(self, X):
                dmat = xgboost.DMatrix(X)
                return self.b.predict(dmat)

        model_obj = XgbWrapper(bst)
        _check_version(metadata, "xgboost_version", xgboost.__version__, "xgboost")

    elif ext == ".npz":
        if mlp_cls is None:
            raise ValueError("Hay que pasar mlp_cls para reconstruir un modelo .npz")

        data = np.load(model_file, allow_pickle=False)
        metadata = json.loads(str(data["__metadata__"]))

        def _weight_sort_key(k):
            match = re.search(r'(\d+)$', k)
            return int(match.group(1)) if match else -1

        weight_keys = sorted((k for k in data.files if k != "__metadata__"), key=_weight_sort_key)
        weights = [data[k] for k in weight_keys]

        n_feats = metadata.get("n_feats")
        n_out = metadata.get("n_out")
        task_meta = metadata.get("task", "classification")

        base_model = mlp_cls(n_feats=n_feats, n_out=n_out, task=task_meta)
        base_model.set_weights(weights)

        class NnWrapperNumpy:
            def __init__(self, m):
                self.m = m

            def predict(self, X):
                x = X.values if hasattr(X, "values") else X
                logits = self.m(x)
                if self.m.task == "regression":
                    return logits
                if self.m.n_out == 1:
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
                    return (probs[:, 0] > 0.5).astype(int)
                return logits.argmax(axis=1)

        model_obj = NnWrapperNumpy(base_model)
        _check_version(metadata, "numpy_version", np.__version__, "numpy")

    else:
        raise ValueError(f"Extensión de modelo no soportada: {ext}")

    return model_obj, metadata


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

        for name, feat in self.features_meta.items():
            dtype = feat["dataType"]
            stats = feat.get("stats", {})

            if name not in dat.columns:
                continue

            if dtype == "NUMERIC":
                if self.normalization_method == "IQR":
                    q1, q2, q3 = stats.get("q1"), stats.get("q2"), stats.get("q3")
                    if None not in (q1, q2, q3):
                        dat[name] = (dat[name] - q2) / (q3 - q1)
                elif self.normalization_method == "MIN_MAX":
                    mini, maxi = stats.get("min"), stats.get("max")
                    if None not in (mini, maxi):
                        dat[name] = (dat[name] - mini) / (maxi - mini)

            elif dtype == "NOMINAL":
                value_set = stats.get("valueSet", [])
                if len(value_set) > 0:
                    cat_map = {cat: i for i, cat in enumerate(value_set)}
                    dat[name] = dat[name].map(cat_map)

            elif dtype == "BOOLEAN":
                dat[name] = dat[name].map(self.boolean_map)

        return dat[self.feature_names]

    def predict(self, df):
        X = self.preprocess(df)

        if hasattr(self.model, "predict_risk"):
            preds = self.model.predict_risk(X)
        else:
            preds = self.model.predict(X)

        target = self.target_names[0] if len(self.target_names) > 0 else None
        target_meta = self.outcomes_meta.get(target, {}) if target else {}
        dtype = target_meta.get("dataType", None)

        if dtype == "NOMINAL":
            value_set = target_meta.get("stats", {}).get("valueSet", [])
            if len(value_set) > 0:
                inv_map = {i: cat for i, cat in enumerate(value_set)}
                preds = [inv_map.get(p, p) for p in preds]

        elif dtype == "BOOLEAN":
            preds = [bool(p) for p in preds]

        elif dtype == "NUMERIC":
            stats = target_meta.get("stats", {})
            if self.normalization_method == "IQR":
                q1, q2, q3 = stats.get("q1"), stats.get("q2"), stats.get("q3")
                if None not in (q1, q2, q3):
                    preds = [p * (q3 - q1) + q2 for p in preds]
            elif self.normalization_method == "MIN_MAX":
                mini, maxi = stats.get("min"), stats.get("max")
                if None not in (mini, maxi):
                    preds = [p * (maxi - mini) + mini for p in preds]

        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        elif not isinstance(preds, list):
            preds = list(preds)

        return preds

    def explain(self, df):
        if hasattr(self.model, "explain"):
            X = self.preprocess(df)
            return self.model.explain(X)
        return None