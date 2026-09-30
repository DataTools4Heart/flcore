"""
Test end-to-end de load_model + InferenceEngine.

No depende de tener modelos reales entrenados en disco: para cada formato
(.pkl sklearn, .npz numpy MLP, .json xgboost, .pt torch MLP) entrena/crea
un modelo sintético mínimo, lo guarda con la MISMA estructura de bundle que
usan los save_model reales (model+metadata juntos), y luego verifica que
load_model + InferenceEngine lo cargan y predicen correctamente.

xgboost y torch son opcionales: si no están instalados, esos tests se
saltan (no fallan) y se reporta al final.

Uso:
    python test_inference.py

Ajusta el import de abajo al path real de tu módulo de inferencia.
"""

import json
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --- AJUSTA ESTE IMPORT al módulo real donde viven load_model/InferenceEngine ---
sys.path.insert(0, str(Path(__file__).parent))
from inference import load_model, InferenceEngine  # noqa: E402


# --------------------------------------------------------------------------- #
# Metadata sintética común: 2 features numéricas, 1 nominal, 1 booleana;
# target nominal binario. Cubre las tres ramas de preprocess()/predict().
# --------------------------------------------------------------------------- #

def make_metadata(model_type, n_feats=4, n_out=1, task="classification", extra=None):
    features_meta = {
        "age": {"dataType": "NUMERIC", "stats": {"q1": 30.0, "q2": 45.0, "q3": 60.0,
                                                  "min": 18.0, "max": 90.0}},
        "score": {"dataType": "NUMERIC", "stats": {"q1": 0.2, "q2": 0.5, "q3": 0.8,
                                                    "min": 0.0, "max": 1.0}},
        "category": {"dataType": "NOMINAL", "stats": {"valueSet": ["A", "B", "C"]}},
        "flag": {"dataType": "BOOLEAN", "stats": {}},
    }
    outcomes_meta = {
        "outcome": {"dataType": "NOMINAL", "stats": {"valueSet": ["neg", "pos"]}}
    }
    metadata = {
        "node_name": "test_node",
        "task": task,
        "n_out": n_out,
        "n_feats": n_feats,
        "model_type": model_type,
        "feature_names": ["age", "score", "category", "flag"],
        "target_names": ["outcome"],
        "metrics": None,
        "features_meta": features_meta,
        "outcomes_meta": outcomes_meta,
        "is_final": False,
        "round": 0,
        "model_name": f"{model_type}_{task}_round_0",
    }
    if extra:
        metadata.update(extra)
    return metadata


def make_raw_dataframe(n=6, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "age": rng.uniform(20, 80, n),
        "score": rng.uniform(0, 1, n),
        "category": rng.choice(["A", "B", "C"], n),
        "flag": rng.choice([True, False], n),
    })


# --------------------------------------------------------------------------- #
# Runner minimalista (sin dependencia de pytest)
# --------------------------------------------------------------------------- #

class TestRunner:
    def __init__(self):
        self.results = []

    def run(self, name, fn):
        try:
            fn()
            self.results.append((name, "PASS", None))
            print(f"[PASS] {name}")
        except unittest_skip as e:  # noqa: F821 (definido más abajo)
            self.results.append((name, "SKIP", str(e)))
            print(f"[SKIP] {name} — {e}")
        except AssertionError as e:
            self.results.append((name, "FAIL", str(e)))
            print(f"[FAIL] {name} — {e}")
        except Exception as e:
            self.results.append((name, "ERROR", f"{type(e).__name__}: {e}"))
            print(f"[ERROR] {name} — {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)

    def summary(self):
        n_pass = sum(1 for _, s, _ in self.results if s == "PASS")
        n_fail = sum(1 for _, s, _ in self.results if s in ("FAIL", "ERROR"))
        n_skip = sum(1 for _, s, _ in self.results if s == "SKIP")
        print("\n" + "=" * 60)
        print(f"Resultado: {n_pass} OK, {n_fail} fallos, {n_skip} saltados "
              f"(de {len(self.results)} tests)")
        print("=" * 60)
        return n_fail == 0


class SkipTest(Exception):
    pass


unittest_skip = SkipTest


# --------------------------------------------------------------------------- #
# Test 1: .pkl — RandomForestClassifier (cubre también linear_models, xgb-sklearn,
#         model_wrapper genérico: todos usan el mismo bundle pickle)
# --------------------------------------------------------------------------- #

def test_pkl_random_forest(tmp_dir):
    from sklearn.ensemble import RandomForestClassifier
    import sklearn
    import pickle

    rng = np.random.RandomState(0)
    X_train = rng.uniform(0, 1, (50, 4))
    y_train = rng.randint(0, 2, 50)

    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X_train, y_train)

    # Simula lo que hace save_model: fija los atributos y embebe versión.
    model.n_outputs_ = 1
    model.n_features_in_ = 4

    metadata = make_metadata("random_forest", extra={"sklearn_version": sklearn.__version__})

    model_dir = tmp_dir / "rf"
    model_dir.mkdir()
    model_path = model_dir / "random_forest_classification_round_0_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata}, f)

    loaded_model, loaded_meta = load_model(model_dir, "random_forest", "classification")
    assert loaded_meta["model_type"] == "random_forest"

    engine = InferenceEngine(loaded_model, loaded_meta)
    df = make_raw_dataframe()
    preds = engine.predict(df)

    assert len(preds) == len(df), f"esperaba {len(df)} predicciones, obtuve {len(preds)}"
    assert all(p in ("neg", "pos") for p in preds), f"valores fuera del valueSet: {preds}"


# --------------------------------------------------------------------------- #
# Test 3: .pt — MLP torch real (state_dict / MC-Dropout)
# --------------------------------------------------------------------------- #

def test_pt_torch_mlp_binary(tmp_dir):
    """BasicNN real, caso binario (n_out=1, BCEWithLogitsLoss/sigmoid)."""
    try:
        import torch
    except ImportError:
        raise SkipTest("PyTorch no instalado en este entorno")
    from flcore.models.nn.basic_nn import BasicNN
    #from basic_nn import BasicNN  # clase real, tal cual la pasó el usuario

    torch.manual_seed(0)
    model = BasicNN(n_feats=4, n_out=1, p=0.3)

    metadata = make_metadata(
        "nn_torch", n_out=1, task="classification",
        extra={"torch_version": torch.__version__, "mc_samples": 5, "dropout_p": 0.3},
    )

    model_dir = tmp_dir / "nn_torch_binary"
    model_dir.mkdir()
    model_path = model_dir / "nn_torch_classification_round_0_model.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, model_path)

    loaded_model, loaded_meta = load_model(
        model_dir, "nn_torch", "classification", mlp_cls=BasicNN
    )
    engine = InferenceEngine(loaded_model, loaded_meta)
    df = make_raw_dataframe()
    preds = engine.predict(df)

    assert len(preds) == len(df)
    assert all(p in ("neg", "pos") for p in preds), f"valores inesperados: {preds}"

    # predict_uncertainty debe devolver una desviación >= 0 por muestra MC.
    X = engine.preprocess(df)
    unc = loaded_model.predict_uncertainty(X)
    assert (unc >= 0).all(), "la incertidumbre estimada no puede ser negativa"


def test_pt_torch_mlp_missing_dropout_p_raises(tmp_dir):
    """Si falta dropout_p en metadata, debe fallar explícito, no con un
    AttributeError/TypeError confuso al reconstruir BasicNN."""
    try:
        import torch
    except ImportError:
        raise SkipTest("PyTorch no instalado en este entorno")

    from flcore.models.nn.basic_nn import BasicNN

    model = BasicNN(n_feats=4, n_out=1, p=0.3)
    metadata = make_metadata("nn_torch", n_out=1, task="classification")
    metadata.pop("dropout_p", None)  # asegurarnos de que falta

    model_dir = tmp_dir / "nn_torch_missing_p"
    model_dir.mkdir()
    model_path = model_dir / "nn_torch_classification_round_0_model.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, model_path)

    try:
        load_model(model_dir, "nn_torch", "classification", mlp_cls=BasicNN)
        raise AssertionError("se esperaba ValueError por falta de dropout_p en metadata")
    except ValueError as e:
        assert "dropout_p" in str(e)


# --------------------------------------------------------------------------- #
# Test 4: .json — XGBoost Booster
# --------------------------------------------------------------------------- #

def test_json_xgboost(tmp_dir):
    try:
        import xgboost
    except ImportError:
        raise SkipTest("xgboost no instalado en este entorno")

    rng = np.random.RandomState(0)
    X_train = rng.uniform(0, 1, (50, 4))
    y_train = rng.randint(0, 2, 50)

    dtrain = xgboost.DMatrix(X_train, label=y_train)
    bst = xgboost.train({"objective": "binary:logistic", "max_depth": 2}, dtrain, num_boost_round=5)

    metadata = make_metadata("xgb", extra={"xgboost_version": xgboost.__version__})

    with tempfile.TemporaryDirectory() as raw_tmp:
        raw_path = Path(raw_tmp) / "bst.json"
        bst.save_model(str(raw_path))
        with open(raw_path, "r") as f:
            bst_json = json.load(f)

    model_dir = tmp_dir / "xgb"
    model_dir.mkdir()
    model_path = model_dir / "xgb_classification_round_0_model.json"
    with open(model_path, "w") as f:
        json.dump({"model": bst_json, "metadata": metadata}, f)

    loaded_model, loaded_meta = load_model(model_dir, "xgb", "classification")
    engine = InferenceEngine(loaded_model, loaded_meta)
    df = make_raw_dataframe()

    # XGBoost Booster.predict devuelve probabilidades, no clases -- para este
    # test basta con comprobar que corre y da el shape correcto; el mapeo a
    # clase/valueSet asume predict() ya discretizado, así que aquí solo
    # validamos preprocess + ejecución del modelo, no el mapeo NOMINAL final.
    X = engine.preprocess(df)
    raw_preds = loaded_model.predict(X)
    assert len(raw_preds) == len(df)
    assert ((raw_preds >= 0) & (raw_preds <= 1)).all(), "probabilidades fuera de [0,1]"


# --------------------------------------------------------------------------- #
# Test 5: selección de archivo — prefer_final y orden por ronda
# --------------------------------------------------------------------------- #

def test_prefer_final_selection(tmp_dir):
    import pickle
    from sklearn.linear_model import LogisticRegression

    model_dir = tmp_dir / "prefer_final"
    model_dir.mkdir()

    rng = np.random.RandomState(0)
    X_train = rng.uniform(0, 1, (30, 4))
    y_train = rng.randint(0, 2, 30)
    base_model = LogisticRegression().fit(X_train, y_train)

    def dump(round_num, is_final, suffix):
        meta = make_metadata("linear_models", extra={"round": round_num, "is_final": is_final})
        path = model_dir / f"linear_models_classification_round_{round_num}_model{suffix}.pkl" \
            if suffix else model_dir / f"linear_models_classification_round_{round_num}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": base_model, "metadata": meta}, f)
        return path

    dump(0, False, "")
    dump(1, False, "")
    dump(2, True, "")
    # Copia explícita como "_final" (tal como hace shutil.copyfile en save_model)
    final_path = model_dir / "linear_models_classification_model_final.pkl"
    shutil.copyfile(model_dir / "linear_models_classification_round_2_model.pkl", final_path)

    # Con prefer_final=True debe coger el _final explícito
    _, meta_final = load_model(model_dir, "linear_models", "classification", prefer_final=True)
    assert meta_final["round"] == 2

    # Con prefer_final=False debe coger la ronda numérica más alta (2), no la 0/1
    _, meta_latest = load_model(model_dir, "linear_models", "classification", prefer_final=False)
    assert meta_latest["round"] == 2


# --------------------------------------------------------------------------- #
# Test 6: aviso de versión desajustada (sklearn_version distinto al instalado)
# --------------------------------------------------------------------------- #

def test_version_mismatch_warns(tmp_dir):
    import pickle
    from sklearn.linear_model import LogisticRegression

    model_dir = tmp_dir / "version_mismatch"
    model_dir.mkdir()

    rng = np.random.RandomState(0)
    X_train = rng.uniform(0, 1, (30, 4))
    y_train = rng.randint(0, 2, 30)
    model = LogisticRegression().fit(X_train, y_train)

    metadata = make_metadata("linear_models", extra={"sklearn_version": "0.0.0-fake"})

    model_path = model_dir / "linear_models_classification_round_0_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata}, f)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_model(model_dir, "linear_models", "classification")
        version_warnings = [w for w in caught if "scikit-learn" in str(w.message)]

    assert len(version_warnings) == 1, (
        f"esperaba exactamente 1 warning de versión sklearn, hubo {len(version_warnings)}"
    )


# --------------------------------------------------------------------------- #
# Test 7: preprocess — normalización IQR/MIN_MAX y mapeos NOMINAL/BOOLEAN
# --------------------------------------------------------------------------- #

def test_preprocess_iqr_normalization():
    metadata = make_metadata("random_forest")
    engine = InferenceEngine(model=None, metadata=metadata, normalization_method="IQR")

    df = pd.DataFrame({
        "age": [45.0], "score": [0.5], "category": ["B"], "flag": [True],
    })
    out = engine.preprocess(df)

    # age: (45-45)/(60-30) = 0 ; score: (0.5-0.5)/(0.8-0.2) = 0
    assert abs(out["age"].iloc[0] - 0.0) < 1e-9
    assert abs(out["score"].iloc[0] - 0.0) < 1e-9
    assert out["category"].iloc[0] == 1  # "B" es índice 1 en ["A","B","C"]
    assert out["flag"].iloc[0] == 1      # True -> 1


def test_preprocess_min_max_normalization():
    metadata = make_metadata("random_forest")
    engine = InferenceEngine(model=None, metadata=metadata, normalization_method="MIN_MAX")

    df = pd.DataFrame({
        "age": [90.0], "score": [0.0], "category": ["A"], "flag": [False],
    })
    out = engine.preprocess(df)

    assert abs(out["age"].iloc[0] - 1.0) < 1e-9   # (90-18)/(90-18) = 1
    assert abs(out["score"].iloc[0] - 0.0) < 1e-9  # (0-0)/(1-0) = 0
    assert out["category"].iloc[0] == 0
    assert out["flag"].iloc[0] == 0


# --------------------------------------------------------------------------- #
# Test 8: predict — mapeo inverso de salida (NOMINAL / BOOLEAN / NUMERIC)
# --------------------------------------------------------------------------- #

class FakeModel:
    """Modelo dummy: predict() devuelve índices fijos para probar el mapeo."""

    def __init__(self, fixed_preds):
        self.fixed_preds = np.array(fixed_preds)

    def predict(self, X):
        return self.fixed_preds[: len(X)]


def test_predict_nominal_mapping():
    metadata = make_metadata("random_forest")
    model = FakeModel([0, 1, 0])
    engine = InferenceEngine(model, metadata)
    preds = engine.predict(make_raw_dataframe(n=3))
    assert preds == ["neg", "pos", "neg"]


def test_predict_boolean_mapping():
    metadata = make_metadata("random_forest")
    metadata["outcomes_meta"]["outcome"] = {"dataType": "BOOLEAN", "stats": {}}
    model = FakeModel([1, 0, 1])
    engine = InferenceEngine(model, metadata)
    preds = engine.predict(make_raw_dataframe(n=3))
    assert preds == [True, False, True]


def test_predict_numeric_denormalization():
    metadata = make_metadata("random_forest")
    metadata["outcomes_meta"]["outcome"] = {
        "dataType": "NUMERIC",
        "stats": {"q1": 10.0, "q2": 20.0, "q3": 30.0},
    }
    model = FakeModel([0.0, 1.0])  # en espacio normalizado
    engine = InferenceEngine(model, metadata, normalization_method="IQR")
    preds = engine.predict(make_raw_dataframe(n=2))
    # p * (q3-q1) + q2 : 0*20+20=20 ; 1*20+20=40
    assert preds == [20.0, 40.0]


# --------------------------------------------------------------------------- #

def main():
    runner = TestRunner()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        runner.run("pkl: RandomForestClassifier", lambda: test_pkl_random_forest(tmp_dir))
        runner.run("pt: BasicNN torch real, binario", lambda: test_pt_torch_mlp_binary(tmp_dir))
        runner.run("pt: falla claro si falta dropout_p", lambda: test_pt_torch_mlp_missing_dropout_p_raises(tmp_dir))
        runner.run("json: XGBoost Booster", lambda: test_json_xgboost(tmp_dir))
        runner.run("selección prefer_final / round", lambda: test_prefer_final_selection(tmp_dir))
        runner.run("aviso de versión sklearn desajustada", lambda: test_version_mismatch_warns(tmp_dir))
        runner.run("preprocess: normalización IQR", test_preprocess_iqr_normalization)
        runner.run("preprocess: normalización MIN_MAX", test_preprocess_min_max_normalization)
        runner.run("predict: mapeo NOMINAL inverso", test_predict_nominal_mapping)
        runner.run("predict: mapeo BOOLEAN inverso", test_predict_boolean_mapping)
        runner.run("predict: desnormalización NUMERIC", test_predict_numeric_denormalization)

    ok = runner.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()