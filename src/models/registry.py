"""Registro de modelos — adicione qualquer classificador compatível com sklearn aqui.

Cada entrada mapeia uma chave de modelo para um dicionário com:
  - "model"  : estimador instanciado e não treinado
  - "params" : hiperparâmetros a registrar no MLflow
  - "mlflow" : configuração de registro no Model Registry (opcional)
               Chaves: model_description, version_description, version_tags, version_alias
"""

import logging

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict = {
    "dummy": {
        "model": DummyClassifier(strategy="most_frequent"),
        "params": {"strategy": "most_frequent"},
        "mlflow": {
            "model_description": "DummyClassifier — piso mínimo que prediz sempre a classe majoritária.",
            "version_description": "DummyClassifier treinado na Etapa 1 — baseline ingênuo de referência.",
            "version_tags": {"stage": "etapa1", "model_family": "baseline", "framework": "sklearn"},
            "version_alias": "baseline",
        },
    },
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "mlflow": {
            "model_description": "Regressão Logística — baseline linear para o dataset Telco Customer Churn.",
            "version_description": "LogisticRegression com C=1.0 treinada na Etapa 1.",
            "version_tags": {"stage": "etapa1", "model_family": "baseline", "framework": "sklearn"},
            "version_alias": "baseline",
        },
    },
    "ridge": {
        "model": RidgeClassifier(alpha=1.0),
        "params": {"alpha": 1.0},
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(max_depth=6, random_state=42),
        "params": {"max_depth": 6},
    },
    "random_forest": {
        "model": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 200, "max_depth": 8},
    },
    "mlp_sklearn": {
        "model": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            learning_rate_init=1e-3,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "params": {"hidden_layers": "128-64-32", "activation": "relu", "max_iter": 200},
    },
}

try:
    from catboost import CatBoostClassifier  # type: ignore[import]

    MODEL_REGISTRY["catboost"] = {
        "model": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
        ),
        "params": {"iterations": 300, "depth": 6, "learning_rate": 0.05},
    }
except ImportError:
    logger.warning("CatBoost não instalado — ignorando do registro")

try:
    from xgboost import XGBClassifier  # type: ignore[import]

    MODEL_REGISTRY["xgboost"] = {
        "model": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    }
except ImportError:
    logger.warning("XGBoost não instalado — ignorando do registro")


PYTORCH_REGISTRY: dict = {
    "MLP_PyTorch": {
        "mlflow": {
            "model_description": "MLP PyTorch com early stopping treinado na Etapa 2 para o dataset Telco Customer Churn.",
            "version_description": "MLP PyTorch — Etapa 2. Arquitetura [128, 64, 32] com dropout e early stopping.",
            "version_tags": {"stage": "etapa2", "model_family": "pytorch", "framework": "pytorch"},
            "version_alias": "challenger",
        },
    },
}
