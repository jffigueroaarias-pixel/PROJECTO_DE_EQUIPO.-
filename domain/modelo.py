import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def seleccionar_features(df: pd.DataFrame, tipo_features: str = "ALL", custom_cols: list = None) -> pd.DataFrame:
    """Selecciona las columnas del DataFrame según la configuración de features."""
    # Excluir columnas que no son features o ID/Target
    cols_a_excluir = ["RowNumber", "CustomerId", "Surname", "Exited"]
    df_feat = df.drop(columns=[c for c in cols_a_excluir if c in df.columns], errors="ignore")
    
    if tipo_features == "ALL":
        return df_feat
    elif tipo_features == "NUMERIC":
        return df_feat.select_dtypes(include=[np.number])
    elif tipo_features == "CUSTOM" and custom_cols:
        cols_existentes = [c for c in custom_cols if c in df_feat.columns]
        return df_feat[cols_existentes]
    
    return df_feat


def construir_pipeline(tipo_modelo: str, parametros: dict, X_train: pd.DataFrame) -> Pipeline:
    """Construye un pipeline de Scikit-Learn con preprocesamiento automático y modelo."""
    # Identificar columnas numéricas y categóricas
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Preprocesador numérico
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Preprocesador categórico
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # Seleccionar clasificador
    if tipo_modelo == "RandomForestClassifier":
        clf = RandomForestClassifier(**parametros)
    elif tipo_modelo == "GradientBoostingClassifier":
        clf = GradientBoostingClassifier(**parametros)
    elif tipo_modelo == "LogisticRegression":
        clf = LogisticRegression(**parametros)
    else:
        clf = RandomForestClassifier(random_state=42)
        
    # Unir preprocesamiento y modelo en un Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    return pipeline


def evaluar_modelo(pipeline, X_test, y_test, umbral: float = 0.5) -> dict:
    """Evalúa el pipeline y retorna un diccionario con las métricas principales."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)
    
    metrics = {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred))
    }
    
    return metrics