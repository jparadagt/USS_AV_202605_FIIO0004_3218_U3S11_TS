from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("diabetes.csv")
FALLBACK_DATA_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
)
TARGET_COLUMN = "Outcome"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelResult:
    model: Pipeline
    feature_columns: list[str]
    metrics: dict[str, float]
    confusion: np.ndarray
    test_size: int
    train_size: int
    target_counts: pd.Series


st.set_page_config(
    page_title="USS_AV_202605_FIIO0004_3218_U3S11_TS - Clasificador de Diabetes",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def find_target_column(data: pd.DataFrame) -> str:
    if TARGET_COLUMN in data.columns:
        return TARGET_COLUMN
    if "Diabetes" in data.columns:
        return "Diabetes"
    return data.columns[-1]


def normalize_target(data: pd.DataFrame, target_column: str) -> pd.Series:
    target = data[target_column]
    if pd.api.types.is_numeric_dtype(target):
        return target.astype(int)

    normalized = target.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "si": 1,
        "sí": 1,
        "positivo": 1,
        "negativo": 0,
    }
    return normalized.map(mapping).astype(int)


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame, target_column: str) -> ModelResult:
    clean_data = data.copy()
    clean_data[target_column] = normalize_target(clean_data, target_column)

    X = clean_data.drop(columns=[target_column])
    X = X.select_dtypes(include=[np.number])
    y = clean_data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    max_depth=None,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    }

    return ModelResult(
        model=model,
        feature_columns=X.columns.tolist(),
        metrics=metrics,
        confusion=confusion_matrix(y_test, y_pred, labels=[0, 1]),
        test_size=len(X_test),
        train_size=len(X_train),
        target_counts=y.value_counts().sort_index(),
    )


def get_data() -> tuple[pd.DataFrame | None, str]:
    uploaded_file = st.sidebar.file_uploader("Subir diabetes.csv", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Archivo subido desde la interfaz"

    if DATA_PATH.exists():
        return load_csv_from_path(str(DATA_PATH)), f"Archivo local: {DATA_PATH}"

    try:
        return load_csv_from_url(FALLBACK_DATA_URL), "Dataset público de respaldo"
    except Exception:
        return None, "Sin datos disponibles"


def show_data_overview(data: pd.DataFrame, target_column: str) -> None:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Vista previa")
        st.dataframe(data.head(12), use_container_width=True)

        st.subheader("Tipos de datos y nulos")
        overview = pd.DataFrame(
            {
                "Columna": data.columns,
                "Tipo": [str(dtype) for dtype in data.dtypes],
                "Valores nulos": data.isna().sum().values,
                "% nulos": (data.isna().mean().values * 100).round(2),
            }
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Distribución de la variable objetivo")
        target_counts = data[target_column].value_counts().sort_index()
        target_labels = {
            0: "Sin diabetes",
            1: "Con diabetes",
            "0": "Sin diabetes",
            "1": "Con diabetes",
        }
        target_plot = pd.DataFrame(
            {
                "Diagnóstico": [
                    target_labels.get(value, str(value)) for value in target_counts.index
                ],
                "Cantidad": target_counts.values,
            }
        )
        fig = px.bar(
            target_plot,
            x="Diagnóstico",
            y="Cantidad",
            text="Cantidad",
            color="Diagnóstico",
            color_discrete_sequence=["#2A9D8F", "#E76F51"],
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Resumen estadístico")
        st.dataframe(data.describe().T.round(2), use_container_width=True)


def show_model_performance(result: ModelResult) -> None:
    metric_cols = st.columns(4)
    for column, (label, value) in zip(metric_cols, result.metrics.items()):
        column.metric(label, f"{value:.3f}")

    st.caption(
        f"Entrenamiento: {result.train_size} pacientes | Prueba: {result.test_size} pacientes | "
        f"Algoritmo: Random Forest"
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Matriz de confusión")
        confusion_fig = go.Figure(
            data=go.Heatmap(
                z=result.confusion,
                x=["Predicción: No", "Predicción: Sí"],
                y=["Real: No", "Real: Sí"],
                colorscale=[[0, "#F5F7FA"], [1, "#264653"]],
                text=result.confusion,
                texttemplate="%{text}",
                hovertemplate="%{y}<br>%{x}<br>Pacientes: %{z}<extra></extra>",
            )
        )
        confusion_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(confusion_fig, use_container_width=True)

    with right:
        st.subheader("Métricas del clasificador")
        metric_df = pd.DataFrame(
            {
                "Métrica": list(result.metrics.keys()),
                "Valor": list(result.metrics.values()),
            }
        )
        fig = px.bar(
            metric_df,
            x="Métrica",
            y="Valor",
            text=metric_df["Valor"].map(lambda value: f"{value:.3f}"),
            range_y=[0, 1],
            color="Métrica",
            color_discrete_sequence=["#287271", "#E9C46A", "#F4A261", "#E76F51"],
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


def default_patient_values(data: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    defaults = data[feature_columns].median(numeric_only=True).to_dict()
    fallback_defaults = {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 79,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.47,
        "Age": 33,
    }
    for column, value in fallback_defaults.items():
        defaults.setdefault(column, value)
    return defaults


def show_prediction_form(data: pd.DataFrame, result: ModelResult) -> None:
    st.subheader("Ingresar características del paciente")
    defaults = default_patient_values(data, result.feature_columns)

    with st.form("patient-form"):
        cols = st.columns(2)
        patient_input: dict[str, float] = {}
        for index, feature in enumerate(result.feature_columns):
            feature_data = data[feature].dropna()
            min_value = float(feature_data.min()) if not feature_data.empty else 0.0
            max_value = float(feature_data.max()) if not feature_data.empty else 300.0
            default_value = float(defaults.get(feature, 0.0))
            step = 1.0 if feature != "BMI" and "Function" not in feature else 0.01
            patient_input[feature] = cols[index % 2].number_input(
                feature,
                min_value=0.0 if min_value >= 0 else min_value,
                max_value=max(max_value * 1.5, default_value + step),
                value=default_value,
                step=step,
            )

        submitted = st.form_submit_button("Predecir diagnóstico")

    if submitted:
        patient_df = pd.DataFrame([patient_input], columns=result.feature_columns)
        prediction = int(result.model.predict(patient_df)[0])
        probability = float(result.model.predict_proba(patient_df)[0][1])

        if prediction == 1:
            st.error(f"Predicción: paciente con posible diabetes ({probability:.1%}).")
        else:
            st.success(f"Predicción: paciente sin diabetes probable ({1 - probability:.1%}).")

        st.progress(probability, text=f"Probabilidad estimada de diabetes: {probability:.1%}")

    st.info(
        "Este resultado es demostrativo para el taller y no reemplaza evaluación clínica."
    )


def main() -> None:
    st.title("USS_AV_202605_FIIO0004_3218_U3S11_TS - Clasificador de Diabetes")
    st.write(
        "Asignatura: Introducción a la ciencia de datos"
    )
    st.write(
        "Profesor: Dr. Mauricio Sepúlveda C."
    )
    st.write(
        "Alumnos: Dr. Cristian Jaramillo B. / Jonathan Parada G."
    )
    st.write(
        "Este proyecto utiliza un dataset de pacientes para entrenar un modelo de Random Forest "
        "que predice la presencia de diabetes. Puedes explorar los datos, revisar el rendimiento "
        "del modelo y probar predicciones con nuevos pacientes."
    )

    data, source = get_data()
    st.sidebar.caption(source)

    if data is None:
        st.warning(
            "No se pudo cargar el dataset. Descarga `diabetes.csv` desde Kaggle y súbelo "
            "en la barra lateral, o guárdalo junto a `app.py`."
        )
        return

    target_column = find_target_column(data)
    if target_column not in data.columns:
        st.error("No se encontró una columna objetivo para entrenar el modelo.")
        return

    if data[target_column].nunique(dropna=True) != 2:
        st.error(
            f"La columna objetivo `{target_column}` debe tener dos clases para este clasificador."
        )
        return

    try:
        result = train_model(data, target_column)
    except Exception as exc:
        st.error(f"No fue posible entrenar el modelo: {exc}")
        return

    st.sidebar.metric("Pacientes", len(data))
    st.sidebar.metric("Variables predictoras", len(result.feature_columns))
    st.sidebar.metric("Columna objetivo", target_column)

    tab_explore, tab_model, tab_predict = st.tabs(
        ["Exploración de datos", "Rendimiento del modelo", "Predicción"]
    )

    with tab_explore:
        show_data_overview(data, target_column)

    with tab_model:
        show_model_performance(result)

    with tab_predict:
        show_prediction_form(data, result)


if __name__ == "__main__":
    main()
