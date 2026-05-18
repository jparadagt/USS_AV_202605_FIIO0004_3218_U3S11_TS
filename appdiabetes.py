# ================================================================
# Taller Sumativo N°3 - Clasificador de Diabetes con Streamlit
# Asignatura: Introducción a la Ciencia de Datos
# Modelo: Random Forest
# Dashboard: Streamlit + Plotly
# ================================================================

# ----------------------------
# 1. Importación de librerías
# ----------------------------

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------
# 2. Configuración general de la aplicación
# ------------------------------------------------

st.set_page_config(
    page_title="Clasificador de Diabetes",
    page_icon="🩺",
    layout="wide"
)


# ------------------------------------------------
# 3. Carga de datos
# ------------------------------------------------

@st.cache_data
def cargar_datos():
    """
    Esta función carga el dataset de diabetes.

    Primero intenta leer un archivo local llamado diabetes.csv.
    Si no lo encuentra, carga una versión disponible desde internet.

    El dataset corresponde al caso Pima Indians Diabetes Database,
    usado comúnmente para tareas de clasificación.
    """

    try:
        df = pd.read_csv("diabetes.csv")
        fuente = "Archivo local diabetes.csv"
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df = pd.read_csv(url)
        fuente = "Dataset cargado desde internet"

    return df, fuente


df, fuente_datos = cargar_datos()


# ------------------------------------------------
# 4. Normalización de nombres de columnas
# ------------------------------------------------

# En algunos archivos el nombre de la variable objetivo puede venir como
# "Outcome", "Diabetes" o similar. Esta parte ayuda a detectar la columna.
posibles_objetivos = ["Outcome", "outcome", "Diabetes", "diabetes", "Class", "class"]

columna_objetivo = None

for columna in posibles_objetivos:
    if columna in df.columns:
        columna_objetivo = columna
        break

if columna_objetivo is None:
    st.error(
        "No se encontró la columna objetivo del dataset. "
        "El archivo debe contener una columna llamada Outcome o diabetes."
    )
    st.stop()


# Si la variable objetivo no se llama Outcome, la renombramos para trabajar ordenado.
if columna_objetivo != "Outcome":
    df = df.rename(columns={columna_objetivo: "Outcome"})


# ------------------------------------------------
# 5. Título e introducción del dashboard
# ------------------------------------------------

st.title("USS_AV_202605_FIIO0004_3218_U3S11_TS - Clasificador de Diabetes")

st.write(
    "Asignatura: Introducción a la ciencia de datos"
)
st.write(
    "Profesor: Dr. Mauricio Sepúlveda C."
)
st.write(
    "Alumnos: Cristian Jaramillo B. / Elizabeth Tiñini A. /Jonathan Parada G."
)

st.markdown("""
Este dashboard permite construir, evaluar y publicar un modelo de clasificación para predecir si un
paciente presenta o no diabetes, utilizando medidas médicas del dataset **Pima Indians Diabetes Database**.

El modelo utilizado es **Random Forest**, un algoritmo de Machine Learning basado en múltiples árboles
de decisión. Además, el dashboard permite ingresar datos de un nuevo paciente para generar una predicción.
""")

st.info(f"Fuente de datos utilizada: {fuente_datos}")


# ------------------------------------------------
# 6. Exploración inicial de datos
# ------------------------------------------------

st.header("1. Carga y exploración inicial de datos")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Cantidad de registros", df.shape[0])

with col2:
    st.metric("Cantidad de columnas", df.shape[1])

with col3:
    st.metric("Variable objetivo", "Outcome")

st.subheader("Vista previa del dataset")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Tipos de datos")
tipos_datos = pd.DataFrame({
    "Columna": df.dtypes.index,
    "Tipo de dato": df.dtypes.astype(str).values
})
st.dataframe(tipos_datos, use_container_width=True)

st.subheader("Valores nulos por columna")
valores_nulos = df.isnull().sum().reset_index()
valores_nulos.columns = ["Columna", "Valores nulos"]
st.dataframe(valores_nulos, use_container_width=True)


# ------------------------------------------------
# 7. Distribución de la variable objetivo
# ------------------------------------------------

st.subheader("Distribución de la variable objetivo")

conteo_objetivo = df["Outcome"].value_counts().reset_index()
conteo_objetivo.columns = ["Resultado", "Cantidad"]

conteo_objetivo["Resultado"] = conteo_objetivo["Resultado"].map({
    0: "No presenta diabetes",
    1: "Presenta diabetes"
})

fig_objetivo = px.bar(
    conteo_objetivo,
    x="Resultado",
    y="Cantidad",
    text="Cantidad",
    title="Distribución de pacientes con y sin diabetes"
)

st.plotly_chart(fig_objetivo, use_container_width=True)

st.markdown("""
**Interpretación:**  
La variable objetivo permite identificar si el paciente presenta diabetes o no.
Esta distribución es importante porque ayuda a observar si el dataset está equilibrado
o si existe una mayor cantidad de casos de una clase sobre otra.
""")


# ------------------------------------------------
# 8. Preprocesamiento de datos
# ------------------------------------------------

st.header("2. Preprocesamiento de datos")

st.markdown("""
Antes de entrenar el modelo se corrigen algunos valores inválidos del dataset.
En variables médicas como **Glucose**, **BloodPressure**, **SkinThickness**, **Insulin** y **BMI**,
un valor 0 no representa una medición realista, por lo que se considera como dato faltante.

Estos ceros se reemplazan por valores nulos y luego se completan usando la **mediana** de cada columna,
manteniendo la estructura del dataset sin eliminar registros.
""")

# Separar variables predictoras y variable objetivo.
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Mantener solo columnas numéricas para evitar problemas con el modelo.
X = X.select_dtypes(include=np.number)

# ------------------------------------------------
# Tratamiento de valores inválidos
# ------------------------------------------------
# En este dataset algunas columnas médicas no deberían
# contener valor 0. Por ejemplo:
# - Glucose
# - BMI
# - BloodPressure
#
# Un valor 0 en estos casos suele representar un dato faltante.
# Por ello:
# 1. Reemplazamos 0 por NaN
# 2. Imputamos usando la mediana

columnas_con_ceros_invalidos = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columnas_con_ceros_invalidos:

    # Reemplazar ceros por NaN
    X[col] = X[col].replace(0, np.nan)

    # Completar valores faltantes con la mediana
    X[col] = X[col].fillna(X[col].median())

# ------------------------------------------------
# Verificación posterior al preprocesamiento
# ------------------------------------------------

st.subheader("Valores nulos después del preprocesamiento")

valores_nulos_post = X.isnull().sum().reset_index()
valores_nulos_post.columns = ["Columna", "Valores nulos"]

st.dataframe(valores_nulos_post, use_container_width=True)

# División del dataset en entrenamiento y prueba.
# 80% para entrenar el modelo y 20% para evaluarlo.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

col4, col5 = st.columns(2)

with col4:
    st.metric("Registros de entrenamiento", X_train.shape[0])

with col5:
    st.metric("Registros de prueba", X_test.shape[0])

st.markdown("""
**Interpretación:**  
El conjunto de datos fue dividido en entrenamiento y prueba.  
El entrenamiento permite que el modelo aprenda patrones desde los datos históricos,
mientras que el conjunto de prueba permite evaluar su rendimiento con datos que no fueron usados
durante el aprendizaje.
""")


# ------------------------------------------------
# 9. Construcción del modelo Random Forest
# ------------------------------------------------

st.header("3. Construcción y evaluación del clasificador")

# Sidebar para controlar algunos parámetros simples del modelo.
st.sidebar.header("Configuración del modelo")

n_arboles = st.sidebar.slider(
    "Cantidad de árboles del Random Forest",
    min_value=50,
    max_value=300,
    value=100,
    step=50
)

profundidad_maxima = st.sidebar.slider(
    "Profundidad máxima de los árboles",
    min_value=2,
    max_value=20,
    value=6,
    step=1
)

# Crear el modelo Random Forest.
modelo = RandomForestClassifier(
    n_estimators=n_arboles,
    max_depth=profundidad_maxima,
    random_state=42
)

# Entrenar el modelo con los datos de entrenamiento.
modelo.fit(X_train, y_train)

# Realizar predicciones sobre los datos de prueba.
y_pred = modelo.predict(X_test)


# ------------------------------------------------
# 10. Métricas de evaluación del modelo
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

met1, met2, met3 = st.columns(3)

with met1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with met2:
    st.metric("Recall", f"{recall:.2%}")

with met3:
    st.metric("F1-score", f"{f1:.2%}")

st.markdown("""
**Interpretación de métricas:**  

- **Accuracy:** porcentaje total de predicciones correctas realizadas por el modelo.
- **Recall:** capacidad del modelo para detectar correctamente los casos positivos de diabetes.
- **F1-score:** equilibrio entre precisión y recall, útil cuando se desea evaluar el rendimiento general del clasificador.

En un problema médico, el **recall** es especialmente relevante, ya que interesa reducir la cantidad de
pacientes con diabetes que podrían ser clasificados incorrectamente como sanos.
""")


# ------------------------------------------------
# 11. Matriz de confusión
# ------------------------------------------------

st.subheader("Matriz de confusión")

matriz = confusion_matrix(y_test, y_pred)

fig_matriz = ff.create_annotated_heatmap(
    z=matriz,
    x=["Predicción: No diabetes", "Predicción: Diabetes"],
    y=["Real: No diabetes", "Real: Diabetes"],
    colorscale="Blues",
    showscale=True
)

fig_matriz.update_layout(
    title="Matriz de confusión del modelo",
    xaxis_title="Predicción del modelo",
    yaxis_title="Valor real"
)

st.plotly_chart(fig_matriz, use_container_width=True)

st.markdown("""
**Interpretación:**  
La matriz de confusión permite comparar los resultados reales con las predicciones del modelo.
Con esto se pueden observar los aciertos y errores del clasificador, diferenciando entre pacientes
clasificados correctamente y pacientes clasificados incorrectamente.
""")


# ------------------------------------------------
# 12. Reporte de clasificación
# ------------------------------------------------

st.subheader("Reporte de clasificación")

reporte = classification_report(
    y_test,
    y_pred,
    target_names=["No diabetes", "Diabetes"],
    output_dict=True
)

df_reporte = pd.DataFrame(reporte).transpose()
st.dataframe(df_reporte, use_container_width=True)


# ------------------------------------------------
# 13. Importancia de variables
# ------------------------------------------------

st.subheader("Importancia de variables")

importancias = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": modelo.feature_importances_
}).sort_values(by="Importancia", ascending=False)

fig_importancia = px.bar(
    importancias,
    x="Importancia",
    y="Variable",
    orientation="h",
    title="Importancia de las variables en el modelo Random Forest"
)

fig_importancia.update_layout(yaxis={"categoryorder": "total ascending"})

st.plotly_chart(fig_importancia, use_container_width=True)

st.markdown("""
**Interpretación:**  
Este gráfico muestra qué variables fueron más relevantes para el modelo al momento de realizar
la clasificación. Una mayor importancia indica que esa característica aportó más información
para distinguir entre pacientes con y sin diabetes.
""")


# ------------------------------------------------
# 14. Predicción de un nuevo paciente
# ------------------------------------------------

st.header("4. Predicción interactiva para un nuevo paciente")

st.markdown("""
En esta sección se pueden ingresar las características médicas de un nuevo paciente.
Luego, el modelo entrenado entrega una predicción indicando si el paciente podría presentar
o no diabetes.
""")

st.sidebar.header("Datos del nuevo paciente")

valores_paciente = {}

# Para cada columna predictora se crea una entrada numérica en el sidebar.
for columna in X.columns:
    valor_min = float(df[columna].min())
    valor_max = float(df[columna].max())
    valor_promedio = float(df[columna].mean())

    valores_paciente[columna] = st.sidebar.number_input(
        label=columna,
        min_value=valor_min,
        max_value=valor_max,
        value=valor_promedio
    )

# Convertir los datos ingresados por el usuario en un DataFrame.
nuevo_paciente = pd.DataFrame([valores_paciente])

st.subheader("Datos ingresados para el nuevo paciente")
st.dataframe(nuevo_paciente, use_container_width=True)

if st.button("Predecir resultado del paciente"):

    prediccion = modelo.predict(nuevo_paciente)[0]
    probabilidad = modelo.predict_proba(nuevo_paciente)[0]

    prob_no_diabetes = probabilidad[0]
    prob_diabetes = probabilidad[1]

    if prediccion == 1:
        st.error(
            f"Resultado del modelo: El paciente presenta riesgo de diabetes. "
            f"Probabilidad estimada: {prob_diabetes:.2%}"
        )
    else:
        st.success(
            f"Resultado del modelo: El paciente no presenta indicios de diabetes según el modelo. "
            f"Probabilidad estimada de no diabetes: {prob_no_diabetes:.2%}"
        )

    fig_prob = px.bar(
        x=["No diabetes", "Diabetes"],
        y=[prob_no_diabetes, prob_diabetes],
        labels={"x": "Clase", "y": "Probabilidad"},
        title="Probabilidad estimada por el modelo"
    )

    st.plotly_chart(fig_prob, use_container_width=True)

    st.warning(
        "Importante: esta predicción corresponde a un ejercicio académico de ciencia de datos. "
        "No debe ser usada como diagnóstico médico real."
    )


# ------------------------------------------------
# 15. Conclusión del dashboard
# ------------------------------------------------

st.header("5. Conclusión")

st.markdown("""
El modelo Random Forest permitió construir un clasificador capaz de predecir la presencia o ausencia
de diabetes a partir de medidas médicas. 

A través del dashboard se publican los resultados del modelo, sus métricas de evaluación, la matriz de confusión 
y una herramienta interactiva para predecir el caso
de un nuevo paciente.

Este tipo de solución demuestra cómo la ciencia de datos puede apoyar la toma de decisiones mediante
modelos predictivos, siempre considerando que en contextos médicos los resultados deben ser interpretados
con precaución y complementados por profesionales de la salud.

En este dashboard se agregó la posibilidad de modificar "Cantidad de árboles del RandomForest" y "Profundidad 
máxima de los árboles" para poder ajustar el comportamiento del modelo de forma interactiva. 
La cant. de árboles influye en la estabilidad del clasificador y la profundidad máxima de árboles controla la complejidad 
de cada árbol evitando que pierda la capacidad de generalización.
""")
