# Clasificador de Diabetes

Aplicación Python para el taller de clasificación de diabetes con Random Forest.
Incluye carga y exploración del dataset, entrenamiento del modelo, métricas de
evaluación y un dashboard interactivo para predecir nuevos pacientes.

## Requisitos

- Python 3.10 o superior
- Dataset `diabetes.csv` de Kaggle:
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Guarda `diabetes.csv` en la raíz del proyecto y ejecuta:

```bash
streamlit run app.py
```

La app también permite subir el CSV desde la barra lateral. Si no encuentra un
archivo local, intenta cargar un dataset público de respaldo con la misma
estructura.

## Qué contiene el dashboard

- Exploración inicial: vista previa, tipos de datos, valores nulos, resumen
  estadístico y distribución de la variable objetivo.
- Modelo: división entrenamiento/prueba, entrenamiento con Random Forest,
  accuracy, precision, recall, F1-score y matriz de confusión.
- Predicción: formulario para ingresar las características de un paciente y
  obtener una predicción interactiva.
