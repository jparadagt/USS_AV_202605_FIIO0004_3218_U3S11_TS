# Taller Sumativo N°3 - Clasificador de Diabetes

## Asignatura

Introducción a la ciencia de datos

## Profesor: 

Dr. Mauricio Sepúlveda C.

## Alumnos: 
Cristian Jaramillo B. / Elizabeth Tiñini A. /Jonathan Parada G.

## Descripción

Este proyecto desarrolla un dashboard interactivo en Python usando Streamlit para publicar un modelo de clasificación de diabetes.

El modelo utiliza el algoritmo Random Forest para predecir si un paciente presenta o no diabetes a partir de variables médicas.

## Archivos incluidos

- `appdiabetes.py`: código principal del dashboard.
- `requirements.txt`: librerías necesarias para ejecutar el proyecto.
- `defensa_oral.txt`: explicación breve para presentar el trabajo.
- `README.md`: instrucciones del proyecto.

## Requisitos

Tener Python 3 instalado.

Para comprobar la instalación:

### Windows

```bash
py --version
```

### Linux y macOS

```bash
python3 --version
```

## Instalación

Abrir una terminal dentro de la carpeta del proyecto y ejecutar el comando correspondiente al sistema operativo.

### Windows

```bash
py -m pip install -r requirements.txt
```

### Linux y macOS

```bash
python3 -m pip install -r requirements.txt
```

## Ejecución

Para abrir el dashboard ejecutar el comando correspondiente.

### Windows

```bash
py -m streamlit run appdiabetes.py
```

### Linux y macOS

```bash
python3 -m streamlit run appdiabetes.py
```

Luego se abrirá el navegador con la aplicación. Si no se abre automáticamente, copiar la URL que muestra la terminal, normalmente `http://localhost:8501`.

## Qué hace el dashboard

1. Carga el dataset de diabetes.
2. Muestra una exploración inicial de los datos.
3. Realiza limpieza de datos, tratando valores inválidos como ceros en variables médicas.
4. Completa valores faltantes usando la mediana de cada columna.
5. Divide los datos en entrenamiento y prueba.
6. Entrena un modelo Random Forest.
7. Evalúa el modelo con accuracy, recall, F1-score, reporte de clasificación y matriz de confusión.
8. Muestra la importancia de las variables utilizadas por el modelo.
9. Permite ingresar datos de un nuevo paciente.
10. Entrega una predicción de diabetes o no diabetes.

## Nota importante

Este proyecto es solo académico. La predicción del modelo no corresponde a un diagnóstico médico real.
