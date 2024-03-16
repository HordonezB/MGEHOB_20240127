# Proyecto de Preprocesamiento y Modelado de Datos

## Descripción

Este proyecto se centra en el preprocesamiento de datos y la creación de modelos predictivos utilizando el lenguaje de programación Python y diversas bibliotecas de análisis de datos y machine learning.

El proyecto consta de tres partes principales:


## Arquitectura (Diadrama del Producto de Datos)
![image](https://github.com/HordonezB/MGEHOB_20240127/assets/141704495/9bedc494-09a6-4897-b23e-ea3937f82ee6)


- **Preprocesamiento de Datos:** Se encarga de realizar la limpieza, transformación e ingeniería de características en los conjuntos de datos de entrada.

- **Entrenamiento del Modelo:** Utiliza algoritmos de aprendizaje automático para entrenar modelos predictivos utilizando los datos preprocesados.

- **Inferencia:** Implementa un sistema para realizar predicciones utilizando el modelo entrenado sobre nuevos datos de entrada.

## Dependencias

El proyecto requiere las siguientes bibliotecas de Python:

- tkinter
- pandas
- scikit-learn
- numpy

## Inputs/Outputs

### Inputs

Archivos CSV que contienen los datos de entrada para el preprocesamiento y el entrenamiento del modelo.

### Outputs

- Archivos CSV que contienen los datos preprocesados.
- Archivos de modelo guardados en disco.
- Predicciones generadas por el modelo.

## Estructura del Repositorio

SRC/
prep.py: # Código para el preprocesamiento de datos.
train.py: # Código para el entrenamiento del modelo.
inference.py: # Código para la inferencia/predicción.
DATA/


## Cómo ejecutar el Proyecto

- **Preprocesamiento de Datos:** Ejecuta `prep.py` desde la línea de comandos o un entorno de desarrollo.
- **Entrenamiento del Modelo:** Ejecuta `train.py` desde la línea de comandos o un entorno de desarrollo.
- **Inferencia/Predicción:** Ejecuta `inference.py` desde la línea de comandos o un entorno de desarrollo.

## Resultado de Pruebas (pytest)

Los resultados de las pruebas unitarias realizadas con pytest se muestran a continuación:
========================================= test session starts =========================================
platform win32 -- Python 3.11.8, pytest-8.0.1, pluggy-1.4.0
rootdir: D:\Horus\Escritorio\Horus Respaldo\Documents\Profesional\MsterDataScienceITAM_20230410\S22024\S2_MetodosaGranEscala
plugins: anyio-3.7.1, dash-2.13.0
collected 5 items

test_utils.py ..... [100%]

========================================== 5 passed in 0.72s ==========================================
