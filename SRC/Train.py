import tkinter as tk
from tkinter import ttk
import pandas as pd  # Para manipulación de datos
from sklearn.model_selection import train_test_split  # Para dividir datos en conjunto de entrenamiento y prueba
from sklearn.ensemble import RandomForestRegressor  # Para usar un modelo de bosque aleatorio
from sklearn.metrics import mean_squared_error  # Para evaluar el rendimiento del modelo
from sklearn.preprocessing import StandardScaler  # Para estandarizar características
from sklearn.pipeline import make_pipeline  # Para construir un pipeline de transformación
from sklearn.metrics import mean_absolute_error
import os  # Para trabajar con funciones relacionadas con el sistema operativo
import pickle

def train_code():
    """Entrena un modelo de RandomForestRegressor y guarda el modelo entrenado en un archivo.
    Inputs:
        Datos procesados parea realizar entrenamiento de modelo

    Paso 1: Obtener la ruta del archivo CSV
    Paso 2: Cargar los datos
    Paso 3: Seleccionar características y variable objetivo
    Paso 4: Dividir los datos en conjuntos de entrenamiento y prueba
    Paso 5: Entrenar el modelo RandomForestRegressor
    Paso 6: Guardar el modelo entrenado en un archivo
    Paso 7: Evaluar el rendimiento del modelo

    Returns:
        Modelo en un objeto pickle ya entrenado y listo para realizar predicts
    """
    # Paso 1: Obtener la ruta del archivo CSV
    current_directory = os.getcwd()
    # Imprimir el directorio de trabajo
    print("Directorio de trabajo actual:", current_directory)

    # Paso 2: Cargar los datos
    file_path = os.path.join(current_directory, "DATA\\Prep.csv")
    prep = pd.read_csv(file_path)
    
    # Paso 3: Seleccionar características y variable objetivo
    X = prep.iloc[:, :13]
    Y = prep.iloc[:, -1]
    print(X)
    print(Y)
    
    # Paso 4: División de datos
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Paso 5: Entrenamiento del modelo
    # Crear y entrenar el modelo
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    # Paso 6: Guardar el modelo entrenado en un archivo
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # Paso 7: Evaluación del ajuste
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print(y_pred)
    mse = mean_absolute_error(y_test, y_pred)
    print(f'mean_absolute_error: {mse}')

# Ejecutar la función si se ejecuta este script directamente
if __name__ == "__main__":
    train_code()
