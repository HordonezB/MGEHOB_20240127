"""
Este módulo contiene una función para realizar inferencia utilizando un modelo entrenado
y guardar los resultados en un archivo CSV.
"""

import os
import pickle
import pandas as pd

def inference_code():
    """
    Realiza inferencia utilizando un modelo entrenado y guarda los resultados en un archivo CSV.
    Inputs:
        Modelo precargado y Nuevas observaciones para realizar el Predict

    Paso 1: Obtener la ruta del archivo CSV
    Paso 2: Cargar los datos
    Paso 3: Seleccionar características
    Paso 4: Cargar el modelo
    Paso 5: Seleccionar nuevas observaciones
    Paso 6: Realizar la predicción
    Paso 7: Guardar los resultados en un archivo CSV

    Returns:
        CSV with new predictions
    """

    # Paso 1: Obtener la ruta del archivo CSV
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "DATA", "Prep.csv")

    # Paso 2: Cargar los datos
    prep = pd.read_csv(file_path)

    # Paso 3: Seleccionar características
    x = prep.iloc[:, :13]

    # Paso 4: Cargar el modelo
    filename = 'finalized_model.sav'
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)

    # Paso 5: Seleccionar nuevas observaciones
    new_observation = x.iloc[:100].values  # Convertir a un arreglo bidimensional

    # Paso 6: Realizar la predicción
    predicted_value = loaded_model.predict(new_observation)

    # Paso 7: Guardar los resultados en un archivo CSV
    predicted_df = pd.DataFrame(predicted_value, columns=['Predicted_Target'])

    # Ruta completa para guardar el archivo CSV en el directorio actual
    file_pathexp = os.path.join(current_directory, "DATA", "Predictions.csv")

    # Exporta el DataFrame a un archivo CSV en el directorio actual
    predicted_df.to_csv(file_pathexp, index=False)

    # Confirmación de exportación
    print("El archivo CSV se ha exportado exitosamente en:", file_pathexp)

# Ejecutar la función si se ejecuta este script directamente
if __name__ == "__main__":
    inference_code()
