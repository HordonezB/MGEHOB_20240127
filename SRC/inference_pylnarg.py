import os
import argparse
import pickle
import pandas as pd

def inference_code(input_model=None, output_file=None):
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
    if input_model is None:
        input_model = 'finalized_model.sav'
    with open(input_model, 'rb') as file:
        loaded_model = pickle.load(file)

    # Paso 5: Seleccionar nuevas observaciones
    new_observation = x.iloc[:100].values  # Convertir a un arreglo bidimensional

    # Paso 6: Realizar la predicción
    predicted_value = loaded_model.predict(new_observation)

    # Paso 7: Guardar los resultados en un archivo CSV
    predicted_df = pd.DataFrame(predicted_value, columns=['Predicted_Target'])

    # Ruta completa para guardar el archivo CSV en el directorio actual
    if output_file is None:
        output_file = os.path.join(current_directory, "DATA", "Predictions.csv")

    # Exporta el DataFrame a un archivo CSV en el directorio actual
    predicted_df.to_csv(output_file, index=False)

    # Confirmación de exportación
    print("El archivo CSV se ha exportado exitosamente en:", output_file)

# Ejecutar la función si se ejecuta este script directamente
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realizar inferencia utilizando un modelo entrenado")
    parser.add_argument("--input_model", type=str, help="Ruta del archivo del modelo entrenado")
    parser.add_argument("--output_file", type=str, help="Ruta para guardar los resultados en un archivo CSV")
    args = parser.parse_args()

    inference_code(args.input_model, args.output_file)
