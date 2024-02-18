import os
import argparse
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train_code(input_file=None, output_file=None):
    """
    Entrena un modelo de RandomForestRegressor y guarda el modelo entrenado en un archivo.
    Inputs:
        Datos procesados para realizar entrenamiento de modelo

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
    if input_file is None:
        input_file = os.path.join(current_directory, "DATA", "Prep.csv")
    prep = pd.read_csv(input_file)

    # Paso 3: Seleccionar características y variable objetivo
    x_features = prep.iloc[:, :13]
    target_variable = prep.iloc[:, -1]
    print(x_features)
    print(target_variable)

    # Paso 4: División de datos
    # Dividir el conjunto de datos en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x_features, target_variable, test_size=0.2, random_state=42)

    # Paso 5: Entrenamiento del modelo
    # Crear y entrenar el modelo
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(x_train, y_train)

    # Paso 6: Guardar el modelo entrenado en un archivo
    if output_file is None:
        output_file = 'finalized_model.sav'
    with open(output_file, 'wb') as file:
        pickle.dump(model, file)

    # Paso 7: Evaluación del ajuste
    # Evaluar el modelo
    y_pred = model.predict(x_test)
    print(y_pred)
    mse = mean_absolute_error(y_test, y_pred)
    print(f'mean_absolute_error: {mse}')

# Ejecutar la función si se ejecuta este script directamente
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo RandomForestRegressor")
    parser.add_argument("--input_file", type=str, help="Ruta del archivo CSV de entrada")
    parser.add_argument("--output_file", type=str, help="Ruta para guardar el modelo entrenado")
    args = parser.parse_args()

    train_code(args.input_file, args.output_file)
