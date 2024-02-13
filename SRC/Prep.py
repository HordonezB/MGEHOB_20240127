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

def prep_code():
    """Preprocesa los datos para el modelado, realizando ingeniería de características y selección de variables.
    Inputs:
        Datos RAW parea realizar procesamiento / feature engineering

    Paso 1: Obtener la ruta del archivo CSV
    Paso 2: Cargar los datos desde el archivo CSV
    Paso 3: Realizar ingeniería de características
    Paso 4: Realizar análisis de correlación y selección de variables
    Paso 5: Combinar características seleccionadas y variable objetivo en un solo DataFrame
    Paso 6: Exportar el DataFrame preprocesado a un archivo CSV

    Returns:
        Data procesada y lista para entrenar un modelo 
    """
    # Paso 1: Obtener la ruta del archivo CSV
    current_directory = os.getcwd()
    # Imprimir el directorio de trabajo
    print("Directorio de trabajo actual:", current_directory)

    # Paso 2: Cargar los datos desde el archivo CSV
    file_path = os.path.join(current_directory, "DATA\\train.csv")
    data = pd.read_csv(file_path)

    # Imprimir el resultado
    print(data)

    # Paso 3: Realizar ingeniería de características
    # Seleccionar solo columnas categóricas
    categorical_columns = data.select_dtypes(include='object').columns.tolist()

    # Transformación de variables categóricas
    data_encoded = pd.get_dummies(data[categorical_columns], drop_first=True)

    # Eliminar las columnas categóricas originales
    data_sc = data.drop(categorical_columns, axis=1)

    # Concatenar las columnas codificadas al conjunto de datos
    data_num = pd.concat([data_sc, data_encoded], axis=1)

    # Paso 4: Realizar análisis de correlación y selección de variables
    # Calcular la matriz de correlación para variables numéricas
    correlation_matrix = data_num.corr()

    # Correlación con respecto a la variable objetivo (por ejemplo, 'SalePrice')
    correlation_with_target = correlation_matrix['SalePrice']
    # Imprimir la correlación con la variable objetivo
    # print(correlation_with_target)
    # Imprimir la matriz de correlación
    # print(correlation_matrix)

    # Seleccionar las variables relevantes para el modelo
    # Analizar y seleccionar características según la correlación con la variable objetivo
    selected_features = correlation_matrix['SalePrice'][abs(correlation_matrix['SalePrice']) > 0.5].index
    selected_features = selected_features.drop('SalePrice')  # Excluir 'SalePrice' de las características seleccionadas
    target = 'SalePrice'
    X = data_num[selected_features]
    y = data_num[target]  

    # Paso 5: Combinar características seleccionadas y variable objetivo en un solo DataFrame
    # Suponiendo que X y y son DataFrames de Pandas
    data_combined = pd.concat([X, y], axis=1)

    # Imprimir el DataFrame combinado
    print("Data Input:")
    print(data_combined)

    # Paso 6: Exportar el DataFrame preprocesado a un archivo CSV
    file_pathexp = os.path.join(current_directory, "DATA\Prep.csv")
    data_combined.to_csv(file_pathexp, index=False)

    # Confirmación de exportación
    print("El archivo CSV se ha exportado exitosamente en:", file_pathexp)

# Ejecutar la función si se ejecuta este script directamente
if __name__ == "__main__":
    prep_code()
