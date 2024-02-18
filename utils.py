import logging
import pandas as pd

# Obtener el logger configurado en el script principal
logger = logging.getLogger(__name__)

def read_file(file_path):
    """
    Lee un archivo CSV y realiza alguna operación con los datos.
    
    Args:
        file_path (str): Ruta al archivo CSV.
        
    Returns:
        pandas.DataFrame: DataFrame con los datos del archivo CSV.
    """
    try:
        # Intenta leer el archivo CSV
        data = pd.read_csv(file_path)

        # Verifica si el archivo tiene al menos una fila de datos
        if len(data) == 0:
            raise ValueError(f"El archivo '{file_path}' está vacío.")

        # Operaciones adicionales con los datos
        # ...

        return data

    except FileNotFoundError as e:
        # Si el archivo no se encuentra
        logger.error(f"Error: El archivo '{file_path}' no existe.")
        return None

    except pd.errors.EmptyDataError as e:
        # Si el archivo está vacío
        logger.error(f"Error: El archivo '{file_path}' está vacío.")
        return None

    except Exception as e:
        # Para cualquier otra excepción inesperada
        logger.error(f"Error inesperado al leer el archivo '{file_path}': {e}")
        return None



def write_file(data, file_path):
    """
    Escribe un DataFrame de Pandas en un archivo CSV.
    
    Args:
        data (pandas.DataFrame): DataFrame que se va a escribir en el archivo CSV.
        file_path (str): Ruta al archivo CSV donde se escribirán los datos.
        
    Returns:
        bool: True si la escritura fue exitosa, False si ocurrió un error.
    """
    try:
        if data.empty:  # Comprueba si el DataFrame está vacío
            logger.error("Error: El DataFrame proporcionado está vacío.")
            return False
        
        # Intenta escribir el DataFrame en el archivo CSV
        data.to_csv(file_path, index=False)

        logger.info(f"Datos guardados exitosamente en '{file_path}'")
        return True

    except Exception as e:
        # Maneja cualquier excepción inesperada
        logger.error(f"Error al escribir en el archivo '{file_path}': {e}")
        return False
