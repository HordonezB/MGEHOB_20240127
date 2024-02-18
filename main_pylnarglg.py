import os
import logging
import traceback
from datetime import datetime
from SRC import prep_pylnarg
from SRC import train_pylnarg
from SRC import inference_pylnarg

# Obtener el timestamp de la ejecución
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

# Crear el directorio "logs" si no existe
logs_directory = os.path.join(os.getcwd(), "logs")
if not os.path.exists(logs_directory):
    os.makedirs(logs_directory)

# Crear el subdirectorio con el timestamp si no existe
logs_timestamp_directory = os.path.join(logs_directory, timestamp)
if not os.path.exists(logs_timestamp_directory):
    os.makedirs(logs_timestamp_directory)

# Configurar el formato de los mensajes de log
log_format = "%(asctime)s - %(levelname)s - %(message)s"

# Configurar el logger para el módulo de preparación
log_prep_file_name = os.path.join(logs_timestamp_directory, "prep.log")
prep_logger = logging.getLogger("prep")
prep_logger.setLevel(logging.DEBUG)
file_handler_prep = logging.FileHandler(log_prep_file_name, mode="w")  # Se especifica modo de escritura (overwrite)
file_handler_prep.setLevel(logging.DEBUG)
file_handler_prep.setFormatter(logging.Formatter(log_format))
prep_logger.addHandler(file_handler_prep)

# Configurar el logger para el módulo de entrenamiento
log_train_file_name = os.path.join(logs_timestamp_directory, "train.log")
train_logger = logging.getLogger("train")
train_logger.setLevel(logging.DEBUG)
file_handler_train = logging.FileHandler(log_train_file_name, mode="w")  # Se especifica modo de escritura (overwrite)
file_handler_train.setLevel(logging.DEBUG)
file_handler_train.setFormatter(logging.Formatter(log_format))
train_logger.addHandler(file_handler_train)

# Configurar el logger para el módulo de inferencia
log_inference_file_name = os.path.join(logs_timestamp_directory, "inference.log")
inference_logger = logging.getLogger("inference")
inference_logger.setLevel(logging.DEBUG)
file_handler_inference = logging.FileHandler(log_inference_file_name, mode="w")  # Se especifica modo de escritura (overwrite)
file_handler_inference.setLevel(logging.DEBUG)
file_handler_inference.setFormatter(logging.Formatter(log_format))
inference_logger.addHandler(file_handler_inference)

# Logs de los pasos de ejecución del script
prep_logger.info("Inicio del script")
prep_logger.info("Paso 1: Cargando datos")
prep_logger.info("Paso 2: Limpiando datos")
prep_logger.info("Paso 3: Creando features")
prep_logger.info("Paso 4: Entrenando modelo")
prep_logger.info("Paso 5: Realizando inferencia")
prep_logger.info("Fin del script")

# Variables de los datasets de entrada y salida
prep_logger.debug(f"Número de filas y columnas en el dataset de entrada: {1460}, {81}")
prep_logger.debug(f"Número de filas y columnas en el dataset de salida: {1460}, {13}")
prep_logger.debug(f"Ruta del archivo de entrada: {logs_timestamp_directory}")
prep_logger.debug(f"Ruta del archivo de salida: {logs_timestamp_directory}")

train_logger.debug(f"Número de filas y columnas en el dataset de entrada para entrenamiento: {1460}, {13}")
train_logger.debug(f"Ruta del archivo de entrada para entrenamiento: {logs_timestamp_directory}")
train_logger.debug(f"Ruta del archivo de salida del modelo entrenado: {logs_timestamp_directory}")

inference_logger.debug(f"Ruta del modelo de inferencia: {logs_timestamp_directory}")
inference_logger.debug(f"Ruta del archivo de salida de las predicciones: {logs_timestamp_directory}")

# Llamar a las funciones de los módulos prep_pylnarg, train_pylnarg e inference_pylnarg
current_directory = os.getcwd()

try:
    # 1) Se ejecuta código de procesamiento/preparación de Datos
    input_file_prep = os.path.join(current_directory, "DATA", "train.csv")
    output_file_prep = os.path.join(current_directory, "DATA", "Prep.csv")
    prep_pylnarg.prep_code(input_file=input_file_prep, output_file=output_file_prep)
except Exception as e:
    prep_logger.error(f"Error en el paso de preparación: {e}")
    prep_logger.error(f"Traceback: {traceback.format_exc()}")

try:
    # 2) Se ejecuta código de entrenamiento/fitting del modelo
    input_file_train = os.path.join(current_directory, "DATA", "Prep.csv")
    output_file_train = os.path.join(current_directory, "finalized_model.sav")
    train_pylnarg.train_code(input_file=input_file_train, output_file=output_file_train)
except Exception as e:
    train_logger.error(f"Error en el paso de entrenamiento: {e}")
    train_logger.error(f"Traceback: {traceback.format_exc()}")

try:
    # 3) Se ejecuta código de scoring/predict del modelo
    input_model_inference = os.path.join(current_directory, "finalized_model.sav")
    output_file_inference = os.path.join(current_directory, "DATA", "Predictions.csv")
    inference_pylnarg.inference_code(input_model=input_model_inference, output_file=output_file_inference)
except Exception as e:
    inference_logger.error(f"Error en el paso de inferencia: {e}")
    inference_logger.error(f"Traceback: {traceback.format_exc()}")
