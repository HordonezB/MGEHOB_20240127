from SRC import prep_pylint
from SRC import train_pylint
#from SRC import inference_pylint


# 1) Se ejecuta código de procesamiento/preparación de Datos
prep_pylint.prep_code()
# 2) Se ejecuta código de entrenamiento/fitting del moelo
train_pylint.train_code()
# 3) Se ejecuta código de scoring/predict del modelo
#inferencepylint.inference_code()