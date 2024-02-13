from SRC import Prep
from SRC import Train
from SRC import Inference


# 1) Se ejecuta código de procesamiento/preparación de Datos
Prep.prep_code()
# 2) Se ejecuta código de entrenamiento/fitting del moelo
Train.train_code()
# 3) Se ejecuta código de scoring/predict del modelo
Inference.inference_code()