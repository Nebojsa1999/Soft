
from tensorflow.keras.models import load_model

file_path = './saved_model'
file_path2 = '.saved_model2'
model = load_model(file_path, compile = True)
model2 = load_model(file_path2, compile= True)