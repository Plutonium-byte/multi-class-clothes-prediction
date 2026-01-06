import numpy as np
from tensorflow.keras.utlis import load_img
from tensorflow import keras

model = keras.models.load_model('model.h5')

path = "path/to/image"
img = load_img(path, target_size=(299, 299))

X = np.array([img])
X = preprocess_input(X)

predictions = model.predict(X)

classes = [
  'Blazer',
  'Celana_Panjang',
  'Celana_Pendek',
  'Gaun',
  'Hoodie',
  'Jaket',
  'Jaket_Denim',
  'Jaket_Olahraga',
  'Jeans',
  'Kaos',
  'Kemeja',
  'Mantel',
  'Polo',
  'Rok',
  'Sweter'
]

scores = predictions[0]
results = [(classes[i], float(score)) for i, score in enumerate(scores)]
results.sort(key=lambda x: x[1], reverse=True)
output = {name: score for name, score in results}

print(output)