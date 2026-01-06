import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow as tf
import os

app = Flask("Clothes Classification")
model = tf.keras.models.load_model('xception_06_0.720.h5')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img = load_img(file_path, target_size=(299, 299))
    
    X = np.array([img])
    X = preprocess_input(X)

    predictions = model.predict(X)
    scores = predictions[0]

    results = list(zip(classes, scores))
    results.sort(key=lambda x: x[1], reverse=True) 

    os.remove(file_path)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)