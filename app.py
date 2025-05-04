from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import tensorflow as tf

tf.keras.backend.clear_session()
MODEL_PATH = r'C:\Users\HAMZA JABBAR\Desktop\Cancer Flask Project\skin_cancer_model.h5'
model = load_model(MODEL_PATH)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    
    image = cv2.imread(filepath) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.resize(image, (128, 128)) 
    image = image.astype('float32') / 255.0 
    image = np.expand_dims(image, axis=0) 

    
    prediction = model.predict(image)[0][0]
    result = "Melanoma" if prediction > 0.5 else "Non-Melanoma"

    return render_template('result.html', result=result, image=file.filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
