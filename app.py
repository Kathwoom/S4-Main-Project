from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.models import load_model
import os
import numpy as np

app = Flask(__name__)

# Define the paths to your models
model_paths = {
    "VGG-19": "model1.h5",
    "Inception V3": "model2.h5",
    "DenseNet201": "model3.h5"
}

# Load all models
models = {name: load_model(path) for name, path in model_paths.items()}

# Function to get X-ray type based on class index
def Get_Xray_Type(argument):
    switcher = {
        1: "NORMAL",
        0: "TUBERCULOSIS",
    }
    return switcher.get(argument, "Invalid X-ray")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('index.html', model_names=model_paths.keys())

@app.route('/predict_single', methods=['POST'])
def predict_single():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        img_path = os.path.join('uploads', filename)
        file.save(img_path)

        # Preprocess the uploaded image for single model prediction
        img = image.load_img(img_path, target_size=(196, 196))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        selected_model_name = request.form['model']  # Get the selected model name
        selected_model = models[selected_model_name]  # Load the selected model

        # Preprocess input based on selected model
        if selected_model_name == "VGG-19":
            img_array = preprocess_input_vgg19(img_array)
        elif selected_model_name == "Inception V3":
            img_array = preprocess_input_inceptionv3(img_array)
        elif selected_model_name == "DenseNet201":
            img_array = preprocess_input_densenet(img_array)

        # Make prediction using the selected model
        prediction = selected_model.predict(img_array)
        predicted_label = np.argmax(prediction)
        xray_type = Get_Xray_Type(predicted_label)

        return render_template('predict_single.html', filename=filename, prediction=xray_type)

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        img_path = os.path.join('uploads', filename)
        file.save(img_path)

        # Preprocess the uploaded image for multiple model predictions
        img = image.load_img(img_path, target_size=(196, 196))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = {}
        for model_name, model in models.items():
            # Preprocess input based on selected model
            if model_name == "VGG-19":
                img_array_preprocessed = preprocess_input_vgg19(img_array)
            elif model_name == "Inception V3":
                img_array_preprocessed = preprocess_input_inceptionv3(img_array)
            elif model_name == "DenseNet201":
                img_array_preprocessed = preprocess_input_densenet(img_array)

            # Make prediction using the selected model
            prediction = model.predict(img_array_preprocessed)
            predicted_label = np.argmax(prediction)
            xray_type = Get_Xray_Type(predicted_label)
            predictions[model_name] = xray_type

        return render_template('predict_multiple.html', filename=filename, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
