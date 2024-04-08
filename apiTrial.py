from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image, ImageOps
import pickle
from skimage.feature import hog
import pandas as pd
import requests

app = Flask(__name__)
# Define an endpoint for a POST request to make predictions
HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>Image Path Submission</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            flex-direction: column;
            background-color: #FFC299;
        }
        form {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 300px; /* Adjust the width as needed */
        }
        input[type="text"] {
            display: flex;
            flex-direction: column;
            width: 100%;
            margin-bottom: 10px;
            padding: 8px;
            align-items: center;
            font-size: 16px;
        }
        button[type="submit"] {
            display: flex;
            flex-direction: column;
            background-color: #DE4B00;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-color: white;
            cursor: pointer;
        }
        .prediction-container {
            text-align: center; /* Center align the prediction message */
            display: flex;
            align-items: center;
            font-size: 24px;
        }
    </style>
</head>
<body>
    <h1 class="title">ML Model Prediction using Lung X-Ray detecting COVID, Pneumonia etc</h1>
    <form action="/" method="post" enctype="multipart/form-data">
        <input type="text" name="image_path" placeholder="Enter image path (e.g., /path/to/image.jpg)"/>
        <button type="submit">Predict</button>
    </form>
    <div class="prediction-container">
        {% if prediction %}
            <p>Prediction: {{ prediction }}</p>
        {% endif %}
        {% if error %}
            <p>Error: {{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
'''

with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

@app.route('/api/predictions', methods=['POST'])
def make_prediction():
    image_path = request.form.get('image_path')
    #image_path = 'user_image/COVID-18_test1.png'
    print(image_path)

    def extract_hog_features(image):
        user_input_gray = np.array(image)
        user_features = hog(user_input_gray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), transform_sqrt=True,
                block_norm='L2-Hys')
        return user_features

    def extract_features_from_user_input(image_path):
        try:
            with open(image_path, 'rb') as img_file:
                image_data = Image.open(img_file)
                img_gray = ImageOps.grayscale(image_data)
                img_resized = img_gray.resize((64, 64))
                user_hog_features = extract_hog_features(img_resized)
                return user_hog_features

        except Exception as e:
            print("Error:", e)
            return None
        
    user_input_features = extract_features_from_user_input(image_path)
    if user_input_features is not None:
        user_input_features_reshaped = user_input_features.reshape(1, -1)
        prediction = svm_model.predict(user_input_features_reshaped)
        label_mapping = {
            0: 'Normal',
            1: 'COVID',
            2: 'Lung Opacity',
            3: 'Viral Pneumonia'
        }
        prediction_label = label_mapping.get(prediction[0], 'Unknown')
        return jsonify({'prediction': prediction_label})
    else:
        return jsonify({'error': 'Check the  image file location'})

@app.route('/', methods=['POST','GET'])
def submit_image():
    if request.method == 'GET':
        return render_template_string(HTML_FORM)
    image_path = request.form['image_path']
    response = requests.post('http://127.0.0.1:5000/api/predictions', data={'image_path': image_path})
    if response.ok:
        result = response.json()
        prediction = result.get('prediction')
        error = result.get('error')
        return render_template_string(HTML_FORM, prediction=prediction, error=error)
    else:
        return render_template_string(HTML_FORM, error='Failed to get a response from the prediction API.')

if __name__ == '__main__':
    app.run(debug=True)
