import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS  
import numpy as np
from PIL import Image, ImageOps
import pickle
from skimage.feature import hog
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define an endpoint for a POST request to make predictions
HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>Image Submission</title>
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
        input[type="file"] {
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
        <input type="file" name="image_file" accept="image/*">
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
    image_file = request.files.get('image_file')
    if image_file:
        image = Image.open(image_file)
        user_input_gray = ImageOps.grayscale(image)
        user_input_resized = user_input_gray.resize((64, 64))
        user_input_array = np.array(user_input_resized)
        user_input_features = hog(user_input_array, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
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
        return jsonify({'error': 'No image file provided'})

@app.route('/', methods=['POST','GET'])
def submit_image():
    if request.method == 'GET':
        return render_template_string(HTML_FORM)
    
    if 'image_file' not in request.files:
        return render_template_string(HTML_FORM, error='No file part')
    
    image_file = request.files['image_file']
    if image_file.filename == '':
        return render_template_string(HTML_FORM, error='No selected file')
    
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)
    
    response = requests.post('http://127.0.0.1:5000/api/predictions', files={'image_file': open(image_path, 'rb')})
    
    if response.ok:
        result = response.json()
        prediction = result.get('prediction')
        error = result.get('error')
        return render_template_string(HTML_FORM, prediction=prediction, error=error)
    else:
        return render_template_string(HTML_FORM, error='Failed to get a response from the prediction API.')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
