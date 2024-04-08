from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image, ImageOps
import pickle
#from keras.src.legacy.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import EarlyStopping
#from keras.preprocessing.image import ImageDataGenerator
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
</head>
<body>
    <form action="/" method="post" enctype="multipart/form-data">
        <input type="text" name="image_path" placeholder="Enter image path"/>
        <button type="submit">Predict</button>
    </form>
    {% if prediction %}
        <p>Prediction: {{ prediction }}</p>
    {% endif %}
    {% if error %}
        <p>Error: {{ error }}</p>
    {% endif %}
</body>
</html>
'''

#wrapper fn
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

@app.route('/api/predictions', methods=['POST'])
def make_prediction():
    #input image using post method

    # Retrieve the image data from the file
    image_path = request.form.get('image_path')
    #image_path = 'user_image/COVID-18_test1.png'
    print(image_path)

    def extract_hog_features(image):
    # Convert image to grayscale
        user_input_gray = np.array(image)

    # Compute HOG features
        user_features = hog(user_input_gray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), transform_sqrt=True,
                block_norm='L2-Hys')

        return user_features

    def extract_features_from_user_input(image_path):
        try:
            with open(image_path, 'rb') as img_file:
                image_data = Image.open(img_file)

                # Convert the image to grayscale
                img_gray = ImageOps.grayscale(image_data)

                # Resize the image to a fixed size (if needed)
                img_resized = img_gray.resize((64, 64))

                # Extract HOG features from the processed image
                user_hog_features = extract_hog_features(img_resized)

                return user_hog_features

        except Exception as e:
            print("Error:", e)
            return None

    # Example usage:
    user_input_features = extract_features_from_user_input(image_path)
    print(user_input_features)

    if user_input_features is not None:
        print("HOG features extracted successfully!")
        print("Shape of features:", user_input_features.shape)
        # Save or use these features for ML model later

        # Reshape the features to match the expected input shape of the models
        user_input_features_reshaped = user_input_features.reshape(1, -1)

        # Obtain predictions from all models
        prediction = svm_model.predict(user_input_features_reshaped)

        # Assuming you have already loaded and trained all the models (logistic_regression, logistic_regression_1, xgb_model, kn_clas, svm_model)
        label_mapping = {
            0: 'Normal',
            1: 'COVID',
            2: 'Lung Opacity',
            3: 'Viral Pneumonia'
        }

        # Map numerical predictions to their corresponding labels
        prediction_label = label_mapping.get(prediction[0], 'Unknown')

        # Display predictions for each model
        print('Support Vector Machine Model Prediction:', prediction_label)
        return jsonify({'prediction': prediction_label})

    else:
        return jsonify({'error': 'Failed to extract features from the image.'})

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
