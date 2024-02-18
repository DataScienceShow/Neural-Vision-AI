import tensorflow as tf
import io
import os
import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request

# Load the saved ML model
model_path = r'C:\Data Scients\Hackatons\AI_Hackathon_17_02_2024\ML model\imageclassifier.h5'
model = tf.keras.models.load_model(model_path)

# Initialize Flask app
app = Flask(__name__)

# Define route for uploading image and classification


@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if image file is present in the request
    if 'file' not in request.files:
        return 'No file uploaded'

    # Get the image file from the request
    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return 'No file selected'

    # Check if the file is allowed based on its extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
    if not file.filename.lower().split('.')[-1] in allowed_extensions:
        return 'Invalid file format'

    # Save the uploaded image temporarily
    temp_image_path = 'temp_image.jpg'
    file.save(temp_image_path)

    # Read and preprocess the image
    img = cv2.imread(temp_image_path)
    resized_img = tf.image.resize(img, (256, 256))
    processed_img = np.expand_dims(resized_img / 255, 0)

    # Make prediction using the loaded model
    prediction = model.predict(processed_img)

    # Determine classification result based on prediction
    if prediction < 0.5:
        result = 'Bored'
    else:
        result = 'Engaged'

    # Delete the temporary image file
    os.remove(temp_image_path)

    # Return the classification result
    return result


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
