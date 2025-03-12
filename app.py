import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
import gdown
import os
from collections import OrderedDict  # Import OrderedDict
import json

app = Flask(__name__)

# Google Drive file ID of the model
GDRIVE_MODEL_URL = "https://drive.google.com/uc?export=download&id=12JG_GpwlSdxyCI6ogX3_hTgDwmxRVJLR"
MODEL_PATH = "plant_disease_model_with_aug.h5"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_MODEL_URL, MODEL_PATH, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def preprocess_image(image):
    """
    Convert uploaded image to the same format used in Streamlit.
    """
    image = image.resize((224, 224))  # Resize to model input size
    input_arr = img_to_array(image)  # Convert to array
    input_arr = np.array([input_arr])  # Convert single image to batch
    return input_arr

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Plant Disease Recognition API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load the image exactly like in Streamlit
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        processed_image = preprocess_image(image)

        # Run prediction
        prediction = model.predict(processed_image)
        result_index = np.argmax(prediction)
        predicted_class = class_names[result_index]
        accuracy = prediction[0][result_index] * 100  # Convert to percentage

        # Ensure "prediction" comes first
        response = OrderedDict([
            ("prediction", predicted_class),
            ("accuracy", f"{accuracy:.2f}%")
        ])

        return app.response_class(
            response=json.dumps(response),  # Convert OrderedDict to JSON string
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
