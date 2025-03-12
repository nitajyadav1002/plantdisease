import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "plant_disease_model_with_aug.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (same as Streamlit)
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

        return jsonify({"prediction": predicted_class})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
