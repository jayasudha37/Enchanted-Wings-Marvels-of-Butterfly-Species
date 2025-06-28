import os
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np

# --- Initialization ---
app = Flask(__name__)

# --- Configuration ---
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Load the Model (Do this once at startup) ---
print("Loading Keras model...")
try:
    model = tf.keras.models.load_model('my_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Define Your Classes ---
# IMPORTANT: These MUST be in the same order as your model's training output
CLASS_NAMES = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING']

# --- Image Preprocessing Function ---
def preprocess_image(img):
    """
    Takes an image, resizes it to 224x224, and prepares it for the model.
    """
    # Resize the image to the size your model expects
    img = img.resize((224, 224))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # If the image has an alpha channel (4 channels), remove it
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    # Add a "batch" dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image data (if your model was trained on normalized data)
    img_array = img_array / 255.0
    
    return img_array

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the image upload and prediction."""
    if model is None:
        return render_template('index.html', prediction_text="Error: Model is not loaded.")

    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file:
        try:
            # Open the image file
            img = Image.open(file.stream)
            
            # Preprocess the image
            processed_img = preprocess_image(img)
            
            # Make prediction
            predictions = model.predict(processed_img)
            
            # Get the top prediction
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class_name = CLASS_NAMES[predicted_class_index
            result_text = f"Prediction: {predicted_class_name} ({confidence:.2f}% confidence)"
            
            # Render the page again with the result
            return render_template('index.html', prediction_text=result_text)

        except Exception as e:
            return render_template('index.html', prediction_text=f"An error occurred: {e}")

    return redirect(url_for('index'))

# --- Main ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
