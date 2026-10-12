
import os

# Set these before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import logging
import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

# Maximum uploaded image size: 8 MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Reduce CPU thread usage
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

logging.basicConfig(level=logging.INFO)

# Preserve the model's original class order
CLASS_NAMES = [
    'ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT',
    'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN',
    'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK',
    'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA',
    'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER',
    'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN',
    'CLOUDED SULPHUR', 'COMMON BANDED AWL',
    'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT',
    'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA',
    'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN',
    'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY',
    'GREAT JAY', 'GREEN CELLED CATTLEHEART',
    'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER',
    'JULIA', 'LARGE MARBLE', 'MALACHITE',
    'MANGROVE SKIPPER', 'MESTRA', 'METALMARK',
    'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK',
    'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW',
    'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE',
    'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK',
    'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL',
    'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE',
    'SCARCE SWALLOW', 'SILVER SPOT SKIPPER',
    'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE',
    'STRAITED QUEEN', 'TROPICAL LEAFWING',
    'TWO BARRED FLASHER', 'ULYSES', 'VICEROY',
    'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING'
]

# Load the model once when the application starts
print("Loading Keras model...", flush=True)

try:
    model = tf.keras.models.load_model(
        "my_model.h5",
        compile=False
    )

    print("Model loaded successfully!", flush=True)
    print("Model input shape:", model.input_shape, flush=True)
    print("Model output shape:", model.output_shape, flush=True)

except Exception:
    app.logger.exception("Could not load the butterfly model")
    model = None


def preprocess_image(file):
    """Convert the uploaded image to a model-ready RGB array."""

    with Image.open(file.stream) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        image_array = np.asarray(img, dtype=np.float32)

    # This normalization must match the model's training preprocessing.
    image_array = image_array / 255.0

    # Shape: (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return render_template(
            "index.html",
            error="The prediction model could not be loaded."
        ), 500

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please select a butterfly image."
        ), 400

    file = request.files["image"]

    if not file.filename:
        return render_template(
            "index.html",
            error="Please select an image first."
        ), 400

    try:
        # Prepare the uploaded image
        image_array = preprocess_image(file)

        # Run inference directly
        output = model(image_array, training=False)
        predictions = np.asarray(output)[0]

        if predictions.ndim != 1:
            raise ValueError("Unexpected model output dimensions.")

        if len(predictions) != len(CLASS_NAMES):
            raise ValueError(
                f"Model returned {len(predictions)} classes, "
                f"but CLASS_NAMES contains {len(CLASS_NAMES)}."
            )

        predicted_index = int(np.argmax(predictions))
        predicted_species = CLASS_NAMES[predicted_index]

        app.logger.info(
            "Prediction completed: %s",
            predicted_species
        )

        # Display only the predicted species name
        return render_template(
            "index.html",
            prediction_text=predicted_species
        )

    except (UnidentifiedImageError, OSError):
        return render_template(
            "index.html",
            error="Please upload a valid image file."
        ), 400

    except Exception:
        app.logger.exception("Butterfly prediction failed")

        return render_template(
            "index.html",
            error="Prediction failed. Please try another image."
        ), 500


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="Image is too large. Please upload an image under 8 MB."
    ), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
