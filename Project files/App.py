
import os
import uuid
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "my_model.h5")
CLASS_FILE = os.path.join(BASE_DIR, "classes.txt")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading butterfly classification model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")
print("Model output shape:", model.output_shape)


# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

CLASS_NAMES = [
    "ADONIS",
    "AFRICAN GIANT SWALLOWTAIL",
    "AMERICAN SNOOT",
    "AN 88",
    "APPOLLO",
    "ATALA",
    "BANDED ORANGE HELICONIAN",
    "BANDED PEACOCK",
    "BECKERS WHITE",
    "BLACK HAIRSTREAK",
    "BLUE MORPHO",
    "BLUE SPOTTED CROW",
    "BROWN SIPROETA",
    "CABBAGE WHITE",
    "CAIRNS BIRDWING",
    "CHECQUERED SKIPPER",
    "CHESTNUT",
    "CLEOPATRA",
    "CLODIUS PARNASSIAN",
    "CLOUDED SULPHUR",
    "COMMON BANDED AWL",
    "COMMON WOOD-NYMPH",
    "COPPER TAIL",
    "CRECENT",
    "CRIMSON PATCH",
    "DANAID EGGFLY",
    "EASTERN COMA",
    "EASTERN DAPPLE WHITE",
    "EASTERN PINE ELFIN",
    "ELBOWED PIERROT",
    "GOLD BANDED",
    "GREAT EGGFLY",
    "GREAT JAY",
    "GREEN CELLED CATTLEHEART",
    "GREY HAIRSTREAK",
    "INDRA SWALLOW",
    "IPHICLUS SISTER",
    "JULIA",
    "LARGE MARBLE",
    "MALACHITE",
    "MANGROVE SKIPPER",
    "MESTRA",
    "METALMARK",
    "MILBERTS TORTOISESHELL",
    "MONARCH",
    "MOURNING CLOAK",
    "ORANGE OAKLEAF",
    "ORANGE TIP",
    "ORCHARD SWALLOW",
    "PAINTED LADY",
    "PAPER KITE",
    "PEACOCK",
    "PINE WHITE",
    "PIPEVINE SWALLOW",
    "POPINJAY",
    "PURPLE HAIRSTREAK",
    "PURPLISH COPPER",
    "QUESTION MARK",
    "RED ADMIRAL",
    "RED CRACKER",
    "RED POSTMAN",
    "RED SPOTTED PURPLE",
    "SCARCE SWALLOW",
    "SILVER SPOT SKIPPER",
    "SLEEPY ORANGE",
    "SOOTYWING",
    "SOUTHERN DOGFACE",
    "STRAITED QUEEN",
    "TROPICAL LEAFWING",
    "TWO BARRED FLASHER",
    "ULYSES",
    "VICEROY",
    "WOOD SATYR",
    "YELLOW SWALLOW TAIL",
    "ZEBRA LONG WING"
]


def load_class_names():
    if os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, "r", encoding="utf-8") as file:
            names = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if len(names) == model.output_shape[-1]:
            return names

        raise ValueError(
            f"classes.txt contains {len(names)} labels, "
            f"but the model has {model.output_shape[-1]} outputs."
        )

    if len(CLASS_NAMES) == model.output_shape[-1]:
        return CLASS_NAMES

    raise ValueError(
        "Class names do not match the model output. "
        "Please check classes.txt."
    )


CLASS_NAMES = load_class_names()

if len(CLASS_NAMES) != model.output_shape[-1]:
    raise ValueError("Number of class names does not match model outputs.")


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    # Default preprocessing: MobileNetV2
    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_butterfly(image_path):
    processed_image = prepare_image(image_path)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    scores = np.asarray(predictions[0], dtype=np.float64)

    # Convert logits to probabilities if necessary
    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(np.sum(scores), 1.0, atol=0.01)
    ):
        scores = tf.nn.softmax(scores).numpy()

    predicted_index = int(np.argmax(scores))

    species = CLASS_NAMES[predicted_index]
    confidence = float(scores[predicted_index] * 100)

    return species, confidence


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, error=None)


# --------------------------------------------------
# PREDICTION ROUTE
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "butterfly_image" not in request.files:
        return render_template(
            "index.html",
            result=None,
            error="Please select a butterfly image."
        )

    file = request.files["butterfly_image"]

    if file.filename == "":
        return render_template(
            "index.html",
            result=None,
            error="Please choose an image before predicting."
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            result=None,
            error="Please upload JPG, JPEG, PNG or WEBP images only."
        )

    original_name = secure_filename(file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()

    unique_filename = f"{uuid.uuid4().hex}.{extension}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        file.save(image_path)

        # Verify that the uploaded file is a valid image
        with Image.open(image_path) as image:
            image.verify()

        species, confidence = predict_butterfly(image_path)

        result = {
            "species": species,
            "confidence": confidence,
            "image_url": url_for(
                "static",
                filename=f"uploads/{unique_filename}"
            ),
            "filename": original_name
        }

        return render_template(
            "index.html",
            result=result,
            error=None
        )

    except (UnidentifiedImageError, OSError):
        if os.path.exists(image_path):
            os.remove(image_path)

        return render_template(
            "index.html",
            result=None,
            error="The selected file is not a valid image."
        )

    except Exception as error:
        print("Prediction error:", error)

        if os.path.exists(image_path):
            os.remove(image_path)

        return render_template(
            "index.html",
            result=None,
            error="Unable to predict this image. Please try again."
        )


# --------------------------------------------------
# FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        result=None,
        error="Image size exceeds the 10 MB upload limit."
    ), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
