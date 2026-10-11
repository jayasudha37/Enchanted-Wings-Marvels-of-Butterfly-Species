
import os
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

CLASS_NAMES = [
    'ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT',
    'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN',
    'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK',
    'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA',
    'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER',
    'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN',
    'CLOUDED SULPHUR', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH',
    'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY',
    'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN',
    'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY',
    'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK',
    'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE',
    'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK',
    'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK',
    'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW',
    'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE',
    'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK',
    'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL',
    'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE',
    'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SLEEPY ORANGE',
    'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN',
    'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES',
    'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL',
    'ZEBRA LONG WING'
]

# Add verified details here as you expand your species guide.
# These entries are illustrative descriptions, not model-generated facts.
SPECIES_INFO = {
    "MONARCH": {
        "scientific_name": "Danaus plexippus",
        "family": "Nymphalidae",
        "features": ["Orange wings with black veins", "Black wing borders with white spots"],
        "note": "Known for its long-distance migration."
    },
    "BLUE MORPHO": {
        "scientific_name": "Morpho species",
        "family": "Nymphalidae",
        "features": ["Bright iridescent blue upper wings", "Broad wings with dark margins"],
        "note": "The exact scientific species may require a more specific model label."
    },
    "PAINTED LADY": {
        "scientific_name": "Vanessa cardui",
        "family": "Nymphalidae",
        "features": ["Orange and dark patterned wings", "White and black markings near the forewing tips"],
        "note": "A widespread butterfly found across many regions."
    },
    "PAPER KITE": {
        "scientific_name": "Idea leuconoe",
        "family": "Nymphalidae",
        "features": ["Large pale wings", "Distinctive black veins and spots"],
        "note": "Also known as the rice paper butterfly."
    },
    "RED ADMIRAL": {
        "scientific_name": "Vanessa atalanta",
        "family": "Nymphalidae",
        "features": ["Dark wings", "Distinctive red-orange bands and white forewing spots"],
        "note": "A widely distributed butterfly."
    },
    "ZEBRA LONG WING": {
        "scientific_name": "Heliconius charithonia",
        "family": "Nymphalidae",
        "features": ["Long narrow wings", "Contrasting black and pale yellow stripes"],
        "note": "Known for its elongated striped wings."
    },
    "VICEROY": {
        "scientific_name": "Limenitis archippus",
        "family": "Nymphalidae",
        "features": ["Orange wings", "Black veins and a dark line across the hindwings"],
        "note": "Resembles the monarch but has a distinctive hindwing line."
    },
    "MOURNING CLOAK": {
        "scientific_name": "Nymphalis antiopa",
        "family": "Nymphalidae",
        "features": ["Dark maroon-brown wings", "Pale yellow margins and blue spots"],
        "note": "Recognized by its contrasting wing border."
    }
}


def load_model():
    model_path = BASE_DIR / "my_model.h5"
    print("Loading Keras model...")

    try:
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return loaded_model
    except Exception as error:
        print(f"Error loading model: {error}")
        return None


model = load_model()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(image):
    # Convert grayscale/RGBA images to the 3-channel format expected by the model.
    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array / 255.0


def get_species_info(species):
    info = SPECIES_INFO.get(species)

    if info:
        return {
            "scientific_name": info["scientific_name"],
            "family": info["family"],
            "features": info["features"],
            "note": info["note"]
        }

    # Do not invent a scientific name or species-specific characteristics.
    return {
        "scientific_name": "Details not yet verified",
        "family": "Butterfly classification",
        "features": [
            f"Model classified this image as {species.title()}",
            "Compare wing patterns and colours for confirmation"
        ],
        "note": "Species-specific reference details have not yet been added."
    }


def render_page(image_url=None, prediction_text=None,
                species_name=None, confidence=None, error=None):
    species_info = get_species_info(species_name) if species_name else None

    return render_template(
        "index.html",
        image_url=image_url,
        prediction_text=prediction_text,
        species_name=species_name,
        confidence=confidence,
        species_info=species_info,
        error=error
    )


@app.route("/", methods=["GET"])
def index():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("image")

    if uploaded_file is None or uploaded_file.filename == "":
        return render_page(error="Please choose a butterfly image first.")

    if not allowed_file(uploaded_file.filename):
        return render_page(error="Please upload a JPG, PNG, or WEBP image.")

    if model is None:
        return render_page(error="The model could not be loaded. Check the server logs.")

    saved_path = None

    try:
        # Validate that the uploaded file really is an image.
        image = Image.open(uploaded_file.stream)
        image.load()
        image = image.convert("RGB")

        # Keep the uploaded image on disk so it can be shown after prediction.
        extension = secure_filename(uploaded_file.filename).rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{extension}"
        saved_path = UPLOAD_FOLDER / unique_filename
        image.save(saved_path, format="JPEG" if extension in {"jpg", "jpeg"} else extension.upper())

        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)

        scores = np.asarray(predictions[0]).reshape(-1)

        if len(scores) != len(CLASS_NAMES):
            raise ValueError(
                f"Model returned {len(scores)} scores, but "
                f"{len(CLASS_NAMES)} class names are configured."
            )

        predicted_index = int(np.argmax(scores))
        species_name = CLASS_NAMES[predicted_index]
        confidence = float(scores[predicted_index]) * 100.0

        return render_page(
            image_url=url_for("static", filename=f"uploads/{unique_filename}"),
            prediction_text=species_name.title(),
            species_name=species_name,
            confidence=confidence
        )

    except (UnidentifiedImageError, OSError):
        if saved_path and saved_path.exists():
            saved_path.unlink()
        return render_page(error="The selected file could not be read as an image.")

    except Exception as error:
        app.logger.exception("Butterfly prediction failed")
        if saved_path and saved_path.exists():
            saved_path.unlink()
        return render_page(error=f"Prediction failed: {error}")


@app.errorhandler(413)
def file_too_large(_error):
    return render_page(error="The image is too large. Please choose an image under 10 MB."), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
