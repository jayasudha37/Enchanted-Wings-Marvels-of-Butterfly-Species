
import os
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "my_model.h5"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Butterfly classes: must match the training dataset order.
CLASS_NAMES = [
    "ADONIS", "AFRICAN GIANT SWALLOWTAIL", "AMERICAN SNOOT",
    "AN 88", "APPOLLO", "ATALA", "BANDED ORANGE HELICONIAN",
    "BANDED PEACOCK", "BECKERS WHITE", "BLACK HAIRSTREAK",
    "BLUE MORPHO", "BLUE SPOTTED CROW", "BROWN SIPROETA",
    "CABBAGE WHITE", "CAIRNS BIRDWING", "CHECQUERED SKIPPER",
    "CHESTNUT", "CLEOPATRA", "CLODIUS PARNASSIAN",
    "CLOUDED SULPHUR", "COMMON BANDED AWL",
    "COMMON WOOD-NYMPH", "COPPER TAIL", "CRECENT",
    "CRIMSON PATCH", "DANAID EGGFLY", "EASTERN COMA",
    "EASTERN DAPPLE WHITE", "EASTERN PINE ELFIN",
    "ELBOWED PIERROT", "GOLD BANDED", "GREAT EGGFLY",
    "GREAT JAY", "GREEN CELLED CATTLEHEART", "GREY HAIRSTREAK",
    "INDRA SWALLOW", "IPHICLUS SISTER", "JULIA", "LARGE MARBLE",
    "MALACHITE", "MANGROVE SKIPPER", "MESTRA", "METALMARK",
    "MILBERTS TORTOISESHELL", "MONARCH", "MOURNING CLOAK",
    "ORANGE OAKLEAF", "ORANGE TIP", "ORCHARD SWALLOW",
    "PAINTED LADY", "PAPER KITE", "PEACOCK", "PINE WHITE",
    "PIPEVINE SWALLOW", "POPINJAY", "PURPLE HAIRSTREAK",
    "PURPLISH COPPER", "QUESTION MARK", "RED ADMIRAL",
    "RED CRACKER", "RED POSTMAN", "RED SPOTTED PURPLE",
    "SCARCE SWALLOW", "SILVER SPOT SKIPPER", "SLEEPY ORANGE",
    "SOOTYWING", "SOUTHERN DOGFACE", "STRAITED QUEEN",
    "TROPICAL LEAFWING", "TWO BARRED FLASHER", "ULYSES",
    "VICEROY", "WOOD SATYR", "YELLOW SWALLOW TAIL",
    "ZEBRA LONG WING"
]

# Verified example information for selected species.
SPECIES_INFO = {
    "MONARCH": {
        "scientific_name": "Danaus plexippus",
        "family": "Nymphalidae",
        "features": [
            "Orange wings with black veins",
            "Black wing borders with white spots"
        ],
        "description": "Known for its long-distance migration."
    },
    "BLUE MORPHO": {
        "scientific_name": "Morpho menelaus",
        "family": "Nymphalidae",
        "features": [
            "Large, iridescent blue upper wings",
            "Brown undersides with eye-like markings"
        ],
        "description": "Commonly associated with tropical forests."
    },
    "PAINTED LADY": {
        "scientific_name": "Vanessa cardui",
        "family": "Nymphalidae",
        "features": [
            "Orange and brown patterned wings",
            "Black and white wing-tip markings"
        ],
        "description": "A widely distributed migratory butterfly."
    },
    "RED ADMIRAL": {
        "scientific_name": "Vanessa atalanta",
        "family": "Nymphalidae",
        "features": [
            "Dark wings with red-orange bands",
            "White spots near the wing tips"
        ],
        "description": "Often found in gardens and woodland edges."
    },
    "PAPER KITE": {
        "scientific_name": "Idea leuconoe",
        "family": "Nymphalidae",
        "features": [
            "Pale wings with dark markings",
            "Distinctive dark wing veins"
        ],
        "description": "Often associated with tropical habitats."
    },
    "ZEBRA LONG WING": {
        "scientific_name": "Heliconius charithonia",
        "family": "Nymphalidae",
        "features": [
            "Long, narrow wings",
            "Distinctive black-and-white stripes"
        ],
        "description": "Often visits flowers for nectar."
    }
}

# Load the model once when the application starts.
print("Loading butterfly model...")

try:
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("Model loaded successfully!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model output classes: {model.output_shape[-1]}, "
            f"but configured class names: {len(CLASS_NAMES)}. "
            "Check the original training labels."
        )

except Exception:
    model = None
    app.logger.exception("Could not load the butterfly model")


def preprocess_image(image):
    """Prepare an RGB image for the trained classifier."""
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


def render_page(**kwargs):
    data = {
        "species": None,
        "confidence": None,
        "image_url": None,
        "scientific_name": None,
        "family": None,
        "features": [],
        "description": None,
        "error_message": None
    }
    data.update(kwargs)

    return render_template("index.html", **data)


@app.route("/", methods=["GET"])
def index():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_page(
            error_message="Model could not be loaded. Check the terminal."
        )

    uploaded_file = request.files.get("image")

    if not uploaded_file or not uploaded_file.filename:
        return render_page(
            error_message="Please select a butterfly image."
        )

    extension = Path(uploaded_file.filename).suffix.lower()

    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        return render_page(
            error_message="Please upload a JPG, PNG, or WEBP image."
        )

    try:
        with Image.open(uploaded_file.stream) as source:
            source.load()
            image = source.convert("RGB")

        processed_image = preprocess_image(image)

        predictions = np.asarray(
            model.predict(processed_image, verbose=0)
        )[0].reshape(-1)

        if len(predictions) != len(CLASS_NAMES):
            raise ValueError("Model output does not match CLASS_NAMES.")

        predicted_index = int(np.argmax(predictions))
        species = CLASS_NAMES[predicted_index]

        # Correct when the model output contains probabilities.
        confidence = float(predictions[predicted_index]) * 100

        filename = f"{uuid.uuid4().hex}.jpg"
        image.save(UPLOAD_FOLDER / filename, format="JPEG", quality=92)

        info = SPECIES_INFO.get(species, {})

        return render_page(
            species=species.title(),
            confidence=confidence,
            image_url=f"uploads/{filename}",
            scientific_name=info.get(
                "scientific_name", "Information unavailable"
            ),
            family=info.get(
                "family", "Information unavailable"
            ),
            features=info.get("features", []),
            description=info.get("description")
        )

    except (UnidentifiedImageError, OSError):
        return render_page(
            error_message="Invalid image. Please upload another photograph."
        )

    except Exception:
        app.logger.exception("Prediction failed")
        return render_page(
            error_message="Prediction failed. Check the terminal for details."
        )


@app.errorhandler(413)
def file_too_large(error):
    return render_page(
        error_message="Image size must not exceed 10 MB."
    ), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
