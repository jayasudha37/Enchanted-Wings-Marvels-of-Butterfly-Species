# Enchanted Wings: Marvels of Butterfly Species

A Flask web application that uses a trained TensorFlow/Keras image-classification model to predict butterfly species from uploaded images.

## Project files

```text
Enchanted_Wings/
├── app.py
├── requirements.txt
├── my_model.h5
├── class_indices.json
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── uploads/
```

## Setup (Windows CMD)

Create and activate a virtual environment:

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy your trained `my_model.h5` and the exact class mapping `class_indices.json` into the project root. Then run:

```cmd
python app.py
```

Open `http://127.0.0.1:5000`.

## Class-label mapping

The number and order of labels must match the model's output neurons. Preferred JSON example:

```json
{
  "Monarch": 0,
  "Blue Morpho": 1
}
```

The example above is illustrative only. Do not use it as the real 75-class mapping unless those are the actual labels and indices from training. Export the mapping from the same training generator used to train the model.

## Image preprocessing

The default is `mobilenet_v2`. This is correct only if the model was trained with `tf.keras.applications.mobilenet_v2.preprocess_input`.

If training used `ImageDataGenerator(rescale=1./255)`, set the environment variable before running:

```cmd
set PREPROCESS_MODE=rescale
python app.py
```

If training used raw pixel values, use `set PREPROCESS_MODE=none`.

Use the same image size, color mode, and preprocessing at inference as during training.

## Deployment

For a Linux-based host such as Render, use:

```bash
gunicorn app:app --workers 1 --timeout 180
```

Set `FLASK_SECRET_KEY` to a long random secret in the hosting provider's environment variables. Add `PREPROCESS_MODE` if it differs from the default.

## Important model limitations

- The application displays the model's top class and top-three scores. These scores are not necessarily calibrated probabilities.
- A closed-set classifier will still choose one of its known classes for an unrelated image. Do not describe it as detecting unknown species unless an out-of-distribution rejection method has been implemented and evaluated.
- The included species descriptions are only populated for a few common labels. Add verified facts for the exact labels in your dataset.
- Evaluate the model on a separate held-out test set and report actual accuracy, precision, recall, F1-score, and confusion matrix results in your project documentation.
