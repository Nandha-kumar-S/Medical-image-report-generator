import io
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from scripts.infer_mlp import generate_report, load_mlp_model

app = Flask(__name__)

# Paths relative to this file so the app runs on any OS.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model', 'mlp_model.pkl')

mlp_model = load_mlp_model(MODEL_PATH)

# The same encoder used to build the training features (see
# data/train_image_enc.py). MobileNetV2 with average pooling gives a 1280-d
# vector per view, so frontal + lateral concatenated is the 2560-d input the
# MLP was trained on. Both views are therefore required at inference.
image_encoder = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
TARGET_SIZE = (224, 224)


def encode_image(file_storage):
    """Uploaded file -> 1280-d MobileNetV2 feature vector."""
    image = Image.open(io.BytesIO(file_storage.read())).convert('RGB')
    image = image.resize(TARGET_SIZE)
    batch = preprocess_input(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0))
    return image_encoder.predict(batch, verbose=0)[0]


@app.route('/')
def index():
    return 'MEDICAL IMAGE REPORT GENERATION MODEL'


@app.route('/predict', methods=['POST'])
def predict():
    if mlp_model is None:
        return jsonify({'error': f'Model not loaded from {MODEL_PATH}'}), 500

    missing = [v for v in ('frontal', 'lateral') if v not in request.files]
    if missing:
        return jsonify({
            'error': f"Missing required image(s): {', '.join(missing)}. "
                     "The model needs both a frontal and a lateral view."
        }), 400

    try:
        frontal = encode_image(request.files['frontal'])
        lateral = encode_image(request.files['lateral'])
    except Exception as exc:
        return jsonify({'error': f'Could not read image: {exc}'}), 400

    features = np.concatenate((frontal, lateral)).reshape(1, -1)
    prediction = generate_report(mlp_model, features)

    if prediction is None:
        return jsonify({'error': 'Prediction failed'}), 500

    # NOTE: this returns the predicted class index, not the impression text.
    # The LabelEncoder fitted during training is not persisted, so there is
    # nothing to map the index back to its string. See README.
    return jsonify({'impression_class': int(np.argmax(prediction))})


if __name__ == '__main__':
    app.run(debug=True)
