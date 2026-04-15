from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

IMG_SIZE             = 64
CONFIDENCE_THRESHOLD = 0.75   # below this on both sides → "Uncertain"

# ── Load CNN model ────────────────────────────────────────────────────────────
model = tf.keras.models.load_model("gender_model.h5")
print("CNN model loaded.")

def predict_image(image_path: str):
    """Return (prediction, male_pct, female_pct) or (None, None, None) on error."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # sigmoid output: 0 → male, 1 → female
    prob_female = float(model.predict(img, verbose=0)[0][0])
    prob_male   = 1.0 - prob_female

    if prob_male >= CONFIDENCE_THRESHOLD:
        prediction = "Male"
    elif prob_female >= CONFIDENCE_THRESHOLD:
        prediction = "Female"
    else:
        prediction = "Uncertain"

    return prediction, round(prob_male * 100, 2), round(prob_female * 100, 2)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/logo")
def logo():
    return send_from_directory(".", "logo.png")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    prediction, male_prob, female_prob = predict_image(filepath)

    if prediction is None:
        return jsonify({"error": "Could not read image. Please try another."})

    return jsonify({
        "prediction":        prediction,
        "male_probability":  f"{male_prob:.1f}%",
        "female_probability": f"{female_prob:.1f}%",
        "male_raw":          male_prob,
        "female_raw":        female_prob,
    })


if __name__ == "__main__":
    app.run(debug=True)
