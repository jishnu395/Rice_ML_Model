from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 64      # Change to your training image size

model = joblib.load("rice_model.pkl")
encoder = joblib.load("label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)

    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    feature = img.flatten().reshape(1, -1)

    pred = model.predict(feature)[0]

    class_name = encoder.inverse_transform([pred])[0]

    return render_template(
        "index.html",
        prediction=class_name,
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)