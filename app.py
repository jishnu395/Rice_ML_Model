from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import os
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "rice_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    image = cv2.imread(filepath)

    if image is None:
        return "Unable to read uploaded image."

    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    feature = img.flatten().reshape(1, -1)

    pred = model.predict(feature)[0]
    class_name = encoder.inverse_transform([pred])[0]

    return render_template(
        "index.html",
        prediction=class_name,
        image_path=f"uploads/{filename}"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)