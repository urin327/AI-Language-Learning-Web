from flask import Flask, request, jsonify, render_template
import cv2
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import numpy as np

app = Flask(__name__)

model = YOLO("yolov8m.pt")
translator = GoogleTranslator(source="en", target="ko")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    file = request.files["image"]

    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    results = model(frame)

    labels = []
    seen = set()

    for r in results:
        for box in r.boxes:

            confidence = float(box.conf[0])

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # duplicate арилгах
            if label in seen:
                continue

            seen.add(label)

            try:
                korean = translator.translate(label)
            except:
                korean = label

            labels.append({
                "english": label,
                "korean": korean,
                "confidence": round(confidence * 100, 1)
            })

    return jsonify(labels)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)