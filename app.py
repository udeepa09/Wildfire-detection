from flask import Flask, render_template, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import base64
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the same model you used in Home.py
MODEL_PATH = "fire-models/fire_n.pt"
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get data from HTML
    file = request.files['image'].read()
    conf = float(request.form.get('conf', 0.20))
    iou = float(request.form.get('iou', 0.50))
    
    # 2. Process Image
    img = Image.open(io.BytesIO(file))
    results = model.predict(img, conf=conf, iou=iou)
    
    # 3. Draw Bounding Boxes
    res_plotted = results[0].plot()
    
    # 4. Convert to Base64 (to send back to HTML)
    res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 5. Count detections
    count = len(results[0].boxes)
    latency = round(sum(results[0].speed.values()) / 1000, 2)
    
    return jsonify({
        'image': img_base64,
        'message': f"Detected {count} objects in {latency} seconds."
    })

if __name__ == '__main__':
    app.run(debug=True)