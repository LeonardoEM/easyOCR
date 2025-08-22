from flask import Flask, request, jsonify
import easyocr
import base64
import cv2
import numpy as np
import time

app = Flask(__name__)

# Inicializamos OCR (en español)
reader = easyocr.Reader(['es'], gpu=False)

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "Falta el campo 'image' en base64"}), 400

    try:
        # Decodificar base64
        image_data = base64.b64decode(data["image"])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Reducir resolución si es muy grande
        height, width = image.shape[:2]
        max_size = 800
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_w, new_h = int(width * scale), int(height * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convertir a gris para mejorar OCR
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # OCR
        start = time.time()
        results = reader.readtext(image, detail=1, paragraph=False)
        end = time.time()

        output = []
        for (coords, text, conf) in results:
            coords_py = [[int(x), int(y)] for [x, y] in coords]
            output.append({
                #"coords": coords_py,
                "text": text
                #"conf": float(conf)
            })

        return jsonify({
            "ocr_time": round(end - start, 2),
            "results": output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    cv2.setNumThreads(2)  # limitar hilos CPU
    app.run(host="0.0.0.0", port=5000, debug=True)
