from flask import Flask, request, jsonify
import easyocr
import base64
import cv2
import numpy as np
import time

app = Flask(__name__)

# Cargar EasyOCR solo una vez
reader = easyocr.Reader(['es'], gpu=False)


def preprocess(image):
    """Mejora contraste y escala para acelerar OCR"""
    # Escalar si es muy grande (máx 800px ancho)
    h, w = image.shape[:2]
    if w > 800:
        scale = 800 / w
        image = cv2.resize(image, (800, int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # mejorar contraste
    return gray


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

        start = time.time()

        # Preprocesar y correr OCR una sola vez
        pre_img = preprocess(image)
        results = reader.readtext(pre_img, detail=1)

        end = time.time()

        # Convertir resultados a JSON
        output = []
        for (coords, text, conf) in results:
            coords_py = [[int(px), int(py)] for [px, py] in coords]
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
    app.run(host="0.0.0.0", port=5001, debug=False)
