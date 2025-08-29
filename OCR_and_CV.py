import cv2
import numpy as np
from easyocr import Reader
from multiprocessing import Pool, cpu_count
import argparse

# --- Preprocesamiento ---
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# --- OCR por fragmento ---
def ocr_worker(args):
    image_fragment, langs, use_gpu = args
    reader = Reader(langs, gpu=use_gpu)
    return reader.readtext(image_fragment)

# --- División en regiones ---
def split_image(image, n_parts):
    h = image.shape[0]
    step = h // n_parts
    return [image[i*step:(i+1)*step, :] for i in range(n_parts)]

# --- Limpieza de texto ---
def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# --- Main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
    ap.add_argument("-l", "--langs", type=str, default="es,en", help="comma separated list of languages to OCR")
    ap.add_argument("-g", "--gpu", type=int, default=-1, help="whether or not GPU should be used")
    args = vars(ap.parse_args())

    langs = args["langs"].split(",")
    use_gpu = args["gpu"] > 0

    print(f"[INFO] OCR'ing with languages: {langs}, GPU: {use_gpu}")
    image = cv2.imread(args["image"])
    if image is None:
        raise ValueError(f"No se pudo abrir la imagen: {args['image']}")

    # Recorte opcional (ajústalo si lo necesitas)
    x_start, y_start, x_end, y_end = 100, 400, 800, 700
    image = image[y_start:y_end, x_start:x_end]

    image = preprocess(image)

    n_regions = min(cpu_count(), 4)
    fragments = split_image(image, n_regions)
    args_list = [(frag, langs, use_gpu) for frag in fragments]

    with Pool(processes=n_regions) as pool:
        results = pool.map(ocr_worker, args_list)

    # --- Dibujar resultados ---
    for region in results:
        for (bbox, text, prob) in region:
            print(f"[INFO] {prob:.4f}: {text}")
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            text = cleanup_text(text)
            cv2.rectangle(image, tl, br, (0, 255, 0), 2)
            cv2.putText(image, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("OCR Result", image)
    cv2.waitKey(0)