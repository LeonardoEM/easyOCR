# import the necessary packages
from easyocr import Reader
import argparse
import cv2
import numpy as np
import time


start_time = time.time()


#metodo para la estala de grises en la imagen 
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

"""
	#quitarle el ruido a la imagen 
	def remove_noise(image):
    return cv2.medianBlur(image,5)

"""


def resize_image(image, scale_percent=50):
    """
    Reduce la resolución de la imagen según el porcentaje indicado.
    scale_percent: porcentaje de la resolución original (ej. 50 = reduce a la mitad)
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


#acomodar los taxtos de la imagen a un angulo correcto
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated



def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-l", "--langs", type=str, default="es",
	help="comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1,
	help="whether or not GPU should be used")
args = vars(ap.parse_args())

# break the input languages into a comma separated list
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))
# load the input image from disk
image = cv2.imread(args["image"])
if image is None:
    raise ValueError("Could not open or find the image: {}".format(args["image"]))
# OCR the input image using EasyOCR
x_start, y_start, x_end, y_end = 500, 900, 2500, 1400    #100, 400, 800, 700  cordenadas de las fotos de las primeras pruebas    500, 900, 2500, 1400 coordenadas para las fotos del 3er dia que tienen una resoliucion mas alta
cropped_img = image[y_start:y_end, x_start:x_end]
image = cropped_img

image= get_grayscale(image)  # convierte a escala de grises la imagen
#image = thresholding(image)  # aplica el umbral a la imagen
image = deskew(image)  # corrige el ángulo de la imagen
#image = remove_noise(image)  # Remove noise

image = resize_image(cropped_img, scale_percent=50)   #para reducir la resolucion de la imagen

print("[INFO] OCR'ing input image...")
reader = Reader(langs, gpu=args["gpu"] > 0)
results = reader.readtext(image)

# loop over the results
for (bbox, text, prob) in results:
	# display the OCR'd text and associated probability
	print("[INFO] {:.4f}: {}".format(prob, text))
	# unpack the bounding box
	(tl, tr, br, bl) = bbox
	tl = (int(tl[0]), int(tl[1]))
	tr = (int(tr[0]), int(tr[1]))
	br = (int(br[0]), int(br[1]))
	bl = (int(bl[0]), int(bl[1]))
	# cleanup the text and draw the box surrounding the text along
	# with the OCR'd text itself
	text = cleanup_text(text)
	cv2.rectangle(image, tl, br, (0, 255, 0), 2)
	cv2.putText(image, text, (tl[0], tl[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# show the output image

end_time = time.time()
print("[INFO] Tiempo de ejecución OCR: {:.2f} segundos".format(end_time - start_time))
cv2.imshow("Image", image)
cv2.waitKey(0)