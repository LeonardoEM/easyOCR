el documento principal es el que diga    easy_ocr.py
el documento que funciona pero menos preciso en ocr es OCR_and_CV.py 


ejecutar 


python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt

para ejecutar el codigo es: 
python easy_ocr.py --image images/auto006econ.jpg --langs es
            ^                          ^                   ^
      acrchivo PY a ejecutar         imagen que         lenguaje a usar en ocr(dejar predeterminado el espanol)
                                  se quiera analizar
