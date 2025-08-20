ejecutar 


python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt

para ejecutar el codigo es: 
python easy_ocr.py --image images/auto006econ.jpg --langs es
            ^                          ^                   ^
      acrchivo PY a ejecutar         imagen que         lenguaje a usar en ocr(dejar predeterminado el espanol)
                                  se quiera analizar
