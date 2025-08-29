import threading
from time import sleep
import random


def ejecutar():
    print(f'Comienza {threading.current_thread().name}')
    sleep((random.random()*10))  # esperamos un tiempo aleatorio entre 0 y 1 segundos
    print(f'Termina {threading.current_thread().name}')


# creamos los hilos
hilo1 = threading.Thread(target=ejecutar, name='Hilo 1')
hilo2 = threading.Thread(target=ejecutar, name='Hilo 2')

# ejecutamos los hilos
hilo1.start()
hilo2.start()

hilo1.join()
hilo2.join()

print('El hilo principal no espera por el resto de hilos.')