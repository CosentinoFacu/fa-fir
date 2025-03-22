import numpy as np
import matplotlib.pyplot as plt # type: ignore

#-----------------------------------------------------------------------------------------------#
#---------- Parámetros -------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
f_senoidal = 300                                    # Frecuencia de la señal senoidal (300 Hz)
f_noise = 12345                                     # Frecuencia de ruido
f_muestreo = 44000                                  # Frecuencia de muestreo (44 kHz)
cantidad_ciclos = 10                                # Cantidad de ciclos a ver
duracion = ( 1 / f_senoidal ) * cantidad_ciclos     # Duración
M = 5                                               # orden del filtro
u = 0.01                                            # tasa de aprendisaje
#-----------------------------------------------------------------------------------------------#

cant_muestras = int (f_muestreo * (1/f_senoidal) * cantidad_ciclos)

y = np.zeros(cant_muestras)         # Salida del filtro
n = np.arange(0, cant_muestras, 1)  # Crea un vector desde 0 hasta n con pasos de 1

# Generación del vector de tiempo
t = np.arange(0, duracion, 1/f_muestreo)

# Señales senoidales
senoidal =  np.sin(2 * np.pi * f_senoidal * t)
noise = 0.1*np.sin(2 * np.pi * f_noise * t)

#iniciamos los pesos en 0
w = np.ones(M)  # Coeficientes iniciales en 0

#Sumatoria total
sen_total = senoidal + noise

#Calcular el error
err = sen_total - senoidal

x_n = np.zeros(M)  # Entrada en 0  

for i in range(cant_muestras):
    x_n = np.roll(x_n,1)
    x_n[0] = sen_total[i]

    y[i] = np.dot(w, x_n)                          
    error = senoidal[i] - y[i] 
    w = w + (u * error * x_n)
    '''
    if (i < 3):
        print("---------------------------------------------")
        print("El vector de entrada es: " + str(x_n))
        y[i] = np.dot(w, x_n)                           # Salida del filtro
        print("Salida: " + str(y[i]))
        print("Salida deseada: " + str(senoidal[i]))
        error = senoidal[i] - y[i] 
        print("El error es: " + str(error))
        w = w + (u * error * x_n)
        print("el vector de pesos en n+1: " + str(w))
        print("---------------------------------------------")
    '''
        


## ---- Descomentar para ver coeficientes finales del filtro ----
'''
print("---------------------------------------------")
print("El vector de coeficiente n: " + str(w))
print("---------------------------------------------")
'''

## ---- Descomentar si queremos ver las dimensiones de vectores de salida ----
'''
print(sen_total[:cant_muestras].shape)
print(y.shape)
print(n.shape)
'''

# Graficar la señal
plt.figure  (figsize=(10, 5)                )
plt.plot    (n, sen_total[:cant_muestras]   )  # Mostrar los 3 ciclos casi completos
plt.plot    (n, y                           )  # Mostrar los 3 ciclos casi completos
plt.title   ("Señal Senoidal de " + str(f_senoidal) + " Hz Muestreada a " + str(f_muestreo/1000) + " KHz")
plt.xlabel  ('Muestras'                     )
plt.ylabel  ('Amplitud'                     )
plt.grid    (True                           )
plt.show    (                               )