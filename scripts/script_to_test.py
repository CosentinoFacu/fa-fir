import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------#
#---------- Parámetros -------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
f_input     = 30000                             # Frecuencia de la señal senoidal (300 Hz)
f_noise     = 10000000                          # Frecuencia de ruido
f_sample    = 50000000                          # Frecuencia de muestreo (44 kHz)
cycles      = 5                                 # Cantidad de ciclos a ver
time        = ( 1 / f_input ) * cycles          # Duración en segundos
M           = 30                                # orden del filtro
u           = 0.01                              # tasa de aprendizaje
#-----------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------#
#--------- Funciones ---------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#

#------------------------------------------------#
#----- Funcion grap_time_response -----------------------------#
#------------------------------------------------#
'''
Funcion para graficar la señal de entrada al filtro y la señal de salida del filtro.

Parametros de entrada:
   input_signal    : señal de entrada con ruido a graficar
   output_signal   : señal de salida del filtro a graficar
   n               : cantidad de muestras a graficar
   fi              : frecuencia de la señal de entrada a querer filtrar
   fs              : frecuencia de la señal de muestreo
'''

def grap_time_response(input_signal,output_signal,n,fi,fs):

    n_array = np.arange(0, n, 1) 
    title = ("Señal Senoidal de " + str(fi) + " Hz Muestreada a " + str(fs/1000) + " KHz")

    # Graficar la señal
    plt.figure  (figsize=(10, 5)                )
    plt.plot    (n_array, input_signal [:n]     ) 
    plt.plot    (n_array, output_signal         )
    plt.title   (title                          )
    plt.xlabel  ('Muestras'                     )
    plt.ylabel  ('Amplitud'                     )
    plt.grid    (True                           )
    plt.show    (                               )

#------------------------------------------------#
#----- Funcion get_muestras ---------------------#
#------------------------------------------------#
'''
Funcion para obtener la cantidad de muestras a una frecuencia de muestro y 
unas determinada cantidad de ciclos a visualizar.

Parametros de entrada:
   fs          : frecuencia de la señal de muestreo
   fi          : frecuencia de la señal de entrada a querer filtrar
   cycles      : cantidad de ciclos a visualizar
'''

def get_muetras(fs, fi, cycles):
    cant_muestras = int (fs * (1/fi) * cycles)
    return cant_muestras

#------------------------------------------------#
#----- Funcion gen_signals ----------------------#
#------------------------------------------------#
'''
Funcion para obtener señal y ruido en el dominio del tiempo

Parametros de entrada:
   time    : ventanan temporal de la señal
   fs      : frecuencia de la señal de muestreo
   fi      : frecuencia de la señal de entrada
   fn      : frecuencia de la señal de ruido
'''

def gen_signals(time, fs, fi, fn):
    t = np.arange(0, time, 1/fs)
    senoidal =  np.sin(2 * np.pi * fi * t)
    noise = 0.1*np.sin(2 * np.pi * fn * t)
    return senoidal, noise

#------------------------------------------------#
#----- Funcion fa_fir ---------------------------#
#------------------------------------------------#
'''
Funcion para desarrollar el algoritmo LMS

Parametros:
   M       : Orden del filtro
   n       : Cantidad de muestras de la señal a discretizar
   sig_i   : Señal ruidosa de entrada
   sig_d   : Señal de referencia
   debug   : flag para habilitar prints de debug
'''

def fa_fir(M, n, sig_i, sig_d, u, debug):
    x_n = np.zeros(M)               # Entrada en 0  
    y = np.zeros(n)     # Salida del filtro
    w = np.zeros(M)                 # Coeficientes iniciales en 0

    for i in range(n):
        x_n = np.roll(x_n,1)
        x_n[0] = sig_i[i]
        y[i] = np.dot(w, x_n)                          
        error = sig_d[i] - y[i] 
        w = w + (u * error * x_n)
        
        if (i < 3 and debug == 1):
            print("---------------------------------------------")
            print("El vector de entrada es: " + str(x_n))
            y[i] = np.dot(w, x_n)                           # Salida del filtro
            print("Salida: " + str(y[i]))
            print("Salida deseada: " + str(sig_d[i]))
            error = sig_d[i] - y[i] 
            print("El error es: " + str(error))
            w = w + (u * error * x_n)
            print("el vector de pesos en n+1: " + str(w))
            print("---------------------------------------------")
        
    return y, w

#-----------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------#
#--------- Main --------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#

cant_muestras = get_muetras(f_sample, f_input, cycles)

senoidal, noise = gen_signals(time, f_sample, f_input, f_noise)

sen_total = senoidal + noise
        
y,w = fa_fir(M,cant_muestras,sen_total,senoidal,u,0)

grap_time_response(sen_total,y,cant_muestras,f_input,f_sample)