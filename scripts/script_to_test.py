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
u           = 0.01                              # tasa de aprendisaje
#-----------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------#
#--------- Funciones ---------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#

#------------------------------------------------#
#----- Funcion grap -----------------------------#
#------------------------------------------------#

#Funcion para graficar la señal de entrada al filtro y la señal de salida del friltro

#input_signal   : señal de entrada con ruido a graficar
#output_signal  : señal de salida del filtro a graficar
#n              : cantidad de muestras a graficar
#fi             : frecuencia de la señal de entrada a querer filtrar
#fs             : frecuencia de la señal de muestreo

def grap(input_signal,output_signal,n,fi,fs):

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

def get_muetras(f_sample, f_input, cycles):
    cant_muestras = int (f_sample * (1/f_input) * cycles)
    return cant_muestras

#------------------------------------------------#
#----- Funcion gen_signals ----------------------#
#------------------------------------------------#

def gen_signals(time, f_sample, f_input, f_noise):
    t = np.arange(0, time, 1/f_sample)
    senoidal =  np.sin(2 * np.pi * f_input * t)
    noise = 0.1*np.sin(2 * np.pi * f_noise * t)
    return senoidal, noise

#------------------------------------------------#
#----- Funcion fa_fir ---------------------------#
#------------------------------------------------#

def fa_fir(M,cant_muestras,sen_total,senoidal):
    x_n = np.zeros(M)               # Entrada en 0  
    y = np.zeros(cant_muestras)     # Salida del filtro
    w = np.zeros(M)                 # Coeficientes iniciales en 0

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
    
    return y, w

#-----------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------#
#--------- Main --------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#

cant_muestras = get_muetras(f_sample, f_input, cycles)

senoidal, noise = gen_signals(time, f_sample, f_input, f_noise)

sen_total = senoidal + noise
        
y,w = fa_fir(M,cant_muestras,sen_total,senoidal)

grap(sen_total, y, cant_muestras, f_input, f_sample)