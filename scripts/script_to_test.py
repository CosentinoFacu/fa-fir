import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------#
#---------- Parámetros -------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
f_input     = 30000.00                          # Frecuencia de la señal senoidal (300 Hz)
f_noise     = 10000000                          # Frecuencia de ruido
f_sample    = 50000000                          # Frecuencia de muestreo (44 kHz)
cycles      = 3                                 # Cantidad de ciclos a ver
time        = ( 1 / f_input ) * cycles          # Duración en segundos
M           = 30                                # orden del filtro
u           = 0.01                              # tasa de aprendizaje
#-----------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------#
#--------- Funciones ---------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#

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
    return senoidal, noise, t

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

def fa_fir(M, sig_i, sig_d, u, debug):
    x_n = np.zeros(M)               # Entrada en 0  
    y = np.zeros(len(sig_i))        # Salida del filtro
    w = np.zeros(M)                 # Coeficientes iniciales en 0

    for i in range(len(sig_i)):
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

#------------------------------------------------#
#----- Funcion get_spectrum ---------------------#
#------------------------------------------------#

def get_spectrum(sig_i,sig_o,fs):
    
    fft_result = np.fft.fft(sig_i)                                  # Calculamos DFT
    freq = (np.arange(len(fft_result)//2 + 1 )) * (fs/len(sig_i))   # Calculamos vector f  
    N = len(fft_result)                                             # Cantidad de puntos de la DFT
    fft_i_p = fft_result                                            # Obtenemos solo la parte positiva + DC                                   

    fft_result = np.fft.fft(sig_o)                                  # Calculamos DFT
    freq = (np.arange(len(fft_result)//2 + 1 )) * (fs/len(sig_o))   # Calculamos vector f  
    N = len(fft_result)                                             # Cantidad de puntos de la DFT
    fft_o_p = fft_result                                            # Obtenemos solo la parte positiva + DC                                   

    return fft_i_p, fft_o_p, freq


#------------------------------------------------#
#----- Funcion graph ----------------------------#
#------------------------------------------------#

def graph(sig_i,sig_o,time,freq,fi,fs):

    n_array = (np.arange(0, len(sig_i), 1)) * (1/fs) 
    title = ("Señal Senoidal de " + str(fi) + " Hz Muestreada a " + str(fs/1000) + " KHz")

    plt.figure(figsize=(8, 6))

    plt.subplot (2, 2, (1, 3)                               )
    plt.plot    (time, sig_i, label='Señal de entrada'   )
    plt.plot    (time, sig_o, label='Señal de Salida'    )
    plt.title   (title                                      )
    plt.xlabel  ('Tiempo'                                   )
    plt.ylabel  ('Amplitud'                                 )
    plt.grid    (True                                       )
    plt.legend  (                                           )

    N = len(spectrum_i)

    plt.subplot (2, 2, (2)                                      )  
    plt.plot    (freq, 2*((np.abs(spectrum_i[:N//2+1]))/N),'o-' ) 
    plt.title   ("Espectro de la Señal"                         )
    plt.xlabel  ("Frecuencia (Hz)"                              )
    plt.ylabel  ("Magnitud"                                     )
    plt.grid    (True                                           )
    plt.legend  (                                               )


    plt.subplot (2, 2, (4)                                      )
    plt.plot    (freq, 2*((np.abs(spectrum_o[:N//2+1]))/N),'o-' )
    plt.title   ("Espectro de la Señal"                         )
    plt.xlabel  ("Frecuencia (Hz)"                              )
    plt.ylabel  ("Magnitud"                                     )
    plt.grid    (True                                           )
    plt.legend  (                                               )

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------#
#--------- Main --------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#

senoidal, noise, t_array = gen_signals(time, f_sample, f_input, f_noise)

sen_total = senoidal + noise
        
y,w = fa_fir(M,sen_total,senoidal,u,0)

spectrum_i,spectrum_o, freq = get_spectrum(sen_total,y,f_sample)

graph(sen_total,y,t_array,freq,f_input,f_sample)