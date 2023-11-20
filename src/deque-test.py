import numpy as np
import matplotlib.pyplot as plt
import sys, time, math
from collections import deque
np.set_printoptions(threshold=sys.maxsize)


from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput


def convolve_rt(x1, x2, trail):
    # x1 & x2: datos a convolucionar, suponemos que len(x1)>len(x2)
    # trail: últimos len(x2)-1 datos de la convolución anterior, que se suman a los len(x2)-1 primeros de la nueva para simular 
    # una convolución continua
    conv = np.convolve(x1,x2, 'full')   # Hacemos la convolución entera, suponiendo que hay len(trail) ceros al principio y final
    conv[:len(trail)] += trail          # Sumamos la cola de la convolución anterior 
    return (conv[:len(x1)], conv[-len(trail):])     # Devolvemos los datos validos de la convolución y la nueva cola

def find_triggers(x, trail, h_th, l_th, h_delta, l_delta):
    # Buscamos los índices de los pasos por cero
#    zerox = [i for i,z in enumerate(x) if z<h_th and z>l_th]
    zerox = np.nonzero((x < h_th) & (x > l_th))[0]

    # Calculamos las distancias entre los pasos por cero
    dist,_ = convolve_rt(zerox, [1, -1], trail) # Aquí trail es la distancia del último zero crossing al final del buffer
#    Hacerlo con una list comprehension es bastante (~15ms) más lento que con el np array

#    tic = time.perf_counter()
#    toc = time.perf_counter()
#    print("Elapsed time:", toc-tic)

#    triggers = [i for i,d in enumerate(dist) if d<h_delta and d>l_delta]
    triggers = np.nonzero((dist<h_delta) & (dist>l_delta))[0]   # Los índices de dist en los que se cumple la condición de trigger

    return (zerox[triggers], [len(x)-zerox[-1]])
#    return ([zerox[i] for i in triggers], [len(x) - zerox[-1]])
# Devolvemos las posiciones de los triggers y la posición del último zero 
                                                    # crossing al final del array
def get_sampling_index(T, trail, ps_flag, ps_rad, sps, buffer_len):

    if len(T) == 0 and trail == None and ps_flag == None:
        # No hay nada que hacer
        return (deque(), None, None)
    elif len(T) == 0:
        T.appendleft(trail) # No hay nada que pueda invalidar el trail, si es que es un PS
    elif ps_flag == False:
        # La trail era un T, por lo que lo añadimos a los que hay nuevos
        T.appendleft(trail)
    elif trail != None:
        # La trail es un PS, lo intentamos invalidar con el primer T. Si no se invalida lo convertimos en un T más
        if T[0]-trail > ps_rad:
            T.appendleft(trail)


    # Calculamos cuantos PS caben entre un T y el siguiente y los metemos cada sps
    if len(T) > 1:
        for i in range(len(T) - 1): 
            n_ps = math.floor((T[i+1] - T[i]) / sps) - 1
            for j in range(i+1,i+n_ps+1): T.insert(j, T[i]+sps*(j-i))   # Los mete directamnte en el índice correcto, es más
                                                                        # rápido que meterlos al final y ordenar
    # El último (o el primero si solo hay uno) se hace con respecto al final del array, ya que no hay otro después de el, ya que
    # es el último
    n_ps = math.floor((buffer_len - T[-1]) / sps)
    T.extend([T[-1] + sps*n for n in range(1, n_ps+1)])

    new_trail = T.pop() - buffer_len
    new_ps_flag = True if n_ps > 0 else False
    return (T, new_trail, new_ps_flag)    

def decisor(soft_decisions, trail_mf, trail_decision, trail_th):
    decisions = [0 for i in range(len(soft_decisions))]
    threshold = trail_th

    # Initial condition
    if soft_decisions[0] > trail_th: decisions[0] = 1
    if trail_decision != decisions[0]: threshold = (trail_mf + soft_decisions[0]) * 0.5

    for i,soft in enumerate([soft_decisions[j] for j in range(1,len(soft_decisions))]):
        #print(threshold)
        if soft >= threshold: decisions[i+1] = 1
        if decisions[i+1] != decisions[i]: treshold =  (soft + soft_decisions[i]) * 0.5

    return (decisions, soft_decisions[-1], decisions[-1], threshold)
        

        

def test():
    # FIXME: Meter los parámetros en un dict
    # PARÁMETROS
    path_envelope = "../resources/bad.grc"
    fs = 20e3                   # Frecuencia de muestreo de los samples recibidos 
    sps = 1200                  # samples per symbol. TIENE QUE SER PAR
    buffer_len = 5*sps          # Tamaño del buffer de recepción 
    ra_len = 200                # Longitud del running averager aplicado a a la salida del matched filter
    zerox_th_h = 0.10            # Thresholds para la detección de los pasos por cero
    zerox_th_l = -zerox_th_h
    sps_th_h = 1700
    sps_th_l = 1100
    ps_rad = 0.9*sps



    # Respuesta impulsional filtros
    h_mf = 1/sps*np.ones(sps)
    h_diff = [3e3,-3e3]
    h_ra = 1/ra_len*np.ones(ra_len)

    # Leer la envolvente (el input del módulo) 
    envelope = np.fromfile(path_envelope, dtype=np.float32)
    #envelope = envelope[56000:70000]
    # También la separamos en una matriz de buffer_len columnas para simular la llegada paulatina de datos
    n_pad = int(np.ceil(len(envelope)/buffer_len))
    envelope = np.pad(envelope, (0, n_pad*buffer_len-len(envelope)))
    envelope = np.reshape(envelope, (int(len(envelope)/buffer_len), buffer_len))


    #np.save("port/data.npy", envelope)

    # Matrices para guardar las convoluciones y poder visualizarlas
    matched_filter_result = np.zeros(np.shape(envelope))
    diff_result = np.zeros(np.shape(envelope))
    trigger_result = np.array([], dtype=int)
    decisions_result = np.array([], dtype=int)

    # Buffers para las colas de las convoluciones
    trail_mf = np.zeros(sps-1)
    trail_diff = np.zeros(1,dtype=int)
    trail_ra = np.zeros(ra_len-1)
    trail_T = np.array([], dtype=int)
    trail_PS = np.array([], dtype=int)

    trail_samp = None
    ps_flag = None

    trail_dec_mf = 0
    trail_decision = 0
    trail_th = 0.2

    # Vector para guardar las sps/2 muestras más antiguas del buffer anterior, necesario para buscar un threshold en el decisor
    # para lo casos en el que la transición se encuentre a menos de sps/2 muestras del final del buffer actual
    prev_mf = np.zeros(int(sps))

    decisions = []
    

    

#    tic = time.perf_counter()
    for i,data in enumerate(envelope):
        # Pasamos por el matched filter
        (mf, trail_mf) = convolve_rt(data, h_mf, trail_mf)
        
   
        # Pasamos por el running averager
        (mf, trail_ra) = convolve_rt(mf, h_ra, trail_ra)

        # Calulamos la primera diferencia (hacia atrás)
        (diff, trail_diff) = convolve_rt(mf, h_diff, trail_diff)


        # Buscamos los triggers en diff
        (T, trail_T) = find_triggers(diff,trail_T, zerox_th_h, zerox_th_l, sps_th_h, sps_th_l)
        T = deque(T)

        # Propagamos los T y el prev_ps
        (sampling_index, trail_samp, ps_flag) = get_sampling_index(T, trail_samp, ps_flag, ps_rad, sps, buffer_len)

        soft_decisions = deque(mf[[sampling_index[i] for i in range(1,len(sampling_index))]])
        if len(sampling_index) !=  0: 
            if sampling_index[0] < 0: 
                soft_decisions.appendleft(prev_mf[sampling_index[0]])
        
        if len(soft_decisions) != 0: 
            (decisions, trail_dec_mf, trail_decision, trail_th) = decisor(soft_decisions, trail_dec_mf, trail_decision, trail_th)

        prev_mf = mf[-sps:]
            # Guardamos el resultado para visualizarlos
        matched_filter_result[i,:] = mf
        diff_result[i,:] = diff
        trigger_result = np.append(trigger_result, i*buffer_len+np.array(sampling_index))
        decisions_result = np.append(decisions_result, decisions)
        print(decisions)

    trigger_result = trigger_result.astype(int)

    matched_filter_result = np.reshape(matched_filter_result, np.size(matched_filter_result))
    diff_result = np.reshape(diff_result, np.size(diff_result))
    triger_result = trigger_result[trigger_result != 0]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True)

    x = np.r_[0:len(matched_filter_result)]

    ax1.plot(x, matched_filter_result, color='blue')
    ax1.stem(x[trigger_result],matched_filter_result[trigger_result])
    ax1.set_title("Output del matched filter y soft decisions")

    ax2.plot(x, diff_result, color='red')
    ax2.stem(x[trigger_result],diff_result[trigger_result]) 
    ax2.set_title("Primera diferencia")

    #print(len(x[trigger_result]))
    #print(len(decisions_result))
        
    #decisions_result = np.append(decisions_result, 14*[0])
    decisions_result = np.append([1], decisions_result)
    print(len(x[trigger_result]))
    print(len(decisions_result))
    ax3.stem(x[trigger_result], decisions_result)
    ax3.set_title("Output bits");
    ax4.plot(x,np.ravel(envelope))
    ax4.set_title("Envolvente de la señal")

    plt.show()

if __name__ == '__main__':

#    with PyCallGraph(output=GraphvizOutput()):
    test()

