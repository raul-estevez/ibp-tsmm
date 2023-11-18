import numpy as np
import matplotlib.pyplot as plt
import sys, time, math
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
    zerox = np.nonzero((x < h_th) & (x > l_th))[0]

    # Calculamos las distancias entre los pasos por cero
    dist,_ = convolve_rt(zerox, [1, -1], trail) # Aquí trail es la distancia del último zero crossing al final del buffer
                                                # por lo que podemos ignorar el trail que devuelve la función

    triggers = np.nonzero((dist<h_delta) & (dist>l_delta))[0]   # Los índices de dist en los que se cumple la condición de trigger
    return (zerox[triggers], [len(x) - zerox[-1]])  # Devolvemos las posiciones de los triggers y la posición del último zero 
                                                    # crossing al final del array
def get_sampling_index(T, prev_ps, prev_t, ps_rad, sps, buffer_len):
    ## Inputs:
    #   T -> vector con los triggers del buffer actual
    #   prev_ps -> distancia desde el primer elemento del buffer actual al último PS del buffer anterior (en negativo)
    #   prev_t -> distancia desde el primer element del buffer actual al último T del buffer anterior (en negativo)
    #       NOTA: solo uno de los dos toma valores a la vez. Ya que lo que queremos es saber si el último índice donde se calculó 
    #       que se debe samplear es T o un PS para intentar invalidarlo/propagarlo en el buffer actual
    #   ps_rad -> distancia máxima desde un T en la que invalida a un PS
    #   sps -> samples per symbol
    #       TODO: Esta es la distancia que se usa para propagar los PS, puede ser un valor fijo (como lo es ahora) pero sería
    #       mejor si fuese la media de las distancia entre lo últimos n triggers.
    #   buffer_len -> el tamaño en samples del buffer

    ## Outputs:
    #   1 -> Un array con los índices donde se debe samplear la salida del matched filter. Hay indices positivos y negativos. Los
    #   positivos se corresponden con los índices en el buffer actual. Y los negativos con las muestras del buffer anterior
    #   (empezando desde el final del buffer anterior/principio del actual, por eso negativas)
    #   prev_ps/prev_t -> Estos arrays nos dicen que el último índice donde se debe muestrear de este array es un PS o un T y que 
    #   índice es. Esto se pasa al siguiente buffer para intentar invalidar el PS (si se manda prev_ps) o para propagar el T (si
    #   se manda prev_t)
    #   FIXME: intentar aclarar esto, seguro que se puede hacer de una forma más eficiente, pero por ahora funciona 

    # FIXME: intenta eliminar tantos np.append como puedas, seguro que ganas tiempo por no tener que allocate copias de arrays
    if prev_t.size != 0: T = np.append(prev_t, T) # Añadimos a T el anterior T si lo hay (con índice negativo)
    #T = np.append(prev_t, T) if prev_t.size != 0 else T
    # Añadimos a new_ps el anterior PS si lo hay (con índice negativo)
    new_ps = prev_ps if prev_ps.size != 0 else np.array([],dtype=int)
    if T.size == 0:
        if prev_ps.size == 0:
            # No hay nada que hacer
            return (np.array([],dtype=int), np.array([],dtype=int), np.array([],dtype=int))
        else:
            # Solo hay prev_ps, no va a ser invalidado por nada
            T=prev_ps
    else:
        if prev_ps.size != 0:
            # Intentamos invalidar el prev_ps
            if np.abs(T[0] - prev_ps[0]) > ps_rad:
                T = np.append(prev_ps, T)
    # LLegados aquí tenemos en T lo que se va a propagar    
    # Calculamos cuantos PS caben entre un T[i] y el siguiente y los metemos cada sps
    if T.size != 1:
        for i in range(T.size - 1):
            num_ps = np.abs(np.floor((T[i+1] - T[i])/sps)) - 1
            new_ps = np.append(new_ps, T[i]+ sps*np.r_[1:(num_ps+1)])
    # Hay que hacer el último ( o el primero si solo tiene un elemento)
    num_ps = int(np.abs(np.floor((T[-1] - buffer_len)/sps)) - 1)
    new_ps = np.append(new_ps, T[-1] + sps*np.r_[1:(num_ps+1)])

    # Limpiamos los PS que estén cerca de un trigger
    new_ps = invalidate_PS(T, new_ps,ps_rad)
    new_t = np.sort(np.append(T,new_ps)).astype(int)
    
    # FIXME: Encapsular esto en una sola estructura de datos e.g. un dict o una tuple
    if new_ps.size == 0:
        return (new_t[:-1], np.array([],dtype=int), np.ravel(new_t[-1]-buffer_len).astype(int))
    elif new_ps[-1] == new_t[-1]:
        # El último es un PS
        return (new_t[:-1], np.ravel(new_ps[-1]-buffer_len).astype(int), np.array([],dtype=int))
    else: 
        # El último es un T
        return (new_t[:-1], np.array([],dtype=int), np.ravel(new_t[-1]-buffer_len).astype(int))


# Devuelve los PS que cumplen con ps_rad
def invalidate_PS(T,PS,ps_rad):
    PS_leftovers = [ps for ps in PS for t in T if abs(ps-t)<ps_rad] # los que sobran
    return np.setdiff1d(PS, PS_leftovers)                             # Me quedo con los que cumplen

def decisor(soft_decisions, trail_mf, trail_decision, trail_th):
    decisions = np.zeros(soft_decisions.size)
    threshold = trail_th

    # Initial condition
    if soft_decisions[0] > trail_th: decisions[0] = 1
    if trail_decision != decisions[0]: threshold = (trail_mf + soft_decisions[0]) * 0.5

    for i,soft in enumerate(soft_decisions[1:]):
        if soft >= threshold: decisions[i+1] = 1
        if decisions[i+1] != decisions[i]: treshold =  (soft + soft_decisions[i]) * 0.5

    return (decisions.astype(int), soft_decisions[-1], decisions[-1], threshold)
        

        


def test():
    # FIXME: Meter los parámetros en un dict
    # PARÁMETROS
    path_envelope = "../resources/envelope.grc"
    fs = 20e3                   # Frecuencia de muestreo de los samples recibidos 
    sps = 1200                  # samples per symbol. TIENE QUE SER PAR
    buffer_len = 5*sps          # Tamaño del buffer de recepción 
    ra_len = 200                # Longitud del running averager aplicado a a la salida del matched filter
    zerox_th_h = 0.08            # Thresholds para la detección de los pasos por cero
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
    prev_ps = np.array([],dtype=int)
    prev_t = np.array([],dtype=int)
    trail_dec_mf = 0
    trail_decision = 0
    trail_th = 0.1

    # Vector para guardar las sps/2 muestras más antiguas del buffer anterior, necesario para buscar un threshold en el decisor
    # para lo casos en el que la transición se encuentre a menos de sps/2 muestras del final del buffer actual
    stale_buff = np.zeros(int(sps))

    #tic = time.perf_counter()
    for i,data in enumerate(envelope):
        # Pasamos por el matched filter
        (mf, trail_mf) = convolve_rt(data, h_mf, trail_mf)
    
        # Pasamos por el running averager
        (mf, trail_ra) = convolve_rt(mf, h_ra, trail_ra)

        # Calulamos la primera diferencia (hacia atrás)
        (diff, trail_diff) = convolve_rt(mf, h_diff, trail_diff)


        # Buscamos los triggers en diff
        (T, trail_T) = find_triggers(diff,trail_T, zerox_th_h, zerox_th_l, sps_th_h, sps_th_l)

        # Propagamos los T y el prev_ps
        (sampling_index, prev_ps, prev_t) = get_sampling_index(T, prev_ps, prev_t, ps_rad, sps, buffer_len)


        decisions = np.array([], dtype=int)
        if sampling_index.size != 0: # TODO: mover esto a dentro de la función igual es más elegante?
            (decisions, trail_dec_mf, trail_decision, trail_th) = decisor(np.append(stale_buff[sampling_index[sampling_index < 0]],
                                                                         mf[sampling_index[sampling_index > 0]]),
                                                                          trail_dec_mf, trail_decision, trail_th)




        stale_buff = mf[-sps:]
        # Guardamos el resultado para visualizarlos
#        matched_filter_result[i,:] = mf
#        diff_result[i,:] = diff
#        trigger_result = np.append(trigger_result, i*buffer_len+sampling_index)
#        decisions_result = np.append(decisions_result, decisions)
#

#    toc = time.perf_counter()
#    print("Elapsed time:", toc-tic)

#    matched_filter_result = np.reshape(matched_filter_result, np.size(matched_filter_result))
#    diff_result = np.reshape(diff_result, np.size(diff_result))
#    triger_result = trigger_result[trigger_result != 0]
#
#    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True)
#
#    x = np.r_[0:len(matched_filter_result)]
#
#    ax1.plot(x, matched_filter_result, color='blue')
#    ax1.stem(x[trigger_result],matched_filter_result[trigger_result])
#    ax1.set_title("Output del matched filter y soft decisions")
#
#    ax2.plot(x, diff_result, color='red')
#    ax2.stem(x[trigger_result],diff_result[trigger_result]) 
#    ax2.set_title("Primera diferencia")
#
#    ax3.stem(x[trigger_result], decisions_result)
#    ax3.set_title("Output bits");
#    ax4.plot(x,np.ravel(envelope))
#    ax4.set_title("Envolvente de la señal")
#
#    plt.show()
if __name__ == '__main__':
    with PyCallGraph(output=GraphvizOutput()):
        test()
