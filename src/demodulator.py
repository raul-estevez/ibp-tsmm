import numpy as np
import math
from collections import deque
from dataclasses import dataclass

# PARÁMETROS

@dataclass
class params_:
    fs = 20e3                   # Frecuencia de muestreo de los samples recibidos 
    sps = 1200                  # samples per symbol. TIENE QUE SER PAR
    buffer_len = 5*1200          # Tamaño del buffer de recepción 
    ra_len = 200               # Longitud del running averager aplicado a a la salida del matched filter
    zerox_th_h = 0.10            # Thresholds para la detección de los pasos por cero
    zerox_th_l = -0.10
    sps_th_h = 1700
    sps_th_l = 1100
    ps_rad = 0.9*1200

params = params_()

# Respuesta impulsional filtros
h_mf = 1/params.sps*np.ones(params.sps)
h_diff = [3e3,-3e3]
h_ra = 1/params.ra_len*np.ones(params.ra_len)

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

    triggers = np.nonzero((dist<h_delta) & (dist>l_delta))[0]   # Los índices de dist en los que se cumple la condición de trigger

    return (zerox[triggers], [len(x)-zerox[-1]])

def get_sampling_index(T, trail, ps_flag, ps_rad, sps, buffer_len):

    if len(T) == 0 and trail == None and ps_flag == None:
        # No hay nada que hacer
        return (deque(), None, None)
    elif len(T) == 0:
        T.appendleft(trail) # No hay nada que pueda invalidar el trail, si es que es un PS
    elif ps_flag == False:
        # La trail era un T, por lo que lo añadimos a los que hay nuevos
        T.appendleft(trail)
        #print("AAA")
        #print(trail)
    elif trail != None:
        # La trail es un PS, lo intentamos invalidar con el primer T. Si no se invalida lo convertimos en un T más
        if T[0]-trail > ps_rad:
            T.appendleft(trail)
#    else:
#        print("Es el primero")


    # Calculamos cuantos PS caben entre un T y el siguiente y los metemos cada sps
    if len(T) > 1:
        #print(T)
        for i in range(len(T) - 1): 
            n_ps = math.floor((T[i+1] - T[i]) / sps) - 1
            for j in range(i+1,i+n_ps+1): T.insert(j, T[i]+sps*(j-i))   # Los mete directamnte en el índice correcto, es más
                                                                        # rápido que meterlos al final y ordenar
    # El último (o el primero si solo hay uno) se hace con respecto al final del array, ya que no hay otro después de el, ya que
    # es el último
    n_ps = math.floor((buffer_len - T[-1]) / sps) #TODO: Hacer round() en vez de floor? No se si rompe algo
#    print(n_ps)
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
#        #print(threshold)
        if soft >= threshold: decisions[i+1] = 1
        if decisions[i+1] != decisions[i]: treshold =  (soft + soft_decisions[i]) * 0.5

    return (decisions, soft_decisions[-1], decisions[-1], threshold)
        
def be_detector(data: np.ndarray, bits: list, be_state: bool,  trails): 

    # TODO: implementar un timeout cuando state=true
    # TODO: probar con un audio con muchas estaciones

    if not be_state:
        # Estamos esperando por un trigger, lo buscamos en el buffer actual y decimos si lo encontramos o no

         # Pasamos por el matched filter
        (mf, trails.mf) = convolve_rt(data, h_mf, trails.mf)

        # Pasamos por el running averager
        (mf, trails.ra) = convolve_rt(mf, h_ra, trails.ra)

        # Calulamos la primera diferencia (hacia atrás)
        (diff, trails.diff) = convolve_rt(mf, h_diff, trails.diff)

        # Buscamos los triggers en diff
        (T, trails.T) = find_triggers(diff, trails.T, params.zerox_th_h, params.zerox_th_l, params.sps_th_h, params.sps_th_l)
        # Si hay triggers
#        print("triggers be: " + str(T))
        return (True, trails) if len(T) else (False, trails)
    else:
        # Estamos esperando por 3 comas i.e. 6 ceros seguidos, que más el cero de símbolo anterior dan 7 ceros consecutivos

        # El trail be_zeros es el numero de ceros consecutivos que hay al final del buffer anterior
        groups = deque()
        groups.append(trails.be_zeros) 
        
        prev_z = 0 if groups[0] else 1 
        for z in bits:
            if prev_z and not z: groups.append(1); prev_z = 0   # 1 -> 0 empezamos a contar un nuevo grupo de ceros
            elif not prev_z and z: prev_z = 1                   # 0 -> 1 se acaba el grupo de ceros actual
            elif not z: groups[-1] += 1                               # Un nuevo cero en el grupo, lo contamos

#        print("groups: " + str(groups))
        trails.be_zeros = groups[-1] if not bits[-1] else 0
        return (not any(g>=7 for g in groups), trails)


def demodulator(data: np.ndarray, trails) -> (tuple[int], dict):
     # Pasamos por el matched filter
    (mf, trails.mf) = convolve_rt(data, h_mf, trails.mf)

    # Pasamos por el running averager
    (mf, trails.ra) = convolve_rt(mf, h_ra, trails.ra)

    # Calulamos la primera diferencia (hacia atrás)
    (diff, trails.diff) = convolve_rt(mf, h_diff, trails.diff)

    # Buscamos los triggers en diff
    (T, trails.T) = find_triggers(diff, trails.T, params.zerox_th_h, params.zerox_th_l, params.sps_th_h, params.sps_th_l)
#    print("triggers demod: " + str(T))
    T = deque(T)

    # Propagamos los T y el prev_ps
    (sampling_index, trails.samp, trails.ps_flag) = get_sampling_index(T, trails.samp, trails.ps_flag, params.ps_rad, params.sps, params.buffer_len)
#    print("samps: " + str(T))

    # por que cojones no se puede indexar por una slice una deque 
#    soft_decisions = deque(mf[[sampling_index[i] for i in range(1,len(sampling_index))]])
#    if len(sampling_index) !=  0: 
#        if sampling_index[0] < 0: 
#            soft_decisions.appendleft(trails.buffer[sampling_index[0]])

    # No es lo más bonito de ver en código pero creo que es lo más rápido
    decisions = []
    if len(sampling_index):
        if sampling_index[0] < 0:
            soft_decisions = deque(mf[[sampling_index[i] for i in range(1,len(sampling_index))]])
            soft_decisions.appendleft(trails.buffer[sampling_index[0]]) # Miramos en el buffer anterior la soft decision
        else:
            soft_decisions = deque(mf[[sampling_index[i] for i in range(len(sampling_index))]]) # llenamos con todo lo que tenemos

#        print("Decisions: " + str(soft_decisions))

        if len(soft_decisions) != 0: 
            (decisions, trails.dec_mf, trails.decision, trails.th) = decisor(soft_decisions, trails.dec_mf, trails.decision, trails.th)
    

    trails.buffer = mf[-params.sps:]

    return (decisions, trails)
