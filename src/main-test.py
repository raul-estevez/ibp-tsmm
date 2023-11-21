import numpy as np
import matplotlib.pyplot as plt
import sys, time, math
from collections import deque
from dataclasses import dataclass
np.set_printoptions(threshold=sys.maxsize)

import demodulator as dm
import decoder as dc


buffer_len = dm.params.buffer_len
sps = dm.params.sps
ra_len = dm.params.ra_len

path_envelope = "../resources/convined.grc"
# Leer la envolvente (el input del módulo) 
envelope = np.fromfile(path_envelope, dtype=np.float32)
# También la separamos en una matriz de buffer_len columnas para simular la llegada paulatina de datos
n_pad = int(np.ceil(len(envelope)/buffer_len))
envelope = np.pad(envelope, (0, n_pad*buffer_len-len(envelope)))
envelope = np.reshape(envelope, (int(len(envelope)/buffer_len), buffer_len))

@dataclass
class trails_:
    mf = np.zeros(sps-1)
    diff = np.zeros(1,dtype=int)
    ra = np.zeros(ra_len-1)
    T = np.array([], dtype=int)
    PS = np.array([], dtype=int)
    samp = None
    ps_flag = None
    decision = 0
    dec_mf = 0
    th = 0.2
    buffer = np.zeros(int(sps))
    be_zeros = 0

    def reset(self):
        self.PS = np.array([], dtype=int)
        self.ps_flag = None
        self.dec_mf = 0
        self.decision = 0
        #self.th = 0.2 # El threshold del decisor lo podemos guardar
        self.buffer = np.zeros(int(sps))
        self.be_zeros = 0



@dataclass
class be_trails_:
    mf = np.zeros(sps-1)
    diff = np.zeros(1,dtype=int)
    ra = np.zeros(ra_len-1)
    T = np.array([], dtype=int)
    samp = None
    be_zeros = 0

    def reset(self):
        self.mf = np.zeros(sps-1)
        self.diff = np.zeros(1,dtype=int)
        self.ra = np.zeros(ra_len-1)
        self.T = np.array([], dtype=int)
        self.samp = None
        self.be_zeros = 0



trails = trails_()
be_trails = be_trails_()

# Initial conditions
decisions = list() 
be_state = False

bit_buffer = deque()

for i,data in enumerate(envelope):

    if not be_state:
        # guardamos el contexto para el demod por si cambiamos de estado
        trails.mf = be_trails.mf
        trails.diff = be_trails.diff
        trails.ra = be_trails.ra
        trails.T = be_trails.T
        trails.samp = be_trails.samp

        # Vemos si hay triggers
        (be_state, be_trails) = dm.be_detector(data, decisions, be_state, be_trails)

        if be_state:
            # TODO: se podría aprovechar todo hasta los T del be_detector, pero me da pereza implementarlo. Asi que como si fuera
            # de cero (pero con el contexto del anterior buffer guardado gracias a lo que hacemos más arriba)

            # Si había triggers demodulamos el buffer
            (decisions, trails) = dm.demodulator(data, trails)
            # Reseteamos la memoria del be_detector
            #print("Inicio de mensaje")
            #print("bits: " + str(decisions))
            be_trails.reset()

            bit_buffer.extend(decisions)
        ################### DEBUG ONLY, ELIMINAR EL ELSE
        #else:
            #print("STANDBY")
    else:
        # Demodulamos el buffer
        (decisions, trails) = dm.demodulator(data, trails)
        #print("bits: " + str(decisions))
        
        bit_buffer.extend(decisions)
        # Vemos si contiene algun final de mensaje i.e. 7 ceros seguidos
        (be_state, be_trails) = dm.be_detector(data, decisions, be_state, be_trails)

        # Si contiene los 7 ceros seguidos damos por concluido el mensaje
        if not be_state:
            # Guardamos el contexto del demod para el be_detector
            be_trails.mf = trails.mf
            be_trails.diff = trails.diff
            be_trails.ra = trails.ra
            be_trails.T = trails.T
            be_trails.samp = trails.samp

            be_trails.be_zeros = 0 # Reset de los zeros
            trails.reset() # Reset de solo las que no comparte con be_trails

            #print("Fin de mensaje")
            station, error = dc.convolve_hamming(list(bit_buffer))
            if station:print(str(list(bit_buffer))); print(station + " with " + str(error) + " bits of error")
            bit_buffer.clear()

            # Aquí no llamamos otra vez al be_detecor para ver si hay trigger por que queremos que la decisión de para el mensaje
            # sea fuerte i.e. que una vez que se detecten los ceros se ignore todo lo demas (los trigger que pueden haber el mismo
            # buffer) y se pare el mensaje. Esto ayuda a rechazar falsos triggers

    #print("----------------------------")

    # Guardamos los resultado para visualizarlos
    #decisions_result.append(decisions)
