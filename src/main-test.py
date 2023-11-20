import numpy as np
import matplotlib.pyplot as plt
import sys, time, math
from collections import deque
from dataclasses import dataclass
np.set_printoptions(threshold=sys.maxsize)

import demodulator as dm


buffer_len = dm.params.buffer_len
sps = dm.params.sps
ra_len = dm.params.ra_len

path_envelope = "../resources/bad.grc"
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
    dec_mf = 0
    decision = 0
    th = 0.2
    buffer = np.zeros(int(sps))


decisions_result = deque()

trails = trails_()

for i,data in enumerate(envelope):
    (decisions, trails) = dm.demodulator(data, trails)

    # Guardamos los resultado para visualizarlos
    #decisions_result.append(decisions)
    print(decisions)
