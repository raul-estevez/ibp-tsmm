import numpy as np
from collections import deque
from dataclasses import dataclass

#import matplotlib.pyplot as plt
#import sys
#np.set_printoptions(threshold=sys.maxsize)

from demodulator import Demodulator, params_

from multiprocessing import Process, Pipe
from time import sleep

params = params_()

buffer_len = params.buffer_len
sps = params.sps
ra_len = params.ra_len

path_envelope = "../resources/convined.grc"
# Leer la envolvente (el input del módulo) 
envelope = np.fromfile(path_envelope, dtype=np.float32)
# También la separamos en una matriz de buffer_len columnas para simular la llegada paulatina de datos
n_pad = int(np.ceil(len(envelope)/buffer_len))
envelope = np.pad(envelope, (0, n_pad*buffer_len-len(envelope)))
envelope = np.reshape(envelope, (int(len(envelope)/buffer_len), buffer_len))

# Demodulator pipes, object and process
demod_pipe, demod_pipe_p = Pipe()
demodulator = Demodulator()
demod_p = Process(target=demodulator.demodulator, args=(demod_pipe_p,))#, args=(demod_pipe_p))
# Start the demodulator process
demod_p.start()



for _,data in enumerate(envelope):
    demod_pipe.send(data)
    if demod_pipe.poll():
        print(demod_pipe.recv())







