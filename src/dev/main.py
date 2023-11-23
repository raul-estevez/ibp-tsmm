import numpy as np
from collections import deque
from dataclasses import dataclass
from multiprocessing import Process, Pipe

from demodulator import Demodulator, params_
from decoder import Decoder

params = params_()

# Necesario para simular la llegada de datos en tiempo real
buffer_len = params.buffer_len
sps = params.sps
path_envelope = "../../resources/convined.grc"
# Leer la envolvente (el input del módulo) 
envelope = np.fromfile(path_envelope, dtype=np.float32)
# También la separamos en una matriz de buffer_len columnas para simular la llegada paulatina de datos
n_pad = int(np.ceil(len(envelope)/buffer_len))
envelope = np.pad(envelope, (0, n_pad*buffer_len-len(envelope)))
envelope = np.reshape(envelope, (int(len(envelope)/buffer_len), buffer_len))


############# PIPES ############# 
bb_pipe_0, bb_pipe_1 = Pipe()
demod_pipe_0, demod_pipe_1 = Pipe()
decod_pipe_0, decod_pipe_1 = Pipe()

############# BASEBAND SETUP ############# 



############# DEMODULATOR SETUP ############# 
demodulator = Demodulator()
demod_p = Process(target=demodulator.demodulator, args=(bb_pipe_1,demod_pipe_0))
demod_p.start()


############# DECODER SETUP ############# 
decoder = Decoder()
decod_p = Process(target=decoder.decoder, args=(demod_pipe_1, decod_pipe_0))
decod_p.start()

# decod_pipe_1 is the pipe end that the main reads

for _,data in enumerate(envelope):
    bb_pipe_0.send(data)
    if decod_pipe_1.poll():
        station = decod_pipe_1.recv()             # Bloqueamos hasta que responda
        if len(station):
            print("Estación recibida: " + str(station[0]) + " con " + str(station[1]) + " bits de error")

