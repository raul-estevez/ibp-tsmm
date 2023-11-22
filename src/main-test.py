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
path_envelope = "../resources/convined.grc"
# Leer la envolvente (el input del módulo) 
envelope = np.fromfile(path_envelope, dtype=np.float32)
# También la separamos en una matriz de buffer_len columnas para simular la llegada paulatina de datos
n_pad = int(np.ceil(len(envelope)/buffer_len))
envelope = np.pad(envelope, (0, n_pad*buffer_len-len(envelope)))
envelope = np.reshape(envelope, (int(len(envelope)/buffer_len), buffer_len))


############# DEMODULATOR SETUP ############# 
demod_pipe, demod_pipe_p = Pipe() # Both ends of the demod pipe
demodulator = Demodulator() # Create a demod object
demod_p = Process(target=demodulator.demodulator, args=(demod_pipe_p,)) # Create the demod process and pass it its pipe 
demod_p.start() # Start the demodulator process


############# DECODER SETUP ############# 
decod_pipe, decod_pipe_p = Pipe()
decoder = Decoder()
decod_p = Process(target=decoder.decoder, args=(decod_pipe_p,))
decod_p.start()

for _,data in enumerate(envelope):
    demod_pipe.send(data)
    if demod_pipe.poll():
        print("Bits recividos del demodulador")
        decod_pipe.send(list(demod_pipe.recv()))      # Le pasamos los bits al decodificador

        station = decod_pipe.recv()             # Bloqueamos hasta que responda
        if len(station):
            print("Estación recibida: " + str(station[0]) + " con " + str(station[1]) + " bits de error")
