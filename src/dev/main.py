import numpy as np
from collections import deque
from dataclasses import dataclass
from multiprocessing import Process, Pipe

from demodulator import Demodulator, params_
from decoder import Decoder
from frontend import Frontend

params = params_()

############# PIPES ############# 
bb_pipe_1, bb_pipe_0 = Pipe(False) # Duplex=false i.e. conn1->recv, conn2->send ONLY
demod_pipe_1, demod_pipe_0 = Pipe(False)
decod_pipe_1, decod_pipe_0 = Pipe(False)

############# FRONT-END SETUP ############# 
frontend = Frontend()
front_p = Process(target=frontend.frontend, args=(bb_pipe_0,))
front_p.start()

############# DEMODULATOR SETUP ############# 
demodulator = Demodulator()
demod_p = Process(target=demodulator.demodulator, args=(bb_pipe_1,demod_pipe_0))
demod_p.start()


############# DECODER SETUP ############# 
decoder = Decoder()
decod_p = Process(target=decoder.decoder, args=(demod_pipe_1, decod_pipe_0))
decod_p.start()

# decod_pipe_1 is the pipe end that the main reads

while True:
    if decod_pipe_1.poll():
        station = decod_pipe_1.recv()             # Bloqueamos hasta que responda
        if len(station):
            print("Estación recibida: " + str(station[0]) + " con " + str(station[1]) + " bits de error", flush=True)
        else:
            print("na de na", flush=True)


demod_p.kill()
decod_p.kill()
front_p.kill()
