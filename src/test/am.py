# SoapySDRUtil --probe="driver=sdrplay"

import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants import numpy as np #use numpy for buffers
from scipy.signal import firwin 
import soundcard as sc
import numpy as np
import matplotlib.pyplot as plt
# configuracion del audio sink
default_speaker = sc.default_speaker()
#enumerate devices
#results = SoapySDR.Device.enumerate()
#for result in results: print(result)

#create device instance
#args can be user defined or from the enumeration result
args = dict(driver="sdrplay")
sdr = SoapySDR.Device(args)

#apply settings
sdr.setGainMode(SOAPY_SDR_RX, 0, True) #AGC
sdr.setSampleRate(SOAPY_SDR_RX, 0, 1e6)
sdr.setFrequency(SOAPY_SDR_RX, 0, 120824000)
sdr.setBandwidth(SOAPY_SDR_RX, 0, 200e3)

#setup a stream (complex floats)
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream) #start streaming

#create a re-usable buffer for rx samples
buff = np.array([0]*2048, np.complex64)

with default_speaker.player(samplerate=48000) as sp:
    print("Empezando recepción")
    taps_h = firwin(numtaps=101, cutoff=10e3, fs=1e6)
    taps_l = firwin(numtaps=21, cutoff=10e3, fs=48e3)

    f_shift = np.exp(-1j*2*np.pi*1000*np.arange(0,2048))
    while True:
        sr = sdr.readStream(rxStream, [buff], len(buff))
        # filtrado paso banda
        am = np.convolve(buff, taps_h, 'valid')

        # decimado fs~=48kHz
        am = am[::21]

        # shift del espectro a bansa base
        buff *= f_shift

        # Filtrado banda base
        #am = np.convolve(am, taps_l, 'valid')

        # decodificación AM
        am = np.abs(am)

        # normalizamos para el audio
        am /= np.max(np.abs(am))
#        plt.clf()
#        data = 10.0*np.log10( np.abs(np.fft.fft(am) **2 /(2048*48e3)))
#        plt.plot(np.arange(len(data)),data,'.-')
#        plt.draw()
#        plt.pause(0.001)

        sp.play(am)

#shutdown the stream
sdr.closeStream(rxStream)
sdr.deactivateStream(rxStream) #stop streaming
