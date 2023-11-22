# SoapySDRUtil --probe="driver=sdrplay"

import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np #use numpy for buffers
import matplotlib.pyplot as plt
from scipy.signal import firwin
import soundcard as sc

# configuracion del audio sink
default_speaker = sc.default_speaker()

#create device instance
#args can be user defined or from the enumeration result
args = dict(driver="sdrplay")
sdr = SoapySDR.Device(args)

#apply settings
sdr.setGainMode(SOAPY_SDR_RX, 0, True) #AGC
sdr.setSampleRate(SOAPY_SDR_RX, 0, 250e3)
sdr.setFrequency(SOAPY_SDR_RX, 0, 106500000)
#sdr.setFrequency(SOAPY_SDR_RX, 0, 87500000)
sdr.setBandwidth(SOAPY_SDR_RX, 0, 300e3)
#print(sdr.getBandwidth(SOAPY_SDR_RX,0))

#print(sdr.getFrequencyCorrection(SOAPY_SDR_RX,0))

#setup a stream (complex floats)
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream) #start streaming

#print(sdr.getStreamMTU(rxStream))
#create a re-usable buffer for rx samples
buff = np.array([0]*2048, np.complex64)

with default_speaker.player(samplerate=48000) as sp:
    print("Empezando recepción")
    taps_h = firwin(numtaps=101, cutoff=15e3, fs=250e3)

    while True:
        sr = sdr.readStream(rxStream, [buff], len(buff))

        fm = np.angle(buff[0:-1] * np.conj(buff[1:]))
        fm = np.convolve(fm, taps_h, 'valid')


        # Este no funciona por como está definido np.angle: devuelve valores en -pi:pi. Entonces si dos ángulos estan uno un poco por
        # encima de pi y el otro un poco de debajo de pi, donde en realidad la diferencia de ángulos es pequeña, al definir np.angle
        # así nos da una diferencia de casi 2pi. Entonces es mejor hacer solo un np.angle, como en el ejemplo que funciona

        #fm = np.angle(buff[:-1]) - np.angle(buff[1:])
        #fm = np.diff(np.angle(buff))

        fm = np.convolve(fm, taps_h, 'valid')

        # decimado fs~=48kHz
        fm = fm[::5]


        # normalizamos para el audio
        fm /= np.max(np.abs(fm))

        sp.play(fm)
        
    #buff.tofile('VGO' + str(i) +'.iq') # Save to file 
    #print(i)

    #print(buff)
    #print("-----------------------------------")

#    plt.clf()
#    data = 10.0*numpy.log10( numpy.abs(numpy.fft.fft(buff) **2 /(1024*5e6)))
#    plt.plot(numpy.arange(512),data[0:512],'.-')
#    plt.draw()
#    plt.pause(0.001)
#    

#    print(sr.ret) #num samples or error code
#    print(sr.flags) #flags set by receive operation
#    print(sr.timeNs) #timestamp for receive buffer

#shutdown the stream
sdr.closeStream(rxStream)
sdr.deactivateStream(rxStream) #stop streaming
