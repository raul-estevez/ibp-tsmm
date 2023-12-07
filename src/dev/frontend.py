import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants import numpy as np #use numpy for buffers
from scipy.signal import firwin 
import numpy as np
from demodulator import Demodulator, params_
from dataclasses import dataclass


args = dict(driver="sdrplay", )
sdr = SoapySDR.Device(args)

# Apply settings
sdr.setAntenna(SOAPY_SDR_RX, 0, "Antenna C")
sdr.setDCOffsetMode(SOAPY_SDR_RX, 0, True)  
sdr.setGainMode(SOAPY_SDR_RX, 0, True) #AGC
sdr.setSampleRate(SOAPY_SDR_RX, 0, 62.5e3) # Creo que le podemos poner lo que necesitemos y la API de Soapy hace lo necesario [1]
# [1] https://github.com/pothosware/SoapySDRPlay3/blob/master/Settings.cpp      línea: 730
print(sdr.getSampleRate(SOAPY_SDR_RX, 0))

sdr.writeSetting("iqcorr_ctrl", True)
sdr.writeSetting("biasT_ctrl", False)
sdr.writeSetting("rfnotch_ctrl", True)
sdr.writeSetting("dabnotch_ctrl", True)

sdr.setFrequency(SOAPY_SDR_RX, 0, 14.095e6) # 14.1 MHz esta 5kHz por encima
sdr.setBandwidth(SOAPY_SDR_RX, 0, 200e3)

# Setup a stream (complex floats)
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream) #start streaming

# Create a re-usable buffer for rx samples
buff = np.array([0]*2048, np.complex64)

# Read the filter coeficients
h_bp_5k_I = np.loadtxt("bp_5k_real.fcf")
h_bp_5k_Q = np.loadtxt("bp_5k_imag.fcf")

demod = Demodulator()

@dataclass
class trails_:
    yi1 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
    yi2 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
    yq1 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
    yq2 = np.zeros(int((h_bp_5k_I.size - 1) / 2))

trails = trails_()
while True:
    data = sdr.readStream(rxStream, [buff], len(buff))

    data_I = np.real(buff)
    data_Q = np.imag(buff)
    # Band pass filter centered in 5KHz
    # In-phase part
    [yi1, trails.yi1] = demod.convolve_rt(data_I, h_bp_5k_I, trails.yi1)
    [yi2, trails.yi2] = demod.convolve_rt(data_Q, h_bp_5k_Q, trails.yi2)
    # Quadrature part
    [yq1, trails.yq1] = demod.convolve_rt(data_I, h_bp_5k_Q, trails.yq1)
    [yq2, trails.yq2] = demod.convolve_rt(data_Q, h_bp_5k_I, trails.yq2)

    # Decimamos a 20KHz

    # Mandamos por la pipe

#shutdown the stream
sdr.closeStream(rxStream)
sdr.deactivateStream(rxStream) #stop streaming
