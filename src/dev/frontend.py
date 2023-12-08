import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants import numpy as np #use numpy for buffers
from scipy.signal import firwin 
import numpy as np
from demodulator import Demodulator, params_
from dataclasses import dataclass
from collections import deque
import math

@dataclass
class trails_:
    yi1 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
    yi2 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
    yq1 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
    yq2 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
    out_buf= np.zeros([])
    dfi = 0 # Decimation First Index
    out_buffer = np.zeros([])

@dataclass
class params_:
    f0 = 14.095e6
    fs = 250e3
    bw = 200e3
    f_decoder = 25e3
    in_buffer_len = 2048
    out_buffer_len = 6000

trails = trails_()
params = params_()

decimation_factor = int(math.florr(params.fs/params.f_decoder))

args = dict(driver="sdrplay")
sdr = SoapySDR.Device(args)

# Apply settings
sdr.setAntenna(SOAPY_SDR_RX, 0, "Antenna C")
sdr.setDCOffsetMode(SOAPY_SDR_RX, 0, True)  
sdr.setGainMode(SOAPY_SDR_RX, 0, True) #AGC
sdr.writeSetting("iqcorr_ctrl", True)   # I/Q Correction
sdr.writeSetting("biasT_ctrl", False)   # Disable Bias-T
sdr.writeSetting("rfnotch_ctrl", True)  # Enable rf notch filer
sdr.writeSetting("dabnotch_ctrl", True) # Enable dab notch filter

sdr.setBandwidth(SOAPY_SDR_RX, 0, params.bw) # IF bandwidth (compatible with zero IF and low IF, can't configure which?)
sdr.setSampleRate(SOAPY_SDR_RX, 0, params.fs) # Sampling frequency
sdr.setFrequency(SOAPY_SDR_RX, 0, params.f0) # 14.1 MHz is 5kHz above

# Setup a stream (complex floats)
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream) #start streaming

# Create a re-usable buffer for rx samples
buff = np.array([0]*params.in_buffer_len, np.complex64)

# Read the filter coeficients
h_bp_5k_I = np.loadtxt("bp_5k_real.fcf")
h_bp_5k_Q = np.loadtxt("bp_5k_imag.fcf")

demod = Demodulator()

out_buffer = deque(maxlen=6000) # Warning: overflows silently

while True:
    sr = sdr.readStream(rxStream, [buff], len(buff))

    data_I = np.real(buff) 
    data_Q = np.imag(buff)

    # Band pass filter centered in 5KHz
    # In-phase part
    [yi1, trails.yi1] = demod.convolve_rt(data_I, h_bp_5k_I, trails.yi1)
    [yi2, trails.yi2] = demod.convolve_rt(data_Q, h_bp_5k_Q, trails.yi2)
    # Quadrature part
    [yq1, trails.yq1] = demod.convolve_rt(data_I, h_bp_5k_Q, trails.yq1)
    [yq2, trails.yq2] = demod.convolve_rt(data_Q, h_bp_5k_I, trails.yq2)

    # In-phase and quadrature parts of the input filtered signal
    yi = yi1 - yi2
    yq = yq1 + yq2
    y = yi + 1j*yq

    # Decimate to 25Khz
    y = y[trails.dfi::decimation_factor]
    trails.dft = (trails.dft + 1) % 8  
    # The 8 comes from: ((in_buffer_len / decimation_factor) % 1) * decimation_factor = ((2048 / 10) % 1) * 10 = 0.8 * 10 = 8

    # Envelope
    y = np.abs(y)

    # len(y) = floor(in_buffer_len / decimation_factor) = floor(2048 / 10) = 204
    if (overflow := (len(out_buffer)+204) - params.out_buffer_len) > 0:
        # Buffer overflow, save it
        out_buffer.extend(y[:-overflow])
        trails.out_buffer = y[-overflow:]
    else:
        out_buffer.extend(y)

sdr.closeStream(rxStream) # shutdown stream
sdr.deactivateStream(rxStream) #stop streaming
