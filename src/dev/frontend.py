import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants import numpy as np #use numpy for buffers
from scipy.signal import firwin 
import numpy as np
from demodulator import Demodulator, params_
from dataclasses import dataclass
from collections import deque
import math
from multiprocessing import Pipe

@dataclass
class params_:
    f0 = 14.095e6
    fs = 250e3
    bw = 200e3
    f_decoder = 25e3
    in_buffer_len = 2048
    out_buffer_len = 6000


class Frontend:

    def __init__(self):
        # Read the filter coeficients
        h_bp_5k_I = np.loadtxt("bp_5k_real.fcf")
        h_bp_5k_Q = np.loadtxt("bp_5k_imag.fcf")

        @dataclass
        class trails_:
            yi1 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
            yi2 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
            yq1 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
            yq2 = np.zeros(int((h_bp_5k_I.size - 1) / 2))
            out_buf= np.zeros([])
            dfi = 0 # Decimation First Index
            out_buffer = np.zeros([])
            overflow = False

        self.trails = trails_()
        self.params = params_()

        self.decimation_factor = int(math.floor(self.params.fs / self.params.f_decoder))

        #################### SDR CONFIG ####################
        args = dict(driver="sdrplay")
        self.sdr = SoapySDR.Device(args)

        # Apply settings
        self.sdr.setAntenna(SOAPY_SDR_RX, 0, "Antenna C")
        self.sdr.setDCOffsetMode(SOAPY_SDR_RX, 0, True)  
        self.sdr.setGainMode(SOAPY_SDR_RX, 0, True) #AGC
        self.sdr.writeSetting("iqcorr_ctrl", True)   # I/Q Correction
        self.sdr.writeSetting("biasT_ctrl", False)   # Disable Bias-T
        self.sdr.writeSetting("rfnotch_ctrl", True)  # Enable rf notch filer
        self.sdr.writeSetting("dabnotch_ctrl", True) # Enable dab notch filter

        self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.params.bw) # IF bandwidth (compatible with zero IF and low IF, can't configure which?)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.params.fs) # Sampling frequency
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.params.f0) # 14.1 MHz is 5kHz above

        # Setup a stream (complex floats)
        selfrxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

    def frontend(self, pipe_out) -> None:
        self.sdr.activateStream(rxStream) #start streaming

        # Create a re-usable buffer for rx samples
        buff = np.array([0]*params.in_buffer_len, np.complex64)

        out_buffer = deque(maxlen=6000) # Warning: overflows silently

        while True:
            sr = sdr.readStream(rxStream, [buff], len(buff))

            data_I = np.real(buff) 
            data_Q = np.imag(buff)

            # Band pass filter centered in 5KHz
            # In-phase part
            [yi1, trails.yi1] = Demodulator.convolve_rt(data_I, h_bp_5k_I, trails.yi1)
            [yi2, trails.yi2] = Demodulator.convolve_rt(data_Q, h_bp_5k_Q, trails.yi2)
            # Quadrature part
            [yq1, trails.yq1] = Demodulator.convolve_rt(data_I, h_bp_5k_Q, trails.yq1)
            [yq2, trails.yq2] = Demodulator.convolve_rt(data_Q, h_bp_5k_I, trails.yq2)

            # In-phase and quadrature parts of the input filtered signal
            yi = yi1 - yi2
            yq = yq1 + yq2
            y = yi + 1j*yq

            # Decimate to 25Khz
            y = y[trails.dfi::decimation_factor]
            trails.dft = (trails.dfi + 1) % 8  
            # The 8 comes from: ((in_buffer_len / decimation_factor) % 1) * decimation_factor = ((2048 / 10) % 1) * 10 = 0.8 * 10 = 8

            # Envelope
            y = np.abs(y)

            # len(y) = floor(in_buffer_len / decimation_factor) = floor(2048 / 10) = 204
            if (overflow := (len(out_buffer)+204) - params.out_buffer_len) >= 0:
                # Buffer overflow, save it
                out_buffer.extend(y[:-overflow])
                trails.out_buffer = y[-overflow:]

                # Send to demodulator
                pipe_out.send(np.array(out_buffer))

                # Clear out_buffer and set overflow flag
                out_buffer.clear()
                trails.overflow = True
            else:
                # If we got an overflow (or just sent data)
                if trails.overflow:
                    out_buffer.extend(trails.out_buffer)
                    trails.overflow = False

                # Fill out_buffer
                out_buffer.extend(y)

        sdr.closeStream(rxStream) # shutdown stream
        sdr.deactivateStream(rxStream) #stop streaming
