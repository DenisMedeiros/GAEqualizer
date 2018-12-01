#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from comm import Transmitter, Channel, Receiver


# Configuration.

SNRdB = 7  # Signal-to-Noise ratio in dB.
SNR = 10.0 ** (SNRdB/10.0)
NTAPS = 5  # Number of taps in the multipath channel.
FD = 5  # Doppler frequency of the channel.

# Creation of the transmitter, channel, and receiver.
transmitter = Transmitter()
channel = Channel(SNR, NTAPS, FD)
receiver = Receiver()



# Simulation.

bits = '0101101011110000010'
symbols = transmitter.bits2symbols(bits)
symbolsn = channel.apply_awgn(symbols)

print(symbols)
print(symbolsn)

#plt.plot(symbols)
#plt.plot(symbolsn)

#

