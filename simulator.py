#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from comm import Channel


# Configuration.

SNRdB = 7  # Signal-to-Noise ratio in dB.
SNR = 10.0 ** (SNRdB/10.0)
NTAPS = 5  # Number of taps in the multipath channel.
FD = 5  # Doppler frequency of the channel.

# Creation of the channel.
channel = Channel(SNR, NTAPS, FD)


# Simulation.

bits = '0101101011110000010'
symbols = channel.bits2symbols(bits)
symbolsn = channel.apply_awgn(symbols)

print(symbols)
print(symbolsn)

#plt.plot(symbols)
#plt.plot(symbolsn)

#

