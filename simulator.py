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

# Symbols table following gray code sequence.
SYMBOLS1 = {
    '00': -1 - 1j,
    '01': -1 + 1j,
    '11': 1 + 1j,
    '10': 1 - 1j,
}

SYMBOLS2 = {
    '000': 1 * np.exp(1j * np.radians(0)),
    '001': 1 * np.exp(1j * np.radians(45)),
    '011': 1 * np.exp(1j * np.radians(90)),
    '010': 1 * np.exp(1j * np.radians(135)),
    '110': 1 * np.exp(1j * np.radians(180)),
    '111': 1 * np.exp(1j * np.radians(225)),
    '101': 1 * np.exp(1j * np.radians(270)),
    '100': 1 * np.exp(1j * np.radians(305)),
}


# Creation of the transmitter, channel, and receiver.
transmitter = Transmitter(SYMBOLS1)
channel = Channel(SNR, NTAPS, FD)
receiver = Receiver()


# Simulation.

bits = '010010000'
symbols = transmitter.bits2symbols(bits)
symbolsn = channel.apply_awgn(symbols)

transmitter.plot_symbols()

print(symbols)
print(symbolsn)

#plt.plot(symbols)
#plt.plot(symbolsn)

input()

