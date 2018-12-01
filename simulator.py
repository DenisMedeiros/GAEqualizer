#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from comm import Transmitter, Channel, Receiver


''' Configuration. '''
SNRdB = 100  # Signal-to-Noise ratio in dB.
SNR = 10.0 ** (SNRdB/10.0)
NTAPS = 1  # Number of taps in the multipath channel.
INITIAL_DELAY = 0  # Initial delay (number of symbols).
FD = 5  # Doppler frequency of the channel.

# Symbols table following gray code sequence.
SYMBOLS_TABLE1 = {
    '00': 1 * np.exp(1j * np.radians(45)),
    '01': 1 * np.exp(1j * np.radians(135)),
    '11': 1 * np.exp(1j * np.radians(225)),
    '10': 1 * np.exp(1j * np.radians(315)),
}

SYMBOLS_TABLE2 = {
    '000': 1 * np.exp(1j * np.radians(22.5)),
    '001': 1 * np.exp(1j * np.radians(67.5)),
    '011': 1 * np.exp(1j * np.radians(112.5)),
    '010': 1 * np.exp(1j * np.radians(157.5)),
    '110': 1 * np.exp(1j * np.radians(202.5)),
    '111': 1 * np.exp(1j * np.radians(247.5)),
    '101': 1 * np.exp(1j * np.radians(292.5)),
    '100': 1 * np.exp(1j * np.radians(337.5)),
}


# Creation of the transmitter, channel, and receiver.
transmitter = Transmitter(SYMBOLS_TABLE1)
channel = Channel(SNR, NTAPS, INITIAL_DELAY, FD)
receiver = Receiver(transmitter)


'''Simulation'''
bits = '00011011'

# Transmitter codification.
symbols = transmitter.process(bits)

# Channel processing.
symbols_c = channel.process(symbols)

# Receiver decodification.
bits_r = receiver.process(symbols_c)

# Evaluation of the results.
print('Bits sent:     {} '.format(bits))
print('Bits received: {} '.format(bits_r))

a_bits = np.array(list(bits))
a_bits_r = np.array(list(bits_r))

n_errors = np.count_nonzero(a_bits != a_bits_r)
ber = n_errors/a_bits.size

print('Number of errors: {}'.format(n_errors))
print('Bit error rate: {}'.format(ber))

