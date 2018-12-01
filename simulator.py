#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from comm import Transmitter, Channel, Receiver

''' Configuration. '''
N = 1000  # Number of bits to send.

SNRdB = 100  # Signal-to-Noise ratio in dB.
SNR = 10.0 ** (SNRdB/10.0)
NTAPS = 1  # Number of taps in the multipath channel.
INITIAL_DELAY = 0  # Initial delay (number of symbols).
FD = 5  # Doppler frequency of the channel.

# Symbols table following gray code sequence.
SYMBOLS_TABLE1 = {
    '0': 1 * np.exp(1j * np.radians(45)),
    '1': 1 * np.exp(1j * np.radians(225)),
}

SYMBOLS_TABLE2 = {
    '00': 1 * np.exp(1j * np.radians(45)),
    '01': 1 * np.exp(1j * np.radians(135)),
    '11': 1 * np.exp(1j * np.radians(225)),
    '10': 1 * np.exp(1j * np.radians(315)),
}

SYMBOLS_TABLE3 = {
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
transmitter = Transmitter(SYMBOLS_TABLE2)
channel = Channel(SNR, NTAPS, INITIAL_DELAY, FD)
receiver = Receiver(transmitter)

'''Simulation'''

# Generate a random sequence of 0's and 1's.
a_bits = np.random.choice(['0', '1'], size=N)
bits = ''.join(a_bits.tolist())

# Transmitter codification.
symbols = transmitter.process(bits)

# Channel processing.
symbols_c = channel.process(symbols)

# Receiver decodification.
bits_r = receiver.process(symbols_c)

# Evaluation of the results.
#print('[1] Bits sent:     {} '.format(bits))
#print('[2] Bits received: {} '.format(bits_r))

a_bits_r = np.array(list(bits_r))
a_bits = a_bits[:a_bits_r.size:]

n_errors = np.count_nonzero(a_bits != a_bits_r)
ber = n_errors/a_bits.size

print('[3] Number of errors: {} from {} symbols.'.format(n_errors, a_bits_r.size))
print('[4] Bit error rate: {}.'.format(ber))

