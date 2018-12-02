#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from comm import Transmitter, Channel, Receiver, Equalizer


''' Configuration. '''
TN = 100 # Number of bits used to train the equalizer.
N = 1000  # Number of bits to send.

SNRdB = 7  # Signal-to-Noise ratio in dB.
SNR = 10.0 ** (SNRdB/10.0)
N_PATHS = 10  # Number of taps in the multipath channel.
INITIAL_DELAY = 0  # Initial delay (number of symbols).
FD = 5  # Doppler frequency of the channel.

TAPS_EQ = 10

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

# Creation of the transmitter, channel, equalizer, and receiver.
transmitter = Transmitter(SYMBOLS_TABLE2)
channel = Channel(SNR, N_PATHS, INITIAL_DELAY, FD)
receiver = Receiver(transmitter)
equalizer = Equalizer(TAPS_EQ)

# Train the equalizer.
a_training_bits = np.random.choice(['0', '1'], size=100)
training_bits = ''.join(a_training_bits.tolist())

training_symbols = transmitter.process(training_bits)
training_symbols_c = channel.process(training_symbols)
equalizer.train(training_symbols, training_symbols_c)

'''Simulation'''
# Generate a random sequence of 0's and 1's.
a_bits = np.random.choice(['0', '1'], size=N)
bits = ''.join(a_bits.tolist())

# Transmitter codification.
symbols = transmitter.process(bits)

# Channel processing.
symbols_c = channel.process(symbols)

#symbols_eq = equalizer.process(symbols_c)
symbols_eq = symbols_c

#print(symbols)
#print(symbols_c)
#print(symbols_eq)

# Receiver decodification.
bits_r = receiver.process(symbols_eq)

# Evaluation of the results.
#print('[1] Bits sent:     {} '.format(bits))
print('[2] Bits received: {} '.format(bits_r))

a_bits_r = np.array(list(bits_r))

min_size = np.min((a_bits.size, a_bits_r.size))

a_bits = a_bits[:min_size:]
a_bits_r = a_bits_r[:min_size:]

n_errors = np.count_nonzero(a_bits != a_bits_r)
ber = n_errors/a_bits.size

print('[3] Number of errors: {} from {} symbols.'.format(n_errors, a_bits_r.size))
print('[4] Bit error rate: {}.'.format(ber))

