#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from comm import Transmitter, Channel, Receiver, Equalizer
from optimizers import GeneticAlgorithm, LeastMeanSquares, ParticleSwarmOptimization
import matplotlib.pyplot as plt

''' Configuration. '''

# Channel configuration.
SNRdB = 100  # Signal-to-Noise ratio in dB.
SNR = 10.0 ** (SNRdB/10.0)
N_PATHS = 4  # Number of taps in the multipath channel.
DOPPLER_F = 10  # Doppler frequency of the channel.
TS = 0.1  # Sampling time.

# Symbols table following gray code sequence.
SYMBOLS_TABLE1 = {
    '0': 1 * np.exp(1j * np.radians(45)),
    '1': 1 * np.exp(1j * np.radians(235)),
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

# Equalizer configuration.
TN = 100  # Number of bits used to train the equalizer.
TAPS_EQ = 4  # Number of equalizer taps.

# Genetic algorithm configuration.
POP_SIZE = 256
ELITE_INDS = 1
GA_MAX_NUM_GEN = 1024
GA_MAX_MSE = 0.4
CX_PB = 1.0
MUT_PB = 0.001
GA_L_MIN = -1.0
GA_L_MAX = 1.0

# Least mean square configuration.
EPOCHS = 50
ETA = 0.01
LMS_MAX_MSE = 0.4

# Particle swarm optimization configuration.
NUM_PART = 40
PSO_MAX_NUM_GEN = 40
PSO_MAX_MSE = 0.4
COG = 0.3
SOCIAL = 0.7
INERTIA = 0.7
PSO_L_MIN = -1.0
PSO_L_MAX = 1.0

'''Simulation'''

# Creation of the transmitter, channel, optimizers, equalizer, and receiver.
transmitter = Transmitter(SYMBOLS_TABLE2)
channel = Channel(SNR, N_PATHS, DOPPLER_F, TS)
receiver = Receiver(transmitter)

ga = GeneticAlgorithm(
    pop_size=POP_SIZE,
    elite_inds=ELITE_INDS,
    max_num_gen=GA_MAX_NUM_GEN,
    max_mse=GA_MAX_MSE,
    cx_pb=CX_PB,
    mut_pb=MUT_PB,
    l_min=GA_L_MIN,
    l_max=GA_L_MAX,
    report=True,
)

lms = LeastMeanSquares(
    epochs=EPOCHS,
    eta=ETA,
    max_mse=LMS_MAX_MSE,
    report=True,
)

pso = ParticleSwarmOptimization(
    num_part=NUM_PART,
    max_num_gen=PSO_MAX_NUM_GEN,
    max_mse=PSO_MAX_MSE,
    cog=COG,
    social=SOCIAL,
    inertia=INERTIA,
    l_min=PSO_L_MIN,
    l_max=PSO_L_MAX,
    report=True,
)


#equalizer = Equalizer(ga,  TAPS_EQ)
#equalizer = Equalizer(lms, TAPS_EQ)
equalizer = Equalizer(pso, TAPS_EQ)


# Train the equalizer.
a_training_bits = np.random.choice(['0', '1'], size=TN)
training_bits = ''.join(a_training_bits.tolist())

training_symbols = transmitter.process(training_bits)
training_symbols_c = channel.process(training_symbols)
equalizer.train(training_symbols, training_symbols_c)

# Prepare the signal to sendo.
N = 1000
a_bits = np.random.choice(['0', '1'], size=N)
bits = ''.join(a_bits.tolist())

# Transmitter codification.
symbols = transmitter.process(bits)

# Channel processing.
symbols_c = channel.process(symbols)

#symbols_eq = equalizer.process(symbols_c)
symbols_eq = symbols_c

# Receiver decodification.
bits_r = receiver.process(symbols_eq)

# Evaluation of the results.


a_bits_r = np.array(list(bits_r))

min_size = np.min((a_bits.size, a_bits_r.size))

a_bits = a_bits[:min_size:]
a_bits_r = a_bits_r[:min_size:]

n_errors = np.count_nonzero(a_bits != a_bits_r)
ber = n_errors/a_bits.size

#print('[1] Bits sent:     {} '.format(bits))
#print('[2] Bits received: {} '.format(bits_r))
print('[3] Impulse response of the channel:     {} '.format(channel.h_c))
print('[4] Impulse response of the equalizer:     {} '.format(equalizer.h_eq))
print('[5] Number of errors: {} from {} symbols.'.format(n_errors, a_bits_r.size))
print('[6] Bit error rate: {0:.1f} %.'.format(ber * 100))

