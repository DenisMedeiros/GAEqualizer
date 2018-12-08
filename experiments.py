import numpy as np
from comm import Transmitter, Channel, Receiver, Equalizer
from optimizers import GeneticAlgorithm, LeastMeanSquares, ParticleSwarmOptimization
import matplotlib.pyplot as plt
import time

# Channel configuration.
SNRdB = 3  # Signal-to-Noise ratio in dB.
SNR = 10.0 ** (SNRdB/10.0)
N_PATHS = 2  # Number of taps in the multipath channel.
DOPPLER_F = 20  # Doppler frequency of the channel.
TS = 0.1  # Sampling time.

# Symbols table following gray code sequence.
SYMBOLS_TABLE1 = {
    '0': 1.0 * np.exp(1j * np.radians(45)),
    '1': 1.0 * np.exp(1j * np.radians(225)),
}

SYMBOLS_TABLE2 = {
    '00': 1.0 * np.exp(1j * np.radians(45)),
    '01': 1.0 * np.exp(1j * np.radians(135)),
    '11': 1.0 * np.exp(1j * np.radians(225)),
    '10': 1.0 * np.exp(1j * np.radians(315)),
}

SYMBOLS_TABLE3 = {
    '000': 1.0 * np.exp(1j * np.radians(22.5)),
    '001': 1.0 * np.exp(1j * np.radians(67.5)),
    '011': 1.0 * np.exp(1j * np.radians(112.5)),
    '010': 1.0 * np.exp(1j * np.radians(157.5)),
    '110': 1.0 * np.exp(1j * np.radians(202.5)),
    '111': 1.0 * np.exp(1j * np.radians(247.5)),
    '101': 1.0 * np.exp(1j * np.radians(292.5)),
    '100': 1.0 * np.exp(1j * np.radians(337.5)),
}


# Equalizer configuration.
TN = 300  # Number of bits used to train the equalizer.
TAPS_EQ = 8  # Number of equalizer taps.

# Genetic algorithm configuration.
POP_SIZE = 16
ELITE_INDS = 1
GA_MAX_NUM_GEN = 100
GA_MAX_MSE = 0.4
CX_PB = 0.8
MUT_PB = 0.1
GA_L_MIN = -1.0
GA_L_MAX = 1.0

# Least mean square configuration.
EPOCHS = 100
ETA = 0.01
LMS_MAX_MSE = 0.4

# Particle swarm optimization configuration.
NUM_PART = 32
PSO_MAX_NUM_GEN = 64
PSO_MAX_MSE = 0.4
COG = 0.3
SOCIAL = 0.7
INERTIA = 0.5
PSO_L_MIN = -1.0
PSO_L_MAX = 1.0

'''Simulation'''

# Creation of the transmitter, channel, optimizers, equalizer, and receiver.
transmitter = Transmitter(SYMBOLS_TABLE3)
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
    report=False,
)

lms = LeastMeanSquares(
    epochs=EPOCHS,
    eta=ETA,
    max_mse=LMS_MAX_MSE,
    report=False,
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
    report=False,
)

# Prepare the signal to send.
N = 9000
a_bits = np.random.choice(['0', '1'], size=N)
bits = ''.join(a_bits.tolist())
symbols = transmitter.process(bits)
symbols_c = channel.process(symbols)

# Prepare signal for the training.
a_training_bits = np.random.choice(['0', '1'], size=TN)
training_bits = ''.join(a_training_bits.tolist())
training_symbols = transmitter.process(training_bits)
training_symbols_c = channel.process(training_symbols)

# Testing for different optimizers.
equalizer_ga = Equalizer(ga,  TAPS_EQ)
equalizer_lms = Equalizer(lms,  TAPS_EQ)
#equalizer_pso = Equalizer(pso,  TAPS_EQ)


#transmitter.plot_symbols()


print()
print(' ---- Training time (in ms) ----')

start = time.time()
equalizer_ga.train(training_symbols, training_symbols_c)
total = time.time() - start
print('Training of the GA = {}'.format(total * 1000))

start = time.time()
equalizer_lms.train(training_symbols, training_symbols_c)
total = time.time() - start
print('Training of the LMS = {}'.format(total * 1000))

#start = time.time()
#equalizer_pso.train(training_symbols, training_symbols_c)
#total = time.time() - start
#print('Training of the PSO = {}'.format(total * 1000))

symbols_eq_ga = equalizer_ga.process(symbols_c)
symbols_eq_lms = equalizer_lms.process(symbols_c)
#symbols_eq_pso = equalizer_pso.process(symbols_c)

print()
print(' ---- Bit Error Rate ----')

bits_r_noeq = receiver.process(symbols_c)

bits_r_ga = receiver.process(symbols_eq_ga)
bits_r_lms = receiver.process(symbols_eq_lms)
#bits_r_pso = receiver.process(symbols_eq_pso)

a_bits = np.array(list(bits))

a_bits_r_noeq = np.array(list(bits_r_noeq))
a_bits_r_ga = np.array(list(bits_r_ga))
a_bits_r_lms = np.array(list(bits_r_lms))
#a_bits_r_pso = np.array(list(bits_r_pso))

n_errors_noeq = np.count_nonzero(a_bits != a_bits_r_noeq)
n_errors_ga = np.count_nonzero(a_bits != a_bits_r_ga)
n_errors_lms = np.count_nonzero(a_bits != a_bits_r_lms)
#n_errors_pso = np.count_nonzero(a_bits != a_bits_r_pso)

ber_noeq = n_errors_noeq/a_bits.size
ber_ga = n_errors_ga/a_bits.size
ber_lms = n_errors_lms/a_bits.size
#ber_pso = n_errors_pso/a_bits.size

print('Bit error rate without equalization = {0:.1f} %.'.format(ber_noeq * 100))
print('Bit error rate of the GA = {0:.1f} %.'.format(ber_ga * 100))
print('Bit error rate of the LMS = {0:.1f} %.'.format(ber_lms * 100))
#print('Bit error rate of the PSO = {0:.1f} %.'.format(ber_pso * 100))

print()
print(' ---- Channel and equalizers impulse response ----')
print('Channel: {}'.format(channel.h_c))
print('Weights of the equalizer GA: {}'.format(equalizer_ga.h_eq))
print('Weights of the equalizer LMS: {}'.format(equalizer_lms.h_eq))
#print('Weights of the equalizer PSO: {}'.format(equalizer_pso.h_eq))