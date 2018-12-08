# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from optimizers import Optimizer


class Transmitter:

    def __init__(self, symbols_table):
        self.symbols_table = symbols_table
        self.bps = len(list(symbols_table.keys())[0])  # bit per symbol.

    '''
    Converts a sequence of bits to a sequence of symbols, defined in SYMBOLS.
    '''
    def process(self, bits):

        # If the num of bits is not multiple of bps, discard the last bits.
        n_bits = len(bits)

        if n_bits % self.bps != 0:
            remainder = n_bits % self.bps
            bits = bits[0:n_bits-remainder:]
            n_bits -= remainder

        n_symbols = int(n_bits / self.bps)
        symbols = np.empty(n_symbols, dtype=complex)
        for i in np.arange(0, n_bits, self.bps):
            key = bits[i:i + self.bps:1]
            symbols[int(i/self.bps)] = self.symbols_table[key]

        return symbols

    '''
    Plots the symbols table in the complex plane.
    '''
    def plot_symbols(self):

        labels = np.array(list(self.symbols_table.keys()))
        symbols = np.array(list(self.symbols_table.values()))

        # Get magnitude and angle of the symbols.
        mags = np.abs(symbols)
        angles = np.angle(symbols)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        fig.suptitle('Symbols', fontsize=14)
        ax.scatter(angles, mags)

        # Add the labels.
        for i in np.arange(0, symbols.size, 1):
            ax.annotate(labels[i],
                        xy=(angles[i], mags[i]),  # theta, radius
                        xytext=(angles[i], mags[i]),
                        textcoords='data',
                        fontweight='bold',
                        fontsize='12'
                        )

        plt.show()


class Channel:

    def __init__(self, snr, n_paths, doppler_f, ts):
        self.snr = snr
        self.n_paths = n_paths
        self.doppler_f = doppler_f
        self.ts = ts

        # Current time of the channel.
        self.t = 0

        # Channel.
        #self.h_c = np.random.randn(n_paths) + 1j * np.random.randn(n_paths)

        #self.h_c = np.array([-1.29-2.68j, -2.01-0.16j, -0.44+0.08j, -0.41+0.40]) # LMS wins
        #self.h_c = np.array([1.07694553 + 0.30761342j, 1.17000129 + 0.75604768j, 0.57177548 - 0.75905257j, 1.35273347 + 1.14252183j])  # GA wins
        #self.h_c = np.array([0.35451883+1.9773459j, 0.31706077+0.6454735j, -0.42064853+0.38354962j, -0.54249331+0.52682473j])
        #self.h_c = np.array([-1.64699742+0.24993397j, 0.7218241-0.1544417j, 0.72504389+2.16147108j,
         #1.02918822-0.30857183j, 0.57359915-1.08949372j, 0.81431953+0.2221552j, -1.45853365-0.27140351j,
         #-0.97335389+2.13144022j, 0.40445492+0.34621335j, -0.24596483+0.81525798j]) PSO wins

        #self.h_c = np.random.randn(n_paths) + 1j * np.random.randn(n_paths)

        self.h_c = np.array([1.08+0.31j, 1.17+0.76j, 0.57-0.76j, 1.35+1.14j, 0.68-1.27j, -0.19+0.02j])


    # Pop-Beaulieu Simulator
    def fading(self, t):
        wd = 2 * np.pi * self.doppler_f
        n = self.n_paths
        phi_n = np.random.uniform(-np.pi, np.pi, n)
        h_r = np.sum(np.cos(wd * t * np.cos(2 * np.pi * np.arange(1, n + 1, 1) / n) + phi_n))
        h_i = np.sum(np.sin(wd * t * np.cos(2 * np.pi * np.arange(1, n + 1, 1) / n) + phi_n))
        h_t = (1 / np.sqrt(n)) * (h_r + 1j * h_i)
        return h_t

    '''Generate AWGN complex noise.'''
    def apply_awgn(self, symbols):

        # Calculate the average signal power.
        symbols_power = np.sum(np.square(np.abs(symbols))) / symbols.size

        # Generate the complex AWGN (mean = 0, variance = noisep).
        noise_power = symbols_power / self.snr

        # Since there are the real and imaginary part, the average power of
        # the noise must be divided by 2.
        std = np.sqrt(noise_power/2)

        awgnr = np.random.normal(0, std, symbols.size)
        awgni = 1j*np.random.normal(0, std, symbols.size)
        awgn = awgnr + awgni

        # Apply the noise to the signal.
        symbols_n = symbols + awgn

        return symbols_n

    def process(self, symbols):

        symbols_c = np.zeros(symbols.size, dtype=complex)

        # Channel time-varying
        '''
        for k in np.arange(self.n_paths-1, symbols.size, 1):
            # For every symbol, the channel changes.
            for l in np.arange(0, self.n_paths, 1):
                self.h_c[l] = self.fading(k * self.ts + self.ts)
                self.t += self.ts
            for l in np.arange(0, self.n_paths, 1):
                symbols_c[k] += self.h_c[l] * symbols[k-l]
        '''

        # Static channel.
        '''
        for k in np.arange(self.n_paths-1, symbols.size, 1):
            for l in np.arange(0, self.n_paths, 1):
                symbols_c[k] += self.h_c[l] * symbols[k-l]
        '''

        # With convolution (must change the GA and PSO).
        symbols_c = np.convolve(symbols, self.h_c, 'same')

        # Apply AWGN noise and return.
        return self.apply_awgn(symbols_c)



class Equalizer:

    def __init__(self, optimizer, n_taps):

        if issubclass(optimizer, Optimizer):
            raise TypeError('The equalizer must receive a optimizer.')

        self.optimizer = optimizer
        self.n_taps = n_taps
        self.h_eq = None

    '''Train the equalizer'''
    def train(self, symbols, symbols_c):
        # Use the optimizer to find the impulse response for the equalizer.
        self.h_eq = self.optimizer.process(self.n_taps, symbols, symbols_c)

    def process(self, symbols_c):
        if self.h_eq is None:
            raise RuntimeError('The equalizer is not trained yet.')

        symbols_eq = np.zeros(symbols_c.size, dtype=complex)

        for k in np.arange(self.n_taps-1, symbols_c.size, 1):
            for l in np.arange(0, self.n_taps, 1):
                symbols_eq[k] += self.h_eq[l] * symbols_c[k-l]

        return symbols_eq


class Receiver:

    def __init__(self, transmitter):

        symbols_table = transmitter.symbols_table

        self.bps = len(list(symbols_table.keys())[0])

        # Generate the points of the constellation.
        self.symbols_points = list(symbols_table.values())

        # Convert each block of bits to an array of chars.
        n_symbols = len(symbols_table)
        bits_blocks = list(symbols_table.keys())

        self.symbols_bits = np.empty((n_symbols, self.bps), dtype=str)
        for i in np.arange(0, n_symbols, 1):
            self.symbols_bits[i, :] = np.array(list(bits_blocks[i]))

    '''Makes a decision and choose a symbol.'''
    def process(self, symbols_c):

        # Create a empty vector for each bit.
        bits = np.full(symbols_c.size * self.bps, 'x', dtype=str)

        k = 0
        for i in np.arange(0, symbols_c.size, 1):
            # Get the current symbol.
            symbol = symbols_c[i]
            # Calculate the euclidean distances from the constellations points.
            euclidean_distance = np.abs(symbol - self.symbols_points)

            # Choose the lower distance and get the related bits.
            chosen_pos = np.argmin(euclidean_distance)
            chosen_bits = self.symbols_bits[chosen_pos]
            bits[k:k+self.bps:1] = chosen_bits
            k += self.bps

        return ''.join(bits.tolist())




