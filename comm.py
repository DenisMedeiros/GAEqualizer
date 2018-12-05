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

        labels = np.array(list(self.symbols.keys()))
        symbols = np.array(list(self.symbols.values()))

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
        self.h_c = np.zeros(n_paths, dtype=complex)
        self.t = 0

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
        '''
        t = self.ts * np.arange(0, symbols.size, 1) + self.current_t

        # Update current t.
        self.current_t += symbols.size * self.ts

        # Create the impulse response of the channel.
        h = np.empty(symbols.size, dtype=complex)
        for i in np.arange(0, symbols.size, 1):
            h[i] = self.fading(t[i])

        # Process the symbols.
        y = np.multiply(symbols, h)
        '''

        gain = self.fading(self.ts * self.t)
        self.h_c = np.full(self.n_paths, gain)

        self.h_c = [0.5 + 1.4j, 1.9 - 1.1j, 0.5 + 1.4j, 2.1 - 2.1j]

        y = np.convolve(symbols, self.h_c)[:symbols.size:]


        # Apply AWGN noise and return.
        return self.apply_awgn(y)



class Equalizer:

    def __init__(self, optimizer, n_taps):

        if issubclass(optimizer, Optimizer):
            raise TypeError('The equalizer must receive a optimizer.')

        self.optimizer = optimizer
        self.n_taps = n_taps

    '''Train the equalizer'''
    def train(self, symbols, symbols_c):
        # Use the optimizer to find the impulse response for the equalizer.
        self.h_eq = self.optimizer.process(self.n_taps, symbols, symbols_c)

    def process(self, symbols):
        if self.h_eq is None:
            raise RuntimeError('The equalizer is not trained yet.')

        symbols_eq = np.convolve(symbols, self.h_eq)
        # Ignore the last samples.
        return symbols_eq[:symbols.size:]
    

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




