# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Transmitter:

    def __init__(self, symbols):
        self.symbols = symbols
        self.bps = len(list(symbols.keys())[0])

    '''
    Converts a sequence of bits to a sequence of symbols, defined in SYMBOLS.
    '''
    def bits2symbols(self, bits):

        # If the num of bits is not multiple of bps, discard the last bits.
        nbits = len(bits)

        if nbits % self.bps != 0:
            remainder = nbits % self.bps
            bits = bits[0:nbits-remainder:]
            nbits -= remainder

        nsymbols = int(nbits / self.bps)
        symbols = np.empty(nsymbols, dtype=complex)
        for i in np.arange(0, nbits, self.bps):
            key = bits[i:i + self.bps:1]
            symbols[int(i/self.bps)] = self.symbols[key]

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

        plt.show(block=False)


class Channel:

    def __init__(self, snr, ntaps, fd):
        self.snr = snr
        self.ntaps = ntaps
        self.fd = fd

    '''
    Generate AWGN complex noise.
    '''
    def apply_awgn(self, symbols):

        # Calculate the average signal power.
        symbolsp = np.sum(np.square(np.abs(symbols))) / symbols.size

        # Generate the complex AWGN (mean = 0, variance = noisep).
        noisep = symbolsp / self.snr

        # Since there are the real and imaginary part, the average power of
        # the noise must be divided by 2.
        std = np.sqrt(noisep/2)

        awgnr = np.random.normal(0, std, symbols.size)
        awgni = 1j*np.random.normal(0, std, symbols.size)
        awgn = awgnr + awgni

        # Apply the noise to the signal.
        symbolsn = symbols + awgn

        return symbolsn


    def process(self, symbols):
        pass



class Receiver:
    pass