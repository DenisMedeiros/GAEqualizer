# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Channel:

    # Symbols table following gray code sequence.
    SYMBOLS = {
        '00': -1 - 1j,
        '01': -1 + 1j,
        '11': 1 + 1j,
        '10': 1 - 1j,
    }

    def __init__(self, snr, ntaps, fd):
        self.snr = snr
        self.ntaps = ntaps
        self.fd = fd

    def bits2symbols(self, bits):

        # If the num of bits is not even, discard the last bit.
        nbits = len(bits)

        if nbits % 2 != 0:
            bits = bits[:-1:]
            nbits -= 1

        nsymbols = nbits >> 1
        symbols = np.empty(nsymbols, dtype=complex)
        for i in np.arange(0, nbits, 2):
            key = bits[i:i + 2:1]
            symbols[i >> 1] = self.SYMBOLS[key]

        return symbols

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


    def send(self, signal):

        symbols = self.bits2symbols(signal)



    def receive(self):


    def plot_symbols(self):

        labels = np.array(list(self.SYMBOLS.keys()))
        symbols = np.array(list(self.SYMBOLS.values()))

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
