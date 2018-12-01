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


    def apply_awgn(self, signal):

        # Calculate the signal power.
        signalp = np.sum(np.square(signal)) / signal.size

        # Generate the AWGN (mean = 0, variance = noisep.
        noisep = signalp / self.snr
        std = np.sqrt(noisep)
        awgn = np.random.normal(0, std, signal.size)

        # Apply the noise to the signal.
        signaln = signal + awgn

        return signaln


    def transmit(self, signal):

        symbols = self.bits2symbols(signal)


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

        plt.show()
