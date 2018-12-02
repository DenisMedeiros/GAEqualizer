# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from collections import Sequence
from itertools import repeat

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

    def __init__(self, snr, n_taps, i_delay, fd):
        self.snr = snr
        self.n_taps = n_taps
        self.i_delay = i_delay
        self.fd = fd
        
        # Model for Rayleigh fading channel
        delay = np.zeros(self.i_delay, dtype=complex)
        self.h = np.append(delay,
                np.random.randn(self.n_taps) + 1j * np.random.randn(self.n_taps))
                
        self.h = np.array([0.1 + 0.2j])

    '''
    Generate AWGN complex noise.
    '''
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

        # Impulse response of the channel.
        y = np.convolve(symbols, self.h)       
        
        # Discard the last samples.

        # Apply AWGN noise.
        symbols_c = self.apply_awgn(y)
        
        # Ignore the first samples.
        return symbols_c[self.h.size-1::]


class Equalizer:

    def __init__(self, n_taps):
        self.n_taps = n_taps
        self.h_eq = np.random.rand(n_taps) + 1j * np.random.rand(n_taps)
        self.n_taps = n_taps

    '''Train the equalizer'''
    def train(self, symbols, symbols_c):
    
        # Genetic algorithm configuration.
        N_INDS = 128
        N_GENERATIONS = 128
        CX_PB = 0.7
        MUT_PB = 0.1

        def evaluate(individual):
            symbols_eq = np.convolve(symbols, individual)
            mse = np.mean((np.abs(symbols - symbols_eq[self.n_taps-1::])))
            return (mse,)
        
        # Function that generates a complex random number.
        def complex_rand():
            return np.random.rand() + 1j * np.random.rand()
            
            
        def mutation(individual, mu, sigma, indpb):
            size = len(individual)

            for i in np.arange(size):
                if np.random.rand() < indpb:
                    individual[i] += sigma * (np.random.randn() + 1j*np.random.randn()) + mu
            
            return individual
            
        # Creating the types.
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization = -1.0
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
           
        toolbox = base.Toolbox()
        toolbox.register("complex_rand", complex_rand)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.complex_rand, n=self.n_taps)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", mutation, mu=0, sigma=5, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logbook = tools.Logbook()
        logbook.header = ("gen", "avg", "min", "max",)
        hof = tools.HallOfFame(1, similar=np.array_equal)
        
         # Create the population.
        population = toolbox.population(n=N_INDS)
   
        # Create the population.
        population = toolbox.population(n=N_INDS)
        
        # Evaluate the entire population.
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Process the generations.
        for gen in range(N_GENERATIONS):
            # Select the next generation individuals.
            offspring = toolbox.select(population, len(population))
            
            # Clone the selected individuals.
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring.
            #for child1, child2 in zip(offspring[::2], offspring[1::2]):
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < CX_PB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.rand() < MUT_PB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness.
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            population[:] = offspring
            hof.update(population)
            
            # Save statistics.
            record = stats.compile(population)
            logbook.record(gen=gen, evals=N_GENERATIONS, **record)
                   
        print(logbook)
        # The best individual is the equalizer weights.
        self.h_eq = hof[0]
    
    def process(self, symbols):
        symbols_eq = np.convolve(symbols, self.h_eq)
        # Ignore the first samples.
        return symbols_eq[self.n_taps-1::]
    

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




