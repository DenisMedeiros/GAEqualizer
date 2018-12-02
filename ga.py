# -*- coding: utf-8 -*-

import numpy as np

class GeneticAlgorithm:

    def __init__(self, pop_size, num_gen, cx_pb, mut_pb, elite_inds, mu, sigma, n_taps, symbols, symbols_c):
        self.pop_size = pop_size
        self.num_gen = num_gen
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.elite_inds = elite_inds
        self.mu = mu
        self.sigma = sigma
        self.n_taps = n_taps
        self.symbols = symbols
        self.symbols_c = symbols_c

        # Initialize the population.
        self.population = sigma * (np.random.randn(pop_size, n_taps) + 1j * np.random.randn(pop_size, n_taps)) + mu
        self.new_population = np.empty((pop_size, n_taps), dtype=complex)
        self.best_individuals = np.empty(n_taps, dtype=complex)

        if self.elite_inds > 0:
            self.elite_individuals = np.empty((elite_inds, n_taps), dtype=complex)

        print('initial population')
        print()
        print(self.population)

        # Create the fitnesses vector.
        self.fitnesses = np.empty(self.pop_size)

        # Evaluate the entire population.
        for l in np.arange(self.pop_size):
            self.fitnesses[l] = self.evaluation(self.population[l])

    # Each individual is a sequence of complex numbers.
    def evaluation(self, individual):
        #symbols_eq = np.convolve(self.symbols, individual)
        #mse = np.mean((np.abs(self.symbols - symbols_eq[self.n_taps - 1::])))
        #return mse
        #print(individual)
        foco = -10 + 1j * 10
        value = np.abs(foco - individual[0])

        return value

    def process(self):

        # Process the generations.
        for k in np.arange(self.num_gen):

            # Save the best individuals (for elitism).
            if self.elite_inds > 0:
                args_fitnesses = np.argsort(self.fitnesses)
                self.elite_individuals = self.population[args_fitnesses[0:self.elite_inds:1]]

            # Selection process.
            for l in np.arange(0, self.pop_size, 2):

                # Pick 4 individuals.
                inds = np.random.randint(0, self.pop_size, size=4)

                if self.fitnesses[inds[0]] < self.fitnesses[inds[1]]:
                    selected1 = inds[0]
                else:
                    selected1 = inds[1]

                if self.fitnesses[inds[2]] < self.fitnesses[inds[3]]:
                    selected2 = inds[2]
                else:
                    selected2 = inds[3]

                # Crossover (for each tap).
                for m in np.arange(self.n_taps):

                    if np.random.rand() < self.cx_pb:  # Crossover test.

                        # One point.
                        weight1 = np.random.rand()
                        weight2 = 1 - weight1

                        offspring1 = weight1 * self.population[selected1][m] + weight2 * self.population[selected2][m]
                        offspring2 = weight1 * self.population[selected2][m] + weight2 * self.population[selected1][m]

                        # Mutation.
                        self.new_population[l][m] = offspring1
                        self.new_population[l+1][m] = offspring2
                    else:
                        self.new_population[l][m] = self.population[l][m]
                        self.new_population[l+1][m] = self.population[l][m]

            # Mutation.
            for l in np.arange(0, self.pop_size, 1):
                if np.random.rand() < self.mut_pb:  # Mutation test.
                    for m in np.arange(self.n_taps):
                        mutation = self.sigma * (np.random.randn() + 1j * np.random.randn()) + self.mu
                        self.new_population[l][m] += mutation

            # Apply the elitism.
            if self.elite_inds > 0:
                self.new_population[0:self.elite_inds:1] = self.elite_individuals

            # Update the old population.
            self.population = self.new_population.copy()

            # Evaluate the entire population.
            for l in np.arange(self.pop_size):
                self.fitnesses[l] = self.evaluation(self.population[l])

            # Report.
            best_ind_index = np.argmin(self.fitnesses)
            self.best_individual = self.population[best_ind_index]

            print('gen = {}, min = {:.2}, avg = {:.2}, best = {}'.format(k, np.min(self.fitnesses), np.mean(self.fitnesses), self.best_individual))


        return self.best_individual






