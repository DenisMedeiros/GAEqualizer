# -*- coding: utf-8 -*-

import numpy as np

class GeneticAlgorithm:

    def __init__(self, pop_size, elite_inds, max_num_gen, max_mse, cx_pb, mut_pb, mu, sigma, n_taps):
        self.pop_size = pop_size
        self.elite_inds = elite_inds
        self.max_num_gen = max_num_gen
        self.max_mse = max_mse
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.mu = mu
        self.sigma = sigma
        self.n_taps = n_taps

        # Initialize the population.
        self.population = sigma * (np.random.randn(pop_size, n_taps) + 1j * np.random.randn(pop_size, n_taps)) + mu
        self.new_population = np.empty((pop_size, n_taps), dtype=complex)
        self.best_individuals = np.empty(n_taps, dtype=complex)

        # Create the fitnesses vector.
        self.fitnesses = np.empty(self.pop_size)


    # Each individual is a sequence of complex numbers.
    def evaluation(self, individual):
        symbols_eq = np.convolve(self.symbols_c, individual)
        mse = np.mean((np.abs(self.symbols - symbols_eq[:self.symbols.size:])))
        return mse

    def process(self, symbols, symbols_c):

        self.symbols = symbols
        self.symbols_c = symbols_c
        best_individual = None

        # Evaluate the entire population.
        for l in np.arange(self.pop_size):
            self.fitnesses[l] = self.evaluation(self.population[l])

        # If the elitism is activated.
        if self.elite_inds > 0:
            elite_individuals = np.empty((self.elite_inds, self.n_taps), dtype=complex)

        # Process the generations.
        k = 0
        best_fitness = float('inf')
        while k < self.max_num_gen and best_fitness > self.max_mse:  # Stop criteria

            # Save the best individuals (for elitism, if it is activated).
            if self.elite_inds > 0:
                ordered_args = np.argsort(self.fitnesses)
                elite_individuals = self.population[ordered_args[0:self.elite_inds:1]]

            # Selection process.
            for l in np.arange(0, self.pop_size, 2):

                # Pick 4 individuals.
                indexes = np.random.randint(0, self.pop_size, size=4)

                if self.fitnesses[indexes[0]] < self.fitnesses[indexes[1]]:
                    selected1 = indexes[0]
                else:
                    selected1 = indexes[1]

                if self.fitnesses[indexes[2]] < self.fitnesses[indexes[3]]:
                    selected2 = indexes[2]
                else:
                    selected2 = indexes[3]

                # Crossover (for each tap).
                for m in np.arange(self.n_taps):

                    if np.random.rand() < self.cx_pb:  # Crossover test.

                        # One point.
                        weight1 = np.random.rand()
                        weight2 = 1 - weight1

                        offspring1 = weight1 * self.population[selected1][m] + weight2 * self.population[selected2][m]
                        offspring2 = weight1 * self.population[selected2][m] + weight2 * self.population[selected1][m]

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

            # Apply the elitism, if it is activated.
            if self.elite_inds > 0:
                self.new_population[0:self.elite_inds:1] = elite_individuals

            # Update the old population.
            self.population = self.new_population.copy()

            # Evaluate the entire population.
            for l in np.arange(self.pop_size):
                self.fitnesses[l] = self.evaluation(self.population[l])

            # Report.
            best_ind_index = np.argmin(self.fitnesses)
            best_fitness = self.fitnesses[best_ind_index]
            best_individual = self.population[best_ind_index]

            # print('gen = {}, min = {:.2}, avg = {:.2}, best = {}'.format(
            # k, np.min(self.fitnesses), np.mean(self.fitnesses), best_individual))
            # print('k = {}, best fitness = {}'.format(k, best_fitness))

            # Next generation.
            k += 1

        return best_individual








