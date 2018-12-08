# -*- coding: utf-8 -*-

import numpy as np
import abc


class Optimizer(abc.ABC):
    """
    A optimizer used by the channel equalizer. Every optimizer must implement a
    method called process, which returns the weights of the equalizer.
    """

    @abc.abstractmethod
    def process(self, n_taps, symbols, symbols_c, report=False):
        # Finds the 'n_taps' weights of the equalizer.
        return


class LeastMeanSquares(Optimizer):

    """A optimizer that uses the LSM algorithm."""

    def __init__(self, epochs, eta, max_mse, report=False):
        self.epochs = epochs
        self.eta = eta
        self.max_mse = max_mse
        self.report = report

    def process(self, n_taps, symbols, symbols_c, report=False):

        weights = np.zeros(n_taps, dtype=complex)
        input_frame = np.zeros(n_taps, dtype=complex)

        k = 0
        mse = float('inf')

        if self.report:
            print(40 * '-')
            print('{0:<4s} {1:>4s}'.format('epoch', 'mse'))
            print(40 * '-')

        while k < self.epochs and mse > self.max_mse:  # Stop criteria
            for l in np.arange(0, symbols.size, 1):
                input_frame[1::] = input_frame[0:-1:]  # Sliding window.
                input_frame[0] = symbols_c[l]  # Current symbol.

                # Separate real and imaginary parts.
                output = weights.dot(input_frame.T)

                error = symbols[l] - output

                weights += self.eta * error * np.conj(input_frame)

                mse = np.mean(np.abs(error) ** 2)

            k += 1
            if self.report:
                print('{0:<4d} {1:>8.4f}'.format(k, mse))

        # Returns the equalizer weights.
        return weights


class GeneticAlgorithm(Optimizer):

    """A optimizer that uses the Genetic algorithm."""

    def __init__(self, pop_size, elite_inds, max_num_gen,
                 max_mse, cx_pb, mut_pb, l_min, l_max, report=False):
        self.pop_size = pop_size
        self.elite_inds = elite_inds
        self.max_num_gen = max_num_gen
        self.max_mse = max_mse
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.l_min = l_min
        self.l_max = l_max
        self.report = report

    def process(self, n_taps, symbols, symbols_c, report=False):

        # Each individual is a sequence of complex numbers.
        def evaluation(individual):

            symbols_eq = np.zeros(symbols_c.size, dtype=complex)

            for k in np.arange(n_taps - 1, symbols.size, 1):
                symbols_eq[k] = symbols_c[k] - individual[1] * symbols_eq[k - 1] - individual[2] * symbols_eq[k - 2] - \
                                individual[3] * symbols_eq[k - 3]

            mse = np.mean((np.abs(symbols - symbols_eq)**2))
            return mse

        # Generates a complex random number with the specified size.
        def complex_rand(size):
            real = np.random.uniform(self.l_min, self.l_max, size)
            imag = np.random.uniform(self.l_min, self.l_max, size)
            return real + 1j * imag


        # Initialize the population.
        population = complex_rand((self.pop_size, n_taps))
        population[:][0] = 1

        new_population = np.empty((self.pop_size, n_taps), dtype=complex)
        best_individual = None

        # Create the fitnesses vector.
        fitnesses = np.empty(self.pop_size)

        # Evaluate the entire population.
        for l in np.arange(self.pop_size):
            fitnesses[l] = evaluation(population[l])

        # If the elitism is activated.
        if self.elite_inds > 0:
            elite_individuals = np.empty((self.elite_inds, n_taps), dtype=float)

        # Process the generations.
        k = 0
        best_fitness = float('inf')

        if self.report:
            print(40 * '-')
            print('{0:<4s} {1:>6s} {2:>8s} {3:>8s}'.format('gen', 'min', 'max', 'avg'))
            print(40 * '-')

        while k < self.max_num_gen and best_fitness > self.max_mse:  # Stop criteria

            # Save the best individuals (for elitism, if it is activated).
            if self.elite_inds > 0:
                ordered_args = np.argsort(fitnesses)
                elite_individuals = population[ordered_args[0:self.elite_inds:1]]

            # Selection process.
            for l in np.arange(0, self.pop_size, 2):

                # Pick 4 individuals.
                indexes = np.random.randint(0, self.pop_size, size=4)

                if fitnesses[indexes[0]] < fitnesses[indexes[1]]:
                    selected1 = indexes[0]
                else:
                    selected1 = indexes[1]

                if fitnesses[indexes[2]] < fitnesses[indexes[3]]:
                    selected2 = indexes[2]
                else:
                    selected2 = indexes[3]

                # Crossover (for each tap).
                for m in np.arange(n_taps):

                    if np.random.rand() < self.cx_pb:  # Crossover test.

                        # One point.
                        weight1 = np.random.rand()
                        weight2 = 1 - weight1

                        offspring1 = weight1 * population[selected1][m] + weight2 * population[selected2][m]
                        offspring2 = weight1 * population[selected2][m] + weight2 * population[selected1][m]

                        new_population[l][m] = offspring1
                        new_population[l+1][m] = offspring2
                    else:
                        new_population[l][m] = population[l][m]
                        new_population[l+1][m] = population[l+1][m]

            # Mutation.
            for l in np.arange(0, self.pop_size, 1):
                if np.random.rand() < self.mut_pb:  # Mutation test.
                    for m in np.arange(n_taps):
                        new_population[l][m] += complex_rand(1)

            # Apply the elitism, if it is activated.
            if self.elite_inds > 0:
                new_population[0:self.elite_inds:1] = elite_individuals

            # Update the old population.
            population = new_population.copy()

            # Evaluate the entire population.
            for l in np.arange(self.pop_size):
                fitnesses[l] = evaluation(population[l])

            # Report.
            best_ind_index = np.argmin(fitnesses)
            best_fitness = fitnesses[best_ind_index]
            best_individual = population[best_ind_index]

            # print('gen = {}, min = {:.2}, avg = {:.2}, best = {}'.format(
            # k, np.min(self.fitnesses), np.mean(self.fitnesses), best_individual))


            if self.report:
                print('{0:<4d} {1:>8.4f} {2:>8.4f} {3:>8.4f}'.format(k, best_fitness, np.max(fitnesses), np.average(fitnesses)))

            # Next generation.
            k += 1

        return best_individual


class ParticleSwarmOptimization(Optimizer):

    """A optimizer that uses the PSO."""

    def __init__(self, num_part, max_num_gen, max_mse, cog, social, inertia, l_min, l_max, report=False):
        self.num_part = num_part
        self.max_num_gen = max_num_gen
        self.max_mse = max_mse
        self.cog = cog
        self.social = social
        self.inertia = inertia
        self.report = report
        self.l_min = l_min
        self.l_max = l_max

    def process(self, n_taps, symbols, symbols_c):

        # Each individual is a sequence of complex numbers.
        def evaluate(individual):

            symbols_eq = np.zeros(symbols_c.size, dtype=float)

            for k in np.arange(n_taps - 1, symbols.size, 1):
                symbols_eq[k] = symbols_c[k] - individual[1] * symbols_eq[k - 1] - individual[2] * symbols_eq[k - 2] - \
                                individual[3] * symbols_eq[k - 3]


            mse = np.mean((np.abs(symbols - symbols_eq[n_taps-1::])**2))
            return mse

        # Generates a complex random number with the specified size.
        def complex_rand(size):
            real = np.random.uniform(self.l_min, self.l_max, size)
            imag = np.random.uniform(self.l_min, self.l_max, size)
            #return real + 1j * imag
            return real

        # Start the particle swarm.

        positions = complex_rand((self.num_part, n_taps))
        velocities = complex_rand((self.num_part, n_taps))
        pbest = positions.copy()
        gbest = positions[0].copy()

        k = 0
        mse = float('inf')

        if self.report:
            print(40 * '-')
            print('{0:<8s} {1:>8s} {2:>8s}'.format('gen', 'mse', 'avg'))
            print(40 * '-')

        while k < self.max_num_gen and mse > self.max_mse:  # Stop criteria.

            total = 0.0

            # Update positions and velocities.
            for l in np.arange(self.num_part):
                velocities[l] = self.inertia * velocities[l] + \
                    np.random.rand() * self.cog * (pbest[l] - positions[l]) + \
                    np.random.rand() * self.social * (gbest - positions[l])

                positions[l] += velocities[l]

                # Correct if the particle left the search space.
                for m in np.arange(n_taps):

                    pos_r = positions[l][m]

                    if pos_r < self.l_min:
                        pos_r = self.l_min
                    elif pos_r > self.l_max:
                        pos_r = self.l_max

                    positions[l][m] = pos_r

            # Evaluate all particles.
            for l in np.arange(self.num_part):
                evaluation = evaluate(positions[l])

                total += evaluation

                if evaluation < evaluate(pbest[l]):
                    pbest[l] = positions[l]
                if evaluation < evaluate(gbest):
                    gbest = positions[l]

            mse = evaluate(gbest)

            if self.report:

                print('{0:<8d} {1:>8.4f} {1:>8.4f}'.format(k, mse, total/self.num_part))

            k += 1

        return gbest


