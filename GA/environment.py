import numpy as np


class Environment:
    def __init__(self, fitness_function):
        self.evaluate_fitness_value_each_chromosome = fitness_function

    def evaluate_fitness_value(self, group):
        fitness_values = [self.evaluate_fitness_value_each_chromosome(one=one) for one in group]
        # exp normalization
        fitness_values = np.array(fitness_values)
        fitness_values = np.exp(fitness_values)
        fitness_values /= np.sum(fitness_values)

        return fitness_values.tolist()
