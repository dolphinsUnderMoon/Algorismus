import population
import environment
import numpy as np


class Darwin:
    def __init__(self, _population, _environment, darwin_config):
        self.population = _population
        self.environment = _environment
        self.breeding_rate = darwin_config['breeding_rate']
        self.mutation_rate = darwin_config['mutation_rate']
        self.posterity_amount = int(1. / self.breeding_rate + 1)

        self.fitness_values_after_selection = None

    @staticmethod
    def compute_fitness_value_threshold(fitness_values, breeding_rate):
        breeding_amount = len(fitness_values) * breeding_rate + 1
        temp = fitness_values
        temp.sort()
        return temp[-breeding_amount]

    @staticmethod
    def crossover(parents):
        length = parents[0].shape[0]
        num_parents = len(parents)
        next_generation = []
        for i in range(length):
            next_generation.append(parents[np.random.randint(0, num_parents)][i])

        return np.array(next_generation)

    @staticmethod
    def random_sample(_list, num_chosen):
        if num_chosen < len(_list):
            return None

        chosen = []
        for i in range(num_chosen):
            chosen_index = np.random.randint(0, num_chosen-i)
            chosen.append(_list[chosen_index])
            del _list[chosen_index]

        return chosen

    def selection(self):
        fitness_values = self.environment.evaluate_fitness_value(self.population.chromosomes)
        self.fitness_values_after_selection = fitness_values

        breeding_threshold = self.compute_fitness_value_threshold(fitness_values, self.breeding_rate)

        for i in range(self.population.size):
            if fitness_values[i] < breeding_threshold:
                del self.population.chromosomes[i]
                del self.fitness_values_after_selection[i]

        self.fitness_values_after_selection = np.array(self.fitness_values_after_selection)
        self.fitness_values_after_selection /= np.sum(self.fitness_values_after_selection)
        self.fitness_values_after_selection = self.fitness_values_after_selection.tolist()

    def breeding(self):
        posterity = []

        parents = self.random_sample(self.population.chromosomes, 2)
