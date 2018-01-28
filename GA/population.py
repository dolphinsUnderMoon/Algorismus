import numpy as np


class Chromosome:
    def __init__(self, length=10, type='binary'):
        self.type = type
        self.length = length
        self.genes = None

        if self.type == 'binary':
            self.genes = np.random.randint(0, 2, size=self.length)
        elif self.type == 'float':
            pass

    def get_gene(self, index):
        if 0 < index < self.length and self.genes is not None:
            return self.genes[index]
        else:
            return None

    def set_genes(self, new_genes):
        self.genes = new_genes


class Population:
    def __init__(self, initial_config):
        self.population_size = initial_config['population_size']
        self.chromosomes_type = initial_config['chromosomes_type']
        self.chromosomes_length = initial_config['chromosomes_length']

        self.chromosomes = self.initial_chromosomes(self.population_size,
                                                    self.chromosomes_length,
                                                    self.chromosomes_type)

    @staticmethod
    def initial_chromosomes(size=10, length=10, type='binary'):
        return [Chromosome(length, type=type) for _ in range(size)]
