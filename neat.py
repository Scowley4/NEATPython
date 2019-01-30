import os
import sys

class Population:
    """"""
    def __init__(self, pop_size=150, d_t=3.0, c1=1.0, c2=1.0, c3=0.4,
                 crossover_rate=.75, interspecies_crossover=0.001):
        self.pop_size = pop_size
        self.d_t = d_t
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.species = []
        self.innovation_num = -1

    def get_next_innovation(self):
        self.innovation_num += 1
        return self.innovation_num

    def compute_pop_fitness(self, fitness_func):
        for spec in species:
            for genome in spec:
                genome.fitness = fitness_func(genome)
                genome.adj_fitness = genome.fitness/len(spec)

    def reproduce(self):
        reps = [random.choice(spec) for spec in self.species]


class Genome:
    """"""
    def __init__(self):
        self.genes = []
        self.fitness = None
        self.adj_fitness = None

    def copy(self):
        new_genome = Genome()
        new_genome.genes = [gene.copy() for gene in self.genes]
        new_genome.fitness = self.fitness
        new_genome.adj_fitness = self.adj_fitness
        return new_genome

    def add_random_node(self):
        connection_gene = random.choice(self.genes)

        # Disable the connection
        connection_gene.enabled = False

        # Two new connections
        gene1 = None

    def add_random_connection(self):
        pass

    def change_weight(self):
        pass

    def get_mutation(self):
        new_genome = self.copy()

        new_genome.add_node()


class Gene:
    """"""
    def __init__(self, from_node, to_node, weight, innov, enabled=True):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.innov = innov
        self.enabled = enabled

        self.attrs = (from_node, to_node, weight, innov, enabled)

    def copy(self):
        return Gene(*self.attrs)
