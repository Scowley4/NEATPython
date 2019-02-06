import os
import sys

class NEAT:
    """"""
    def __init__(self):
        population = None
        pass

class Population:
    """"""
    def __init__(self, pop_size=150, d_t=3.0, c1=1.0, c2=1.0, c3=0.4,
                 crossover_rate=.75, interspecies_crossover=0.001):
        self.pop_size = pop_size
        self.d_t = d_t
        self.c1 = c1 # Excess gene coefficient
        self.c2 = c2 # Disjoin gene coefficient
        self.c3 = c3 # Weight different coefficient

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

    def spawn_initial(self):
        """Population of all the same topology with weights slightly
        perturbed"""
        # See genetics.cpp:2498

    def speciate(self):
        """Separates organisms into species.

        Checks compatibility of each organism against each species, using the
        species_hint to speed up iteration, as stated in
        https://www.cs.ucf.edu/~kstanley/neat.html .

        Organisms not compatible with any species start a new species."""
        pass

    def reproduce(self):
        # reps = [random.choice(spec) for spec in self.species]
        # Species get to produce a different number of offspring in proportion
        # to
        pass

class Species:
    """"""
    def __init__(self):
        self.id = None
        self.start_gen = 0
        self.average_fitness = None
        self.gens_since_improvement = 0
        self.genomes = []
        pass

    def add_genome(self, genome):
        self.genomes.add(genome)

    def sort_genomes(self):
        pass

    def get_average_fitness(self):
        pass

    def get_champion(self):
        pass

    def get_random(self):
        pass

    def get_next_gen(self, num_offspring):
        # interspecies offspring?
        # mutation
        # crossover
        pass

    def reproduce(self):
        pass

    def adjust_fitness(self):
        # NOTE I don't believe this is found in the paper
        # see genetics.cpp:2668 "Can change the fitness of the organisms in the
        # species to be higher for very new species (to protect them)"
        pass

class Genome:
    """"""
    def __init__(self):
        self.link_genes = []
        self.node_genes = []
        self.fitness = None
        self.adj_fitness = None
        self.species_hint = None # id of the genome's parent species

    def copy(self):
        new_genome = Genome()
        new_genome.link_genes = [gene.copy() for gene in self.link_genes]
        new_genome.node_genes = [gene.copy() for gene in self.node_genes]
        new_genome.fitness = self.fitness
        new_genome.adj_fitness = self.adj_fitness
        return new_genome

    def mutate_add_node(self):
        link_gene = random.choice(self.link_genes)

        # Disable the link
        link_gene.enabled = False

        # New node gene
        node_id = len(node_genes) + 1
        innov_num = get_node_innov_num(node_id)
        new_node = NodeGene(node_id,innov_num)
        self.node_genes.append(new_node)

        # Two new links
        gene1 = None

    def mutate_link_weight(self):
        pass

    def mutate_toggle_enable(self):
        # See gnetics.cpp:779 - must check to make sure the in-node has other
        # enabled out-node links
        pass

    def mutate_gene_reenable(self):
        # Not sure why the naming is different from the above
        pass

    def mutate_add_link(self):
        pass

    def change_weight(self):
        pass

    def get_mutation(self):
        new_genome = self.copy()

        new_genome.add_node()

    def get_crossover(self, other):
        # Choose randomly when genes match up
        # Only inherit disjoin and excess genes from more fit parent
        # If they are the same fitness, use the smaller genome's disjoint and excess genes only
        # genetics.cpp 1351

        # Disabled in either parent means 75% chance of disabled in child
        pass

    def get_compatibility(self, other):
        pass

    def get_network(self):
        """Returns the network representation of this genome (the phenotype)"""
        # TODO DJ
        # Figure out how activations propogate
        pass


# Potentially just a data class
class LinkGene:
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

# Potentially just a data class
class NodeGene:
    """"""
    def __init__(self, node_id, innov):
        self.node_id = node_id
        self.innov = innov

# Potentially just a data class
class Innovation:
    """"""
    def __init__(self):
        self.node_in = None
        self.node_out = None
        self.innov_num_1 = None
        self.innov_num_2 = None
        self.new_weight = None
        self.newnode_id = None
        self.old_innov_num = None
        self.recursive = None
        pass

