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
        self.species_dropoff_age = 15 #Not sure about this


        self.species = dict()
        self.population = []
        self.gen_num = 0
        self.species_num = -1
        self.innov_num = -1
        self.node_num = -1

    def get_next_species_num(self):
        self.species_num += 1
        return self.species_num

    def get_next_innov_num(self):
        self.innov += 1
        return self.innov

    def get_next_node_num(self):
        self.node_num += 1
        return self.node_num

    def compute_pop_fitness(self, fitness_func):
        for spec in species:
            for genome in spec:
                genome.fitness = fitness_func(genome)
                genome.adj_fitness = genome.fitness/len(spec)

    def spawn_initial_population(self, inputs, outputs):
        """Population of all the same topology with weights slightly
        perturbed"""
        # See genetics.cpp:2498
        # In their code, they initialize a genome from a file and use that to
        # make the initial population.

        # I would prefer to start with no connections and mutate the
        # connections in as needed.
        in_nodes = [NodeGene(self.get_next_node_num(), is_sensor=True)
        out_nodes = [NodeGene(self.get_next_node_num(), is_sensor=False)
        links = []

        # Make the first genome
        genesis_genome = Genome(nodes=in_nodes+out_nodes, links=links)

        # Make the population just this genome
        self.population = [genesis_genome]

        # Make the first spec
        spec_num = self.get_next_species_num()
        spec = Species(spec_num, self.gen_num)
        spec.add_genome(genesis_genome)

        self.species[spec_num] = spec



    def speciate(self):
        """Separates organisms into species.

        Checks compatibility of each organism against each spec, using the
        species_hint to speed up iteration, as stated in
        https://www.cs.ucf.edu/~kstanley/neat.html .

        Organisms not compatible with any spec start a new spec."""


        # Clear out the previous generation
        for spec in self.species.values():
            # spec.flush?
            pass

        for genome in self.population:
            if genome.species_hint is not None:
                # check compatibility with that species champion
                pass
            else:
                for spec in self.species:
                    # check compatibility until found
                    pass
                else: # make a new spec
                    spec_num = self.get_next_species_num()
                    spec = Species(spec_num, self.gen_num)
                    spec.add_genome(genome)
                    self.species[spec_num] = spec

        # Delete unnecessary species
        for spec_num, spec in list(self.species.items()):
            if len(spec)==0:
                self.species.pop(spec_num)


    def remove_unimproved_species(self):
        """Removes all species that haven't improved for some time"""
        for spec_num, spec in list(self.species.items()):
            if self.gen_num - spec.last_improved_gen > 15:
                self.species.pop(spec_num)

    def reproduce(self):
        # reps = [random.choice(spec) for spec in self.species]
        # Species get to produce a different number of offspring in proportion
        # to
        pass

class Species:
    """"""
    def __init__(self, species_num, start_gen):
        self.species_num = species_num
        self.start_gen = start_gen
        self.average_fitness = None
        self.last_improved_gen = 0
        self.genomes = []
        pass

    def __len__(self):
        return len(self.genomes)

    def add_genome(self, genome):
        self.genomes.add(genome)

    def sort_genomes(self, reverse=True):
        """Sorts the gnomes by fitness.

        Defaults to sorting highest to lowest."""
        self.genomes.sort(key=lambda genome: genome.get_fitness(), reverse=reverse)

    def get_average_fitness(self):
        pass

    def get_champion(self):
        return max(self.genomes, key=lambda genome: genome.get_fitness())

    def get_random_genome(self):
        return random.choice(self.genomes)

    def get_next_generation(self, num_offspring):
        pass

    def reproduce(self):
        # interspecies offspring?
        # mutation
        # crossover

        # genetics.cpp:3419 - If species gets more than 5 offspring,
        # clone the species champ

        # genetics.cpp:3579 - interspecies mating tends toward better species.
        # This is juged based on the species size (since this indirectly
        # represents the fitness of the species).
        # dad (parent 2) is the species champ of other species

        # genetics.cpp:1354 - Between two parents, first order in higher
        # fitness, second ordering is lower gene count.
        # Looks like if the fitness and the gene count is the same, they just
        # use the second parents excess/disjoin (as if it was better). Very
        # arbitrary.


        pass

    def adjust_fitness(self):
        # see genetics.cpp:2668 "Can change the fitness of the organisms in the
        # species to be higher for very new species (to protect them)"
        # NOTE I don't believe this is found in the paper
        pass

class Genome:
    """"""
    c1 = 1
    c2 = 1
    c3 = 1

    def __init__(self, nodes, links):
        self.node_genes = []
        self.link_genes = []
        self.fitness = None
        self.adj_fitness = None
        self.species_hint = None # id of the genome's parent species

    def get_fitness(self):
        return self.fitness

    def copy(self):
        new_genome = Genome()
        new_genome.node_genes = [gene.copy() for gene in self.node_genes]
        new_genome.link_genes = [gene.copy() for gene in self.link_genes]
        new_genome.fitness = self.fitness
        new_genome.adj_fitness = self.adj_fitness
        return new_genome

    def calculate_compatibility(self, other):

        # Get the number of genes for each
        gene_count1 = self.get_gene_count()
        gene_count2 = other.get_gene_count()

        # They do not use this N in their code, even though they explain it
        # this way in their paper.
        # genetics.cpp:2273
        N = min(gene_count1, gene_count2)
        N = N if max(gene_count1, gene_count2)>20 else 1

        pass



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

    def get_disjoin(self, other):
        pass

    def get_excess(self, other):
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
        return LinkGene(*self.attrs)

# Potentially just a data class
class NodeGene:
    """"""
    def __init__(self, node_id, is_sensor=False):
        self.node_id = node_id
        self.is_sensor = is_sensor

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

    def __eq__(self, other):
        return (
                (self.node_in == other.node_in) and
                (self.node_out == other.node_out) and
                (self.innov_num_1 == other.innov_num_1) and
                (self.innov_num_2 == other.innov_num_2) and
                (self.new_weight == other.new_weight) and
                (self.newnode_id == other.newnode_id) and
                (self.old_innov_num == other.old_innov_num) and
                (self.recursive == other.recursive)
               )

