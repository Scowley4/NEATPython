import os
import sys
import numpy as np

INPUT = 'input'
OUTPUT = 'output'
HIDDEN = 'hidden'
BIAS = 'bias'

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
        in_nodes = [NodeGene(self.get_next_node_num(), node_type=INPUT)
                    for i in inputs]
        bias_nodes = [NodeGene(self.get_next_node_num(), node_type=BIAS)]
        out_nodes = [NodeGene(self.get_next_node_num(), node_type=OUTPUT)
                     for i in outputs]
        links = []

        # Make the first genome
        genesis_genome = Genome(nodes=in_nodes+bias_nodes+out_nodes, links=links)

        # Make the population just this genome
        self.population = [genesis_genome]

        # Make the first spec
        spec_num = self.get_next_species_num()
        spec = Species(spec_num, self.gen_num)
        spec.add_genome(genesis_genome)

        self.species[spec_num] = spec


    def next_epoch():
        # Calculate adjusted fitness for species

        # Mark for death

        # Calculate expected offspring for each organism
        pass


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

    def __len__(self):
        return len(self.genomes)

    def add_genome(self, genome):
        """Adds a genome to the species."""
        self.genomes.add(genome)

    def sort_genomes(self, reverse=True):
        """Sorts the gnomes inplace.

        Defaults to sorting highest to lowest. Uses the __lt__ function on
        the Genome class.
        """
        self.genomes.sort(reverse=reverse)

    def get_average_fitness(self):
        return sum(g.fitness for g in self.genomes)/len(self.genomes)

    def get_champion(self):
        """Returns the best genome in the species."""
        return max(self.genomes)

    def get_random_genome(self):
        """Chooses a uniform random genome from the species."""
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
    # I think the parameters actually belong to the genome to rank it's
    # compatibility with other genomes. Since we don't really want to have to
    # give each genome the parameters (waste of memory and it's verbose), maybe
    # make them class parameters and have the containing class own the class or
    # modify the parameters on this class? Not quite sure.
    c1 = 1
    c2 = 1
    c3 = 1

    def __init__(self, nodes, links):
        self.node_genes = []
        self.link_genes = []
        self.fitness = None
        self.adj_fitness = None
        self.species_hint = None # id of the genome's parent species


    def __lt__(self, other):
        # This would order the genomes as in the paper, with fitness as first
        # criterion and # of link genes as second.
        return ((self.fitness, -len(self.link_genes)) <
                (other.fitness, -len(other.link_genes)))

    def get_fitness(self):
        return self.fitness

    def copy(self):
        """Performs a deep copy of the genome."""
        new_genome = Genome()
        new_genome.node_genes = [gene.copy() for gene in self.node_genes]
        new_genome.link_genes = [gene.copy() for gene in self.link_genes]
        new_genome.fitness = self.fitness
        new_genome.adj_fitness = self.adj_fitness
        new_genome.species_hint = self.species_hint
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
        # genetics.cpp:819 Genome::mutate_add_node
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

    def mutate_toggle_enable(self):
        # See gnetics.cpp:779 - must check to make sure the in-node has other
        # enabled out-node links
        pass

    def mutate_gene_reenable(self):
        # Not sure why the naming is different from the above

        # genetics.cpp:797 - Just finds the "first" disabled gene and reenables
        # it. Not sure if the genes are sorted in any meaningful way other than
        # when they were added? "First" in this case means the first in the
        # list of genes, NOT the gene that was first disabled.
        pass

    def mutate_add_link(self):
        pass

    def mutate_link_weights(self, perturb_prob=.9, cold_prob=.1):
        """Attempts a mutation on all links in the genome.

        Mutation rate is determined by the sum of perturb_prob and cold_prob.
        """
        # genetics.cpp:737 - Looks like they either just add a random value
        # in (-1,1) or they make the weight a value (-1,1). This seems a bit
        # odd. Also, not sure why they say "GAUSSIAN" since I think they are
        # using a uniform value. This is complicated somewhat by the power and
        # powermod, but randposneg()*randfloat() just yields a random number in
        # (-1,1). These functions are defined in networks.h

        # Their code for this section contains much more than was described in
        # the paper. For now, I'm implementing it as it sounds from the paper
        # "There was an 80% chance of a genome having its connection weights
        # mutated, in which case each weight had a 90% chance of being
        # uniformly perturbed and a 10% chance of being assigned a new random
        # value.

        if perturb_prob + cold_prob > 1:
            raise ValueError('perturb_prob + cold_prob cannot be greater than 1')
        for g in self.link_genes:
            r = random.random()
            weight_change = random.uniform(-1,1)
            if r < perturb_prob:
                g.weight += weight_change
            elif r < perturb_prob+cold_prob:
                g.weight = weight_change
            # Else do nothing to that weight


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

    def get_disjoint(self, other):
        # Wasn't sure if this should return the genes, the innov numbers, or
        # the Innovations
        """Returns the innovation numbers that are disjoint from this genome.

        Disjoint genes are genes within the max innovation number of P1 that are
        not included in P1. Shown below as D3.

        P1 - G1 G2    G4
        P2 - G1 G2 D3 G4 E5
        """
        innovs = {g.innov_num for g in self.link_genes}
        max_innov = max(innovs)
        return [g.innov_num for g in other.link_genes
                if g.innov_num < max_innov and g.innov_num not in innovs]

    def get_excess(self, other):
        """Returns the innovation numbers that are excess to this genome

        Excess genes are genes outside the max innovation number of P1. Shown
        below as E5.

        P1 - G1 G2    G4
        P2 - G1 G2 D3 G4 E5
        """
        innovs = {g.innov_num for g in self.link_genes}
        max_innov = max(innovs)
        return [g.innov_num for g in other.link_genes
                if g.innov_num > max_innov and g.innov_num not in innovs]

    def get_network(self):
        """Returns the network representation of this genome (the phenotype)"""

        # Find which nodes are input and which are output. We may want to store
        # this info somewhere else (like in the genome)

        inputs = []
        outputs = []
        node_num = dict() #Map from node_id to zero index node number

        for i, node in enumerate(self.node_genes):
            # Create mapping
            node_num[node.node_id] = i

            # Store input and output node_numbers
            if node.is_input:
                inputs.append(i)
            elif node.is_output:
                outputs.append(i)

        # Create edge list.
        for gene in self.link_genes:
            edges.append((node_num[gene.to_node], node_num[gene.from_node], gene.weight))


        # Build an adjacency matrix for the network
        n = len(node_num)
        adj_matrix = np.zeros((n, n))
        for e in edges:
            adj_matrix[e[:2]] = e[2]

        return Network(adj_marix, inputs, outputs)




class Network:
    """"""
    def __init__(self, adj_matrix, input_nodes, output_nodes):

        # Network adjacency matrix
        self.A = adj_matrix
        self.inputs = input_nodes
        self.outputs = output_nodes
        # Node activation values
        self.node_vals = np.zeros((self.A.shape[0], 1))

        # Bool to check if node was activated in the current time step
        self.active_nodes = np.zeros((self.A.shape[0], 1), dtype=bool)

        # We probably need to add something to be a BIAS node to this function
        def activate(inputs, max_iters=100):
            """ Returns the acvtivation values of the output nodes. These are
            computed by passing a signal through the network until all output nodes are
            active. If after max_iter iterations, the output nodes remain off,
            an array of nans is returned instead.

            Additionally, we reactivate the input nodes at each time step.
            """
            # Label inputs as active
            self.active_nodes[self.inputs] = True

            # While some output nodes are inactive, pass the signal farther
            # through the network

            i=0
            while not self.active_nodes[self.outputs].all():

                # Activate inputs
                self.node_vals[self.inputs] = inputs

                # Drive the activations one time step farther through the network
                self.node_vals = self.activation_func(self.A.dot(self.node_vals))

                # Keep track of new node activations
                self.active_nodes = (self.A != 0).dot(self.active_nodes) or self.active_nodes

                # Stop if all output nodes are active
                if self.active_nodes[output].all():
                    break

                i += 1
                # Stop if the number of iterations exceeds max_iters
                if i > max_iters:
                    return np.array([np.nan]*len(self.outputs))

            return self.node_vals[self.outputs]




# Potentially just a data class
class LinkGene:
    """"""
    def __init__(self, from_node, to_node, weight, innov, enabled=True):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        # Should this be the Innovation or the innov number?
        self.innov = innov
        self.innvo_num = None
        self.enabled = enabled

        self.attrs = (from_node, to_node, weight, innov, enabled)

    def copy(self):
        return LinkGene(*self.attrs)


# Potentially just a data class
# NODE TYPES: INPUT, OUTPUT, BIAS, HIDDEN
# - INPUT : no other inputs to these nodes
# - BIAS  : no other inputs to this node
# - OUTPUT: can have both inputs and outputs
# - HIDDEN: can have both inputs and outputs
# Which basically breaks them into 2 catagories. Although, the Network needs to
# know about all 4 catagories since it needs to put inputs in the right place,
# have a constant BIAS, and collect outputs.
class NodeGene:
    """"""
    def __init__(self, node_id, node_type=HIDDEN, is_sensor=False, is_input=False):
        self.node_id = node_id
        self.node_type = node_type
        self.is_output = is_output
        self.is_input = is_input



# genetics.cpp:3254 - "Remove the innovations of the current generation"
# genetics.cpp:1424 - Yet they still check if the genes match up without using
# a global system of innovation... Seems like it would be better to not reset
# each go. Or at least keep the innovation numbers on the genes globally
# incrementing. I guess it kinda makes sense.. but why even keep tract of
# innovations at all? That's not quite right.
#
# Okay, if you get two genes with the same innovation numbers, and one or each
# of them gets mutated. Later, you come back and mate them, choosing one or
# the other gene.
# However, a third gene gets a mutation on a later generation producing a link
# in the exact same spot. When this genome tries to mate with one from above,
# we copy the first parent's gene? Doesn't make sense.
# Potentially just a data class
class Innovation:
    """"""
    def __init__(self,):
        self.node_in = node_in
        self.node_out = node_out
        self.innov_num1 = innov_num1
        self.innov_num2= innov_num2
        # Their code remembers the weight that this innovation used, assigning
        # it to the next time this innovation occurs
        # genetics.cpp:1234 - This weight is uniform (-10,10)
        self.new_weight = new_weight
        self.newnode_id = newnode_id
        self.old_innov_num = old_innov_num
        self.recursive = recursive

    def __eq__(self, other):
        return (
                (self.node_in       == other.node_in) and
                (self.node_out      == other.node_out) and
                (self.innov_num1    == other.innov_num1) and
                (self.innove_num2   == other.innove_num2) and
                (self.new_weight    == other.new_weight) and
                (self.newnode_id    == other.newnode_id) and
                (self.old_innov_num == other.old_innov_num) and
                (self.recursive     == other.recursive)
               )

