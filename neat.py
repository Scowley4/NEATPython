import os
import sys
import numpy as np
from itertools import permutations
import random

INPUT = 'input'
OUTPUT = 'output'
HIDDEN = 'hidden'
BIAS = 'bias'
NEWNODE = 'newnode'
NEWLINK = 'newlink'

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
        self.survival_thresh = 0.3
        self.species_dropoff_age = 15 #Not sure about this
        self.mutate_only_prob = 0.25 # From the paper
        self.interspecies_mate_rate = 0.001
        self.mate_only_prob = 0.2 # Don't think this is found in the paper, but
                                  # it's in their code...

        self.mutate_add_node_prob = 0.03
        self.mutate_add_link_prob = 0.05
        self.mutate_link_weights_prob = 0.8
        self.mutate_toggle_enable_prob = 0.0
        self.mutate_toggle_reenable_prob = 0.0


        self.pop_champ = None
        self.pop_champs = []
        self.overall_pop_champ = None
        self.overall_pop_champs = []

        self.gen_innovations = []

        self.species = dict()
        self.all_genomes = []
        self.gen_num = 0
        self.species_num = -1
        self.innov_num = -1
        self.node_map = dict()
        self.node_num = -1

    def get_next_species_num(self):
        self.species_num += 1
        return self.species_num

    def get_next_innov_num(self):
        self.innov_num += 1
        return self.innov_num

    def get_next_node_num(self):
        self.node_num += 1
        return self.node_num

    def compute_pop_fitness(self, fitness_func):
        for spec in self.species.values():
            for genome in spec.genomes:
                network = genome.get_network()
                genome.fitness = fitness_func(network)

    # def get_average_fitness(self):
    #     total_fit = 0
    #     n = 0
    #     for spec in self.species:
    #         for genome in spec.genomes:
    #             total_fit += genome.fitness
    #             n += 1
    #     return total_fit/n

    def get_total_fitness(self):
        return sum(g.fitness for g in self.all_genomes)

    def get_total_adj_fitness(self):
        return sum(g.adj_fitness for g in self.all_genomes)


    def get_average_fitness(self):
        return self.get_total_fitness()/len(all_genomes)

    def spawn_initial_population(self, n_inputs, n_outputs):
        """Population of all the same topology with weights slightly
        perturbed"""
        # See genetics.cpp:2498
        # In their code, they initialize a genome from a file and use that to
        # make the initial population.

        # I would prefer to start with no connections and mutate the
        # connections in as needed.
        in_nodes = [NodeGene(self.get_next_node_num(), node_type=INPUT)
                    for i in range(n_inputs)]
        bias_nodes = [NodeGene(self.get_next_node_num(), node_type=BIAS)]
        out_nodes = [NodeGene(self.get_next_node_num(), node_type=OUTPUT)
                     for i in range(n_outputs)]
        nodes = in_nodes + bias_nodes + out_nodes

        self.node_map = {n.node_id: n for n in nodes}
        self.base_nodes = [n for n in nodes]

        links = []

        # Make the first genome
        genesis_genome = Genome(self, nodes=nodes, links=links)

        # Make the population just this genome
        self.all_genomes = [genesis_genome]

        # Make the first spec
        spec_num = self.get_next_species_num()
        spec = Species(self, spec_num)
        spec.add_genome(genesis_genome)
        spec.champ = genesis_genome

        self.species[spec_num] = spec

    def get_champion(self):
        return max(self.all_genomes)

    def get_biggest(self):
        return max(self.all_genomes,
            key=lambda g: len(g.node_genes) + len(g.link_genes))

    def verify_genomes(self):
        unaccounted = []
        for spec in self.species.values():
            for g in spec.genomes:
                if g not in self.all_genomes:
                    unaccounted.append(g)
        if unaccounted:
            self.unaccounted = unaccounted
            print('unaccounted')
            sys.exit()

        double = []
        for g in self.all_genomes:
            c = sum(1 for s in self.species.values() if g in s.genomes)
            if c > 1:
                double.append(g)
                for s in self.species.values():
                    if g in s.genomes:
                        print(s.species_num)
        if double:
            self.double = double
            print('double', len(double))
            sys.exit()



    def update_overall_pop_champ(self):
        if self.overall_pop_champ is None:
            self.overall_pop_champ = self.pop_champ

        if self.pop_champ.fitness > self.overall_pop_champ.fitness:
            self.overall_pop_champs.append(self.pop_champ)
            self.overall_pop_champ = self.pop_champ
            if len(self.overall_pop_champs) > 1:
                old = self.overall_pop_champs[-2].fitness
                new = self.overall_pop_champs[-1].fitness
                print(f'**Gen: {self.gen_num}\n  PopChamp: {old} -> {new}')


    def update_pop_champ(self):
        self.pop_champ = self.get_champion()
        self.pop_champs.append(self.pop_champ)

    def get_most_nodes(self):
        return max(self.all_genomes, key=lambda g:len(g.node_genes))

    def get_most_links(self):
        return max(self.all_genomes, key=lambda g:len(g.link_genes))


    def next_epoch(self, fitness_func):
        if self.overall_pop_champ is not None:
            print('BEST', self.overall_pop_champ.fitness)
        print('Most Nodes', len(self.get_most_nodes().node_genes))
        print('Most Links', len(self.get_most_links().link_genes))
        self.gen_innovations = []
        self.gen_num += 1
        self.update_pop_champ()
        self.update_overall_pop_champ()

        new_genomes = []
        # Calculate adjusted fitness for species
        for spec in self.species.values():
            spec.adjust_fitness()
        self.remove_unimproved_species()

        tot_adj_fit = self.get_total_adj_fitness()
        tot2 = sum(s.get_total_adj_fitness() for s in self.species.values())
        if tot_adj_fit != tot2:
            print('1-   ', tot_adj_fit)
            print('2-   ', tot2)
            print(sum(sum(g.adj_fitness for g in s.genomes) for s in self.species.values()))
        print(tot_adj_fit)
        self.verify_genomes()

        total_offspring = 0
        offspring_counts = []
        percents = []
        sums = []

        for spec in self.species.values():
            spec_fitness = spec.get_total_adj_fitness()
            percent_offspring = spec_fitness / tot_adj_fit
            if len(self.species)==1:
                percent_offspring = 1
            n_offspring = int(self.pop_size * percent_offspring)
            total_offspring += n_offspring
            offspring_counts.append(n_offspring)
            percents.append(percent_offspring)
            sums.append(spec_fitness)

        if total_offspring > 160:
            print('NUM SPECIES', len(self.species))
            print(offspring_counts)
            print(percents)
            print()
            print(tot_adj_fit)
            print(sums)
            print(len(self.all_genomes))
            print(sum(len(s) for s in self.species.values()))

        print('Reproducing')
        for spec in self.species.values():
            percent_offspring = sum(g.adj_fitness/tot_adj_fit for g in spec.genomes)

            if len(self.species)==1:
                percent_offspring = 1
            n_offspring = int(self.pop_size * percent_offspring)
            new_genomes += spec.reproduce(n_offspring)

        self.all_genomes = new_genomes
        print(f'{len(self.all_genomes)} new genomes')

        before_n_species = len(self.species)
        print('Speciating')
        self.speciate()
        after_n_species = len(self.species)
        print(f'{before_n_species}->{after_n_species} species')


        print('Computing fitness')
        self.compute_pop_fitness(fitness_func)
        print()
        if len(self.all_genomes) > 160:
            sys.exit()


    def get_random_champ(self, weighted=False, spec=None):

        species = [s for s in self.species.values() if s is not spec]

        # If there is only one species
        if len(species) == 0:
            return spec.get_random_genome()

        # This is not how it's done in their code. Their function (gaussrand)
        # is weird. So I just decided to weight probabilities evenly based on
        # species size

        # genetics.cpp:3579 - interspecies mating tends toward better species.
        # This is juged based on the species size (since this indirectly
        # represents the fitness of the species).
        # dad (parent 2) is the species champ of other species
        if weighted:
            probs = np.array([len(s) for s in species])
            probs = probs / probs.sum()
            return np.random.choice(species, p=probs).get_champion()
        else:
            return np.random.choice(species).get_champion()

    def speciate(self, track=None):
        """Separates organisms into species.

        Checks compatibility of each organism against each spec, using the
        species_hint to speed up iteration, as stated in
        https://www.cs.ucf.edu/~kstanley/neat.html .

        Organisms not compatible with any spec start a new spec."""


        # Clear out the previous generation
        for spec in self.species.values():
            spec.champ = spec.get_champion()
            spec.flush()

        for genome in self.all_genomes:
            if genome.species_hint is not None:
                spec = self.species[genome.species_hint]
                if spec.is_compatible(genome):
                    spec.add_genome(genome)
                    continue

            for spec in self.species.values():
                # check compatibility until found
                if spec.is_compatible(genome):
                    spec.add_genome(genome)
                    break
            else: # make a new spec
                spec_num = self.get_next_species_num()
                spec = Species(self, spec_num)
                spec.add_genome(genome)
                spec.champ = genome
                self.species[spec_num] = spec

        # Delete unnecessary species
        for spec_num, spec in list(self.species.items()):
            if len(spec)==0:
                self.species.pop(spec_num)


    def remove_unimproved_species(self):
        """Removes all species that haven't improved for some time"""
        for spec_num, spec in list(self.species.items()):
            if self.gen_num - spec.gen_last_improved > self.species_dropoff_age:
                self.species.pop(spec_num)

class Species:
    """Holds a set of genomes, performs reproduction step."""
    def __init__(self, population, species_num):
        self.pop = population
        self.species_num = species_num
        self.gen_start = population.gen_num

        self.genomes = []

        self.average_fitness = None

        self.max_fitness_ever = -float('inf')
        self.gen_last_improved= population.gen_num

        self.champ = None

    def __len__(self):
        return len(self.genomes)

    def flush(self):
        """Removes all the genomes from the species."""
        self.genomes = []

    def add_genome(self, genome):
        """Adds a genome to the species."""
        self.genomes.append(genome)

    def sort_genomes(self, reverse=True):
        """Sorts the gnomes inplace.

        Defaults to sorting highest to lowest. Uses the __lt__ function on
        the Genome class.
        """
        self.genomes.sort(reverse=reverse)

    def get_average_fitness(self):
        return sum(g.fitness for g in self.genomes)/len(self)

    def get_total_fitness(self):
        return sum(g.fitness for g in self.genomes)

    def get_total_adj_fitness(self):
        return sum(g.adj_fitness for g in self.genomes)

    def get_champion(self):
        """Returns the best genome in the species."""
        return max(self.genomes)

    def get_random_genome(self):
        """Chooses a uniform random genome from the species."""
        return random.choice(self.genomes)

    def is_compatible(self, genome):
        return genome.calculate_compatibility(self.champ) < self.pop.d_t

    def reproduce(self, n_offspring=0):
        # NOTE It looks like they have a mate_only_prob. I haven't been able to
        # find this in the paper, but it's in genetics.cpp:3626 and it's set to
        # .2 in the settings.

        new_genomes = []

        # Remove all the unfit parents
        self.cull()

        # genetics.cpp:3419 - If species gets more than 5 offspring,
        # clone the species champ
        if n_offspring > 5:
            genome = self.get_champion()
            genome.species_hint = self.species_num
            new_genomes.append(genome)
            n_offspring -= 1


        for i in range(n_offspring):
            # get a new genome
            p1 = self.get_random_genome()
            crossed_over = False

            # If we will only mutate without crossover
            if ((random.random() < self.pop.mutate_only_prob) or
               (len(self) < 2)):
               genome = p1.copy()
            else:
                # Decide between inter/intra species
                if random.random() < self.pop.interspecies_mate_rate:
                    p2 = self.pop.get_random_champ(weighted=True,
                                                          spec=self)
                else:
                    p2 = self.get_random_genome()


                genome = p1.get_multipoint_crossover(p2)
                crossed_over = True

            # Decide if the new genome will mutate
            if (not crossed_over or
                (random.random() < self.pop.mate_only_prob)):
                genome.mutate()

            genome.species_hint = self.species_num
            new_genomes.append(genome)

        return new_genomes



    # Offspring are computed for each genome using adjusted fitness and then
    # shared with the species.

    def cull(self):
        """Remove the genomes that are not fit to be parents."""
        # genetics.cpp:2716
        num_parents = int(self.pop.survival_thresh * len(self) + 1)
        self.sort_genomes()
        self.genomes = self.genomes[:num_parents]

    def adjust_fitness(self):
        """Adjust the fitness of inidividuals based on age and time since
        improvement.
        """
        # see genetics.cpp:2668 "Can change the fitness of the organisms in the
        # species to be higher for very new species (to protect them)"
        # NOTE I don't believe this is found in the paper
        # Looks like they used a 1 for this param anyway, so it didn't do
        # anything

        cur_max = self.get_champion().fitness
        if cur_max > self.max_fitness_ever:
            self.max_fitness_ever = cur_max
            self.gen_last_improved = self.pop.gen_num

        for g in self.genomes:
            g.adj_fitness = g.fitness/len(self)

        # genetics.cpp:2699 Kill species that haven't progressed for a long
        # time by dividing fitness of all individuals in spec by 100. Weird way
        # to do it.
        if ((self.pop.gen_num - self.gen_last_improved) >
            self.pop.species_dropoff_age):
            for g in self.genomes:
                g.adj_fitness *= .01


class Genome:
    """"""
    def __init__(self, population, nodes=[], links=[]):
        for l in links:
            if not isinstance(l, LinkGene):
                print('I CAUGHT IT')
        self.pop = population
        self.node_genes = nodes
        self.link_genes = links
        self.fitness = -float('inf')
        self.adj_fitness = -float('inf')
        self.species_hint = None # id of the genome's parent species
        self.super_champ = False # marker for population champion


    def __lt__(self, other):
        # This would order the genomes as in the paper, with fitness as first
        # criterion and # of link genes as second.
        return ((self.fitness, -len(self.link_genes)) <
                (other.fitness, -len(other.link_genes)))

    def is_valid(self):
        for l in self.link_genes:
            pass
        if len(self.node_genes) != len(set(n.node_id for n in self.node_genes)):
            return False
        return True

    def get_str(self):
        return f'nodes {[n.node_id for n in self.node_genes]}\n links {[l.innov_num for l in self.link_genes]}'

    def get_fitness(self):
        return self.fitness

    def has_node_num(self, num):
        for node in self.node_genes:
            if node.node_id == num:
                return True
        return False

    def copy(self):
        """Performs a deep copy of the genome."""
        new_genome = Genome(self.pop)
        #new_genome.node_genes = [gene.copy() for gene in self.node_genes]
        new_genome.node_genes = [n for n in self.node_genes]
        new_genome.link_genes = [gene.copy() for gene in self.link_genes]
        new_genome.fitness = self.fitness
        new_genome.adj_fitness = self.adj_fitness
        new_genome.species_hint = self.species_hint
        return new_genome

    def calculate_compatibility(self, other):

        # Get the number of genes for each
        n_genes1 = len(self.link_genes)
        n_genes2 = len(other.link_genes)
        if n_genes1==0 and n_genes2==0:
            return 0

        # They do not use this N in their code, even though they explain it
        # this way in their paper.
        # genetics.cpp:2273
        N = min(n_genes1, n_genes2)
        N = N if max(n_genes1, n_genes2)>20 else 1

        excess = self.get_excess_genes(other)
        disjoint = self.get_disjoint_genes(other)
        m1 = self.get_matching_genes(other)
        m2 = other.get_matching_genes(self)

        # Check to make sure the ordering is right...
        for g1, g2 in zip(m1, m2):
            assert g1.innov_num == g2.innov_num

        # Average weight difference
        n_m = max(1, len(m1))
        W = sum(abs(g1.weight - g2.weight) for g1, g2 in zip(m1, m2))/n_m


        c1, c2, c3 = self.pop.c1, self.pop.c2, self.pop.c3,
        return (c1*len(excess) + c2*len(disjoint))/N + c3*W

    def mutate(self):
        # Add a node
        if random.random() < self.pop.mutate_add_node_prob:
            self.mutate_add_node()

        # Add a link
        elif random.random() < self.pop.mutate_add_link_prob:
            self.mutate_add_link()

        # Mutate or enable/disable links
        else:
            if random.random() < self.pop.mutate_link_weights_prob:
                self.mutate_link_weights()
            if random.random() < self.pop.mutate_toggle_enable_prob:
                self.mutate_toggle_enable()
            if random.random() < self.pop.mutate_toggle_reenable_prob:
                self.mutate_gene_reenable()



    def mutate_add_node(self):
        # genetics.cpp:819 Genome::mutate_add_node

        n_links = len(self.link_genes)
        if n_links == 0:
            return

        # genetics.cpp:850 - Bias splitting toward older links
        if n_links < 15:
            i_link = 0
            while random.random()<0.3:
                i_link = (i_link + 1) % (n_links)
        else:
            i_link = np.random.randint(0, n_links)

        link_gene = self.link_genes[i_link]
        from_node = link_gene.from_node
        to_node = link_gene.to_node

        # Disable the link
        link_gene.enabled = False

        # Check to see if the node is novel for this generation
        is_new = True
        for innov in self.pop.gen_innovations:
            if (innov.innovation_type == NEWNODE and
                innov.from_node == from_node and
                innov.to_node == to_node and
                innov.old_innov_num == link_gene.innov_num):
                is_new = False
                break


        if is_new:
            node_id = self.pop.get_next_node_num()
            newnode = NodeGene(node_id)
            self.pop.node_map[node_id] = newnode

            innov_num1 = self.pop.get_next_innov_num()
            innov_num2 = self.pop.get_next_innov_num()

            innov = Innovation(NEWNODE, from_node, to_node, innov_num1,
                               innov_num2, newnode_id=node_id,
                               old_innov_num=link_gene.innov_num)
            self.pop.gen_innovations.append(innov)

        else:
            newnode = self.pop.node_map[innov.newnode_id]
            innov_num1 = innov.innov_num1
            innov_num2 = innov.innov_num2


        # Link into new node gets weight 1
        link1 = LinkGene(from_node, newnode, 1.0,
                         innov_num1)

        # Link out of new node gets old weight
        link2 = LinkGene(newnode, to_node, link_gene.weight,
                         innov_num2)

        self.node_genes.append(newnode)
        self.link_genes.append(link1)
        self.link_genes.append(link2)
        if len(self.node_genes) != len(set(n.node_id for n in self.node_genes)):
            print('HERE')
            print(is_new)
            print([n.node_id for n in self.node_genes])
            print([l.innov_num for l in self.link_genes])
            print('HERE')

    def get_link_by_node_indices(self, i_node1, i_node2):
        node1 = self.node_genes[i_node1]
        node2 = self.node_genes[i_node2]
        for l in self.link_genes:
            if l.from_node == node1 and l.to_node == node2:
                return l
        return None

    def mutate_add_link(self):
        # All possible node combinations. Excludes having INPUT or BIAS as
        # to_node. Excludes all places where we already have links
        possible = [x for x in permutations(range(len(self.node_genes)), 2)
                    if (self.node_genes[x[1]].node_type not in [INPUT, BIAS])
                    and (self.get_link_by_node_indices(x[0], x[1]) is None)]

        # If all the link spots are filled
        if len(possible) == 0:
            return

        i_node1, i_node2 = random.choice(possible)
        from_node = self.node_genes[i_node1]
        to_node = self.node_genes[i_node2]

        is_new = True
        for innov in self.pop.gen_innovations:
            if (innov.innovation_type == NEWLINK and
                innov.from_node == from_node and
                innov.to_node == to_node
                ):
                is_new = False
                break

        if is_new:
            weight = random.random() * 10 * (-1)**(random.randint(0, 1))
            innov_num = self.pop.get_next_innov_num()
            innov = Innovation(NEWLINK, from_node, to_node, innov_num,
                               new_weight=weight)
            self.pop.gen_innovations.append(innov)

        else:
            weight = innov.new_weight
            innov_num = innov.innov_num1

        link = LinkGene(from_node, to_node, weight, innov_num)
        self.link_genes.append(link)


    def mutate_toggle_enable(self):
        # See gnetics.cpp:779 - must check to make sure the in-node has other
        # enabled out-node links
        # NOTE Going to try it without accounting for the above...
        gene = random.choice(self.link_genes)
        gene.enabled = not gene.enabled


    def mutate_gene_reenable(self):
        # Not sure why the naming is different from the above

        # genetics.cpp:797 - Just finds the "first" disabled gene and reenables
        # it. Not sure if the genes are sorted in any meaningful way other than
        # when they were added? "First" in this case means the first in the
        # list of genes, NOT the gene that was first disabled.
        for link in self.link_genes:
            if not link.enabled:
                link.enabled = True
                return


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


    # def get_mutation(self):
    #     new_genome = self.copy()

    def get_crossover(self, other):
        # Choose randomly when genes match up
        # Only inherit disjoin and excess genes from more fit parent
        # If they are the same fitness, use the smaller genome's disjoint and excess genes only
        # genetics.cpp 1351

        # Disabled in either parent means 75% chance of disabled in child
        pass

    def get_multipoint_crossover(self, other):

        # genetics.cpp:1354 - Between two parents, first order in higher
        # fitness, second ordering is lower gene count.
        # Looks like if the fitness and the gene count is the same, they just
        # use the second parents excess/disjoin (as if it was better). Very
        # arbitrary.

        # Make p1 the better parent
        p1, p2 = sorted([self, other], reverse=True)

        p1_matching = p1.get_matching_genes(p2)
        p2_matching = p2.get_matching_genes(p1)
        disjoint = p1.get_disjoint_genes(p2)
        excess = p1.get_excess_genes(p2)

        # Select randomly from p1 and p2
        mask = np.random.randint(2, size=len(p1_matching))
        genes = [p1_matching[i] if mask[i] else p2_matching[i]
                 for i in range(len(mask))]

        for i, g in enumerate(genes):
            if (not p1_matching[i].enabled) or (not p2_matching[i].enabled):
                if random.random() < .75: #FIXME
                    g.enabled = False
                else:
                    g.enabled = True

        genes += disjoint + excess

        child_genes = []
        child_links = set()
        for g in genes:
            in_out = (g.from_node.node_id, g.to_node.node_id)
            if in_out not in child_links:
                child_genes.append(g.copy())
                child_links.add(in_out)

        #FIXME Probably not the cleanest way to do this
        nodes = [n for n in self.pop.base_nodes]
        node_set = set(n.node_id for n in nodes)
        for l in child_genes:
            if l.from_node.node_id not in node_set:
                node_set.add(l.from_node.node_id)
                nodes.append(l.from_node)
            if l.to_node.node_id not in node_set:
                node_set.add(l.to_node.node_id)
                nodes.append(l.to_node)

        return Genome(self.pop, nodes=nodes, links=[l.copy() for l in child_genes])



    def get_matching_genes(self, other):
        """Returns the genes that are shared.

        Matching genes are genes that each of P1 and P2 have.

        Note that, like get_disjoint and get_excess, this returns P1's version
        of these genes.

        P1 - G1 G2 D3    G5    E7
        P2 - G1 G2    D4 G5 E6

        returns [G1, G2, G4]
        """
        innovs = {g.innov_num for g in other.link_genes}
        if not innovs:
            return []
        max_innov = max(innovs)
        return [g for g in self.link_genes
                if g.innov_num in innovs]

    def get_disjoint_genes(self, other):
        """Returns this genome's genes that are disjoint from the other genome.

        Disjoint genes are genes within the max innovation number of P1 that are
        not included in P1. Shown below as D3.

        P1 - G1 G2 D3    G5    E7
        P2 - G1 G2    D4 G5 E6

        returns [D3]
        """
        innovs = {g.innov_num for g in other.link_genes}
        if not innovs:
            return []
        max_innov = max(innovs)
        return [g for g in other.link_genes
                if g.innov_num < max_innov and g.innov_num not in innovs]

    def get_excess_genes(self, other):
        """Returns this genome's genes that are excess from the other genome.

        Excess genes are genes outside the max innovation number of P1. Shown
        below as E7.

        P1 - G1 G2 D3    G5    E7
        P2 - G1 G2    D4 G5 E6

        returns [E7]
        """
        innovs = {g.innov_num for g in other.link_genes}
        if not innovs:
            return []
        max_innov = max(innovs)
        return [g for g in self.link_genes
                if g.innov_num > max_innov and g.innov_num not in innovs]

    def get_network(self):
        """Returns the network representation of this genome (the phenotype)"""

        # Find which nodes are input and which are output. We may want to store
        # this info somewhere else (like in the genome)

        inputs = []
        outputs = []
        bias = []
        edges = []
        node_num = dict() #Map from node_id to zero index node number

        for i, node in enumerate(self.node_genes):
            # Create mapping
            node_num[node.node_id] = i

            # Store input and output node_numbers
            if node.node_type is INPUT:
                inputs.append(i)
            elif node.node_type is OUTPUT:
                outputs.append(i)
            elif node.node_type is BIAS:
                bias.append(i)

        # Create edge list.
        for link in self.link_genes:
            if link.enabled:
                edges.append((node_num[link.to_node.node_id],
                              node_num[link.from_node.node_id], gene.weight))


        # Build an adjacency matrix for the network
        n = len(node_num)
        adj_matrix = np.zeros((n, n))
        try:
            for e in edges:
                adj_matrix[e[:2]] = e[2]
        except:
            global GENOME
            GENOME = self
            print([node.node_id for node in self.node_genes])
            print()
            print('len(node_genes)', len(self.node_genes))
            print('edge', e)
            print('adj.shape', adj_matrix.shape)
            sys.exit()

        return Network(adj_matrix, inputs, outputs, bias)


# Potentially just a data class
class LinkGene:
    """"""
    def __init__(self, from_node, to_node, weight, innov_num, enabled=True):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.innov_num = innov_num
        self.enabled = enabled

        self.attrs = (from_node, to_node, weight, innov_num, enabled)

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
    def __init__(self, node_id, node_type=HIDDEN):
        self.node_id = node_id
        self.node_type = node_type

    def __eq__(self, other):
        return ((self.node_id   == other.node_id) and
                (self.node_type == other.node_type))



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
    def __init__(self, innovation_type, from_node, to_node,
                 innov_num1=None, innov_num2=None, new_weight=None,
                 newnode_id=None, old_innov_num=None, recursive=False):
        self.innovation_type = innovation_type
        self.from_node = from_node
        self.to_node = to_node
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



class Network:
    """"""
    def __init__(self, adj_matrix, input_nodes, output_nodes, bias_nodes):

        # Network adjacency matrix
        self.A = adj_matrix
        self.inputs = input_nodes
        self.outputs = output_nodes
        self.bias = bias_nodes

        # Activation function
        self.sigmoid = lambda x : 1/(1+np.exp(-4.924273*x))

    def activate(self, inputs, max_iters=10, verbose=False):
        """ Returns the acvtivation values of the output nodes. These are
        computed by passing a signal through the network until all output nodes are
        active. If after max_iter iterations, the output nodes remain off,
        an array of nans is returned instead.

        Additionally, we reactivate the input nodes at each time step.
        """
        # Node activation values
        self.node_vals = np.zeros(self.A.shape[0]) + .5

        # Bool to check if node was activated in the current time step
        self.active_nodes = np.zeros((self.A.shape[0]), dtype=bool)

        # Label inputs and bias as active
        self.active_nodes[self.inputs] = True
        self.active_nodes[self.bias] = True

        # While some output nodes are inactive, pass the signal farther
        # through the network

        i=0
        # while not self.active_nodes[self.outputs].all():
        while True:

            # Activate inputs
            # NOTE: This step disallows recurrent connections between hidden and input nodes

            self.node_vals[self.inputs] = inputs
            self.node_vals[self.bias] = 1.

            # Drive the activations one time step farther through the network
            self.node_vals = self.A.dot(self.node_vals)

            # Keep track of new node activations
            self.active_nodes = (self.A != 0).dot(self.active_nodes) + self.active_nodes

            # Apply sigmoid to active nodes
            self.node_vals[self.active_nodes] = self.sigmoid(self.node_vals[self.active_nodes])
            if verbose:
                print(self.node_vals)



            i += 1
            # Stop if the number of iterations exceeds max_iters
            if i > max_iters:
                break
                #
                # return np.array([np.nan]*len(self.outputs))

        return self.node_vals[self.outputs]




