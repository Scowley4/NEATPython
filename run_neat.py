from neat import *
from fitnessFunc import *

import sys
del sys.modules['neat']
del sys.modules['fitnessFunc']
from neat import *
from fitnessFunc import *

def get_genomes_by_node_number(pop, num):
    return [g for g in pop.all_genomes if g.has_node_num(num)]

def check_random_compatibility(pop, num):
    comp = []
    for i in range(num):
        g1 = random.choice(pop.all_genomes)
        g2 = random.choice(pop.all_genomes)
        comp.append(g1.calculate_compatibility(g2))
    return comp


pop = Population()
# pop.spawn_initial_population(2, 1)
# while True:
#    pop.next_epoch(fit_xor)

n=2
fit_nparity = get_determined_fit_dparity(n)
pop.spawn_initial_population(n, 1)
while True:
    pop.next_epoch(fit_nparity)

# n=5
# fit_nparity = get_fit_dparity(n)
# pop.spawn_initial_population(n, 1)
# while True:
#    pop.next_epoch(fit_nparity)

# pop.spawn_initial_population(4, 1)
# while True:
#     pop.next_epoch(fit_pole_balance)

# pop.spawn_initial_population(10, 1)
# while True:
#     pop.next_epoch(fit_flappy)


