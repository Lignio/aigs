# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv
from pcgym.envs.helper import get_int_prob, get_string_map
from tqdm import tqdm
from PIL import Image


# %% n-dimensional function with a strange topology
@partial(np.vectorize, signature="(d)->()")
def griewank_function(pop):  # this is kind of our fitness function (except we a minimizing)
    return 1 + np.sum(pop**2) / 4000 - np.prod(np.cos(pop / np.sqrt(np.arange(1, pop.size + 1))))


@partial(np.vectorize, signature="(d)->()")
def sphere_function(pop):  # sphere function instead of griewank, very simple but should demonstrate the idea.
    return np.sum(pop**2)


@partial(np.vectorize, signature="(d)->(d)", excluded=[0])
def mutate(sigma, pop):  # What are we doing here?
    return pop + np.random.normal(0, sigma, pop.shape)


@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover(x1, x2):  # TODO: think about what we are doing here. Is it smart?
    return x1 * np.random.rand() + x2 * (1 - np.random.rand())


def step(pop, cfg):
    loss = sphere_function(pop)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)]  # select best
    best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
    pop = crossover(best, best[np.random.permutation(best.shape[0])])  # mutate
    return mutate(cfg.sigma, pop), loss  # return new generation and loss


# %% Setup


#Main function for part of lab 3.
def main(cfg):
    pop = np.random.uniform(-75.0, 75.0, size=(cfg.population, 2))
    for cycle in range(cfg.generation):
        pop, loss = step (pop, cfg)
        best_idx = np.argmin(loss)
        best_specimen, best_val = pop[best_idx], loss[best_idx]
        #creates plot and display current population (blue) + fittest speciment (green)
        plt.clf()
        plt.scatter(pop[:, 0], pop[:, 1], s=10, alpha=0.4, label="population")
        plt.scatter(*best_specimen, c="green", s=40, marker="*", label="fittest")
        plt.xlim(-75, 75)
        plt.ylim(-75, 75)
        plt.title(f"generation {cycle:03d} | fittest={best_val:.6f}")
        # Pause to make progress observeable
        plt.pause(1)
    exit()


# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.n, *env._rep._map.shape))  # type: ignore
    return env, pop


# Problem specific functions:
def sample_random():
    return np.random.uniform(0, 1, (2,))


#Very simple fitness evulation, just the inverse of dist-win with a +1 to avoid dividing by 0.
def fitness(map):
    return 1/(map['dist-win']+1)

#Behavior evalution, ensures behavior is between 0-1. The +1 is to ensure no divsion by 0.
def evalBehavior(map):
    return [1/(map['enemies']+1), 1/(map['jumps']+1)] 

def evaluate(x):
    fitness = np.exp(-np.std(x))
    fitness(x)
    behavior = [np.mean(x)]
    return fitness, behavior


# MAP-Elites auxiliary functions:
def get_key(b, resolution):
    # suppose that b is in [0, 1]*
    return tuple(
        [int(x * resolution) if x < 1 else (resolution - 1) for x in b]
    )  # edge case when the behavior is exactly the bound you put it with the previous cell


def iso_line_dd(p1, p2, iso_sigma=0.01, line_sigma=0.2):
    # suppose that the search space is in [0, 1]*
    candidate = p1 + np.random.normal(0, iso_sigma) + np.random.normal(0, line_sigma) * (p2 - p1)
    return np.clip(candidate, np.zeros(p1.shape), np.ones(p1.shape))


def variation_operator(Archive):
    keys = list(Archive.keys())
    key1 = keys[np.random.randint(0, len(keys))]
    key2 = keys[np.random.randint(0, len(keys))]
    return iso_line_dd(Archive[key1]["solution"], Archive[key2]["solution"])


 # MAP-Elites hyperparameters
n_budget = 1000  # total number of evaluations
n_init = int(0.1 * n_budget)  # number of random solutions to start filling the archive
resolution = 10  # number of cells per dimension
def MapElite(env):
   
    # MAP-Elites:
    Archive = {}  # empty archive
    for i in tqdm(range(n_budget)):
        if i < n_init:  # initialize with random solutions
            candidate = MakeMap(env)
        else:  # mutation and/or crossover
            candidate = mutateMap(candidate, env)
        map = env._prob.get_stats(get_string_map(candidate, env._prob.get_tile_types()))
        f = fitness(map)
        b = evalBehavior(map)
        print("f", f)
        key = get_key(b, resolution)  # get the index of the niche/cell
        if key not in Archive or Archive[key]["fitness"] < f:  # add if new behavior or better fitness
            Archive[key] = {"fitness": f, "behavior": (map['enemies'], map['jumps']), "solution": candidate}

    return Archive

#Creates a random map.
def MakeMap(env):
    return np.random.randint(0, env.get_num_tiles(), env._rep._map.shape)

#Mutates the map by exchanging 250 tiles to something else. This could be changed to be done more dynamically (in terms of size change).
def mutateMap(map, env):
    env._rep._map = map
    newMap = map.copy()
    for _ in range(250):
        x = np.random.randint(0, map.shape[0])
        y = np.random.randint(0, map.shape[1])
        newMap[x, y] = np.random.randint(0, env.get_num_tiles())
    env._rep._map = newMap
    return newMap

#Main function for Map-Elite"
"""
def main(cfg):
    env, pop = init_pcgym(cfg)
    map = get_string_map(env._rep._map, env._prob.get_tile_types())
    behavior = env._prob.get_stats(map)
    mapElite = MapElite(env)
    best_key = min(mapElite, key=lambda k: mapElite[k]["fitness"])
    env._rep._map = mapElite[best_key]["solution"]
    print("solution", env._prob.get_stats(get_string_map(env._rep._map, env._prob.get_tile_types())))
    Image.fromarray(env.render()).save("map.png")
    exit()
"""