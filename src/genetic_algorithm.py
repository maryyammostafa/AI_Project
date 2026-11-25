import numpy as np


def create_chromosome(num_features, rng):

    chromosome = rng.integers(0, 2, num_features)

    if np.sum(chromosome) == 0:
        random_index = rng.integers(0, num_features)
        chromosome[random_index] = 1

    return chromosome


def create_population(population_size, num_features, rng):

    population = []

    for i in range(population_size):
        chromosome = create_chromosome(num_features, rng)
        population.append(chromosome)

    return population


def get_selected_features(chromosome, feature_names):

    selected_features = []

    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            selected_features.append(feature_names[i])

    return selected_features
