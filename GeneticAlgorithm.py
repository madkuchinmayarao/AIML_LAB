

import numpy as np

# HELLO Problem Parameters
target_string = "HELLO"
string_length = len(target_string)
population_size = 10000
generations = 10

# Initialize Population
population = np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "), size=(population_size, string_length))

def fitness(chromosome):
    return np.sum(chromosome == np.array(list(target_string)))

# Genetic Algorithm Loop
while True:
    # Genetic Algorithm
    for generation in range(generations):
        # Evaluation
        fitness_values = np.array([fitness(chromosome) for chromosome in population])

        # Selection
        selected_indices = np.argsort(fitness_values)[-population_size // 2:]
        parents = population[selected_indices]

        # Crossover
        crossover_point = string_length // 2
        offspring = np.concatenate([parents[:, :crossover_point], parents[:, crossover_point:]], axis=1)

        # Mutation
        mutation_rate = 0.1
        mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
        offspring[mutation_mask] = np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "), size=np.sum(mutation_mask))

        # Replacement
        population[:population_size // 2] = offspring

    # Final Evaluation
    final_fitness_values = np.array([fitness(chromosome) for chromosome in population])
    best_solution_index = np.argmax(final_fitness_values)
    best_solution = ''.join(population[best_solution_index])

    print("Best Solution:", best_solution)
    print("Fitness:", np.max(final_fitness_values))

    # Check if the best solution matches the target
    if best_solution == target_string:
        break
