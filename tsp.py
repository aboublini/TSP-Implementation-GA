import numpy as np
import random

# Initialize cities
cities = 5

# Initialize city names
# A : 0
# B : 1
# C : 2
# D : 3
# E = 4
city_names = np.array(['A', 'B', 'C', 'D', 'E'])

# Initialize population
population = 10


# Function that computes the score between two cities given their ids
def compute_score(city1, city2):
    # Initialize score array (based on assigment instructions)
    scores = np.array([
        [0, 4, 4, 7, 3],
        [4, 0, 2, 3, 5],
        [4, 2, 0, 2, 3],
        [7, 3, 2, 0, 6],
        [3, 5, 3, 6, 0]
    ])

    # Return the score
    return scores[city1][city2]


# Function that creates a population set
def create_population_set(names, population_count):
    # Initialize an empty array representing the population set
    population_set = []

    for i in range(population_count):
        # Randomly selecting each solution
        city_ids = list(range(cities))
        solution = names[np.random.choice(city_ids, cities, replace=False)]

        # Adding each solution to population set
        population_set.append(solution)

    print("Set is:")
    print(np.array(population_set))
    print("")
    return np.array(population_set)


# Fitness function: returns the score of the given population set
def fitness_function(set_s):
    # Initialize the array that the score of each solution will be stored
    scores = []
    for solution in set_s:
        # Initialize a temporary array (here we'll store the numbers corresponding to each city)
        temp = []

        # Initialize total (here we'll store the score of each solution)
        total = 0

        # Calculating temp for each row
        for element in solution:
            temp.append(ord(element) - 65)

        # Calculating the score of each solution
        for i in range(5):
            if i == 4:
                total = total + compute_score(temp[4], temp[0])
            else:
                total = total + compute_score(temp[i], temp[i+1])

        # Adding the score of each solution to scores array
        scores.append(total)
    print("Fitness of set is:")
    print(scores)
    print("")
    return scores


# Evaluation function: returns a probability vector (probability of an element to be chosen as a parent)
def evaluation(fitness_scores):
    # Initialize probability list
    plist = []

    # Calculate sum of fitness_scores
    total = sum(fitness_scores)

    for element in fitness_scores:
        plist.append(element/total)

    return plist


# Renewal Elements function: Selects a random row of the generation in order to add it to the new
# generation for the population renewal
def renewal_elements(set_s):
    # Select the rows of the set
    sel = np.random.choice(list(range(len(set_s))), 2, replace=True)

    return set_s[sel]


# Initialize n: the number of parents to be the basis of the next generation
n = 8


# Parent Selection function: selects the progenitors using the roulette wheel selection
def parent_selection(set_s, fitness_scores):
    # Calculate the probability list that a solution will be chosen as a parent
    probability_list = evaluation(fitness_scores)
    print("Probability list of set is:")
    print(probability_list)
    print("")

    # Select the rows of the set
    selection = np.random.choice(list(range(len(set_s))), n, p=probability_list, replace=True)
    print("Selection of parents:")
    print(selection)
    print("")

    # Parents array: the solutions of the set that are going ot be the basis of the next generation
    parents = []
    for i in selection:
        parents.append(set_s[i])

    print("Parents:")
    for parent in parents:
        print(parent)
    print("")
    return np.array(parents)


# One point crossover: Identical to the way we tried in class
def crossover(pr):
    children = []
    print("Children:")
    for i in range(0, n, 2):
        # Child 1
        # Keep the first part of the first parent
        child1_part1 = pr[i][0:3]
        parent2 = pr[i+1]
        # Delete elements of first parent from second parent
        for value in parent2:
            if value in child1_part1:
                parent2 = np.delete(parent2, np.where(parent2 == value))

        # Concatenate first part of parent1 with remains of parent2
        child1 = np.concatenate((child1_part1, parent2))

        # Child2
        # Keep the first part of the second parent
        child2_part1 = pr[i+1][0:3]
        parent1 = pr[i]
        # Delete elements of second parent from first parent
        for value in parent1:
            if value in child2_part1:
                parent1 = np.delete(parent1, np.where(parent1 == value))

        # Concatenate first part of parent1 with remains of parent2
        child2 = np.concatenate((child2_part1, parent1))

        # Add children to children list
        children.append(child1)
        children.append(child2)

    print(np.array(children))
    print("")
    return np.array(children)


# Population Renewal Function: Adds new population to current generation
def population_renewal(current_gen, extra_members):
    current_gen = np.concatenate((current_gen, extra_members))
    return current_gen


# Initialize mutation rate
mutation_rate = 0.1


# Mutate child function: mutates a single offspring
def mutate_child(child):
    for y in range(int(cities * mutation_rate)):
        # Index of first element to be swiped
        index_a = np.random.randint(0, cities)

        # Index of second element to be swiped
        index_b = np.random.randint(0, cities)

        # Swipe
        temp = child[index_a]
        child[index_a] = child[index_b]
        child[index_b] = temp

    return child


# Mutation function: for each child of the new population call the mutate_child() method
def mutation(offsprings):
    mutated_pop = []

    for child in offsprings:
        mutated_pop.append(mutate_child(child))

    return mutated_pop


# Initialize score table
score_table = []

best_set = []
best_fitness = 0
min_distance = 300
best = 0

# Genetic algorithm loop for 20 generations
for i in range(20):
    print("---------------------------------------------------------------------------------")
    print("Generation " + str(i))
    if i == 0:
        # First generation set
        gen = create_population_set(city_names, population)
    else:
        # Generation set
        print("Set is:")
        print(gen)
        print("")

    # Fitness generation set
    distances = fitness_function(gen)

    # Adding the total score of generation to score table
    score_table.append(sum(distances))

    # Store best set
    total_distance = sum(distances)
    if total_distance < min_distance:
        # Minimum distance: best distance
        min_distance = total_distance
        # Best set
        best_set = gen
        # Best fitness
        best_fitness = distances
        # Best Generation
        best = i

    # Select parents
    par = parent_selection(gen, distances)

    # Population that will be added to new generation without being part of reproduction
    extra_population = renewal_elements(gen)

    # Single point crossover
    gen = crossover(par)

    # Population renewal
    gen = population_renewal(gen, extra_population)

    # Mutation
    mutation(gen)

    print("---------------------------------------------------------------------------------\n")

# Print total score for each generation
print("Fitness of each generation:")
for i in range(20):
    print("Generation " + str(i) + ": " + str(score_table[i]))
print("---------------------------------------------------------------------------------")

# Print best solution and general information
print("Generation with best score: " + str(best))
print("Score: " + str(min_distance))

min_index = np.argmin(best_fitness)
min_element = np.amin(best_fitness)

print("Best Path: " + str(best_set[min_index]))
print("Distance of best path: " + str(min_element))
