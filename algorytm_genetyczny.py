import random
import math
from data_loader import load_data  # Wcześniej zaimplementowana funkcja

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("280_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

def fitness(route, items):
    distance = calculate_total_distance(route)
    total_weight = sum(itemset[item][1] for item in items if item != -1)
    total_value = sum(itemset[item][0] for item in items if item != -1)
    time = distance / ((Vmax + Vmin) / 2)

    if total_weight > W:
        return 0
    return total_value - R * time

def solve_knapsack(route):
    """Rozwiązuje problem plecakowy dla podanej trasy."""
    distances = [0] * len(route)
    for i in range(1, len(route)):
        for dest, dist in graph[route[i - 1]]:
            if dest == route[i]:
                distances[i] = distances[i - 1] + dist
                break

    if distances[-1] == 0:
        distances[-1] = 1  # Zapobieganie dzieleniu przez zero

    finalitemset = []
    time = distances[-1] * 2 * (Vmax + Vmin)
    
    for key, value in itemset.items():
        for item in value:
            item_id, profit, weight = item
            for city in route:
                if city == item_id:  # Sprawdzamy, czy miasto jest na trasie
                    route_index = route.index(city)
                    if distances[route_index] == 0:
                        distances[route_index] = 1  # Zapobieganie dzieleniu przez zero
                    score = int(profit - (0.25 * profit * (distances[route_index] / distances[-1])) - (R * time * weight / W))
                    finalitemset.append([item_id, city, weight, score, profit])

    finalitemset.sort(key=lambda x: int(x[3]), reverse=True)

    wc = 0
    picked_items = []
    totalprof = 0

    for item in finalitemset:
        if wc + item[2] <= W:
            picked_items.append(item)
            wc += item[2]
            totalprof += item[4]

    result = {}
    for item in picked_items:
        if item[1] not in result:
            result[item[1]] = []
        result[item[1]].append(item[0])

    fin = [[city, sorted(items)] for city, items in result.items()]
    fin.sort(key=lambda x: x[0])

    return fin, totalprof, wc

def calculate_total_distance(route):
    """Oblicza całkowitą odległość dla podanej trasy."""
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    # Dodaj odległość powrotu do miasta startowego
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    return total_distance

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, num_generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.population = []

    def initialize_population(self):
        """Inicjalizuje populację losowymi permutacjami miast."""
        cities = list(graph.keys())
        for _ in range(self.population_size):
            route = cities[:]
            random.shuffle(route)
            self.population.append(route)

    def select_parent(self):
        """Turniejowy wybór rodziców."""
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: fitness(x, []), reverse=True)
        return tournament[0]

    def crossover(self, parent1, parent2):
        """Krzyżowanie tras z zachowaniem unikalności miast."""
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]
        parent2_index = 0
        for i in range(len(child)):
            if child[i] == -1:
                while parent2[parent2_index] in child:
                    parent2_index += 1
                child[i] = parent2[parent2_index]
        return child

    def mutate(self, route):
        """Mutacja trasy."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]

    def evolve(self):
        """Ewoluuje populację."""
        new_population = []
        for _ in range(self.population_size):
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1[:]
            self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def run(self):
        """Uruchamia algorytm genetyczny."""
        self.initialize_population()
        for _ in range(self.num_generations):
            self.evolve()
        best_route = max(self.population, key=lambda x: fitness(x, []))
        return best_route, fitness(best_route, [])

def print_solution(route, total_distance, picked_items, total_profit, total_weight):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Złodziej powinien zabrać następujące przedmioty:")
    for i in picked_items:
        print("Miasto : " + str(i[0]) + "   Przedmioty : " + ', '.join(str(e) for e in i[1]))
    print("Całkowity zysk : " + str(total_profit))
    print("Waga przenoszona w plecaku : " + str(total_weight))

# Parametry algorytmu genetycznego
population_size = 100
mutation_rate = 0.01
crossover_rate = 0.9
num_generations = 200

# Testowanie algorytmu genetycznego
print("Uruchamianie algorytmu genetycznego...")
ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, num_generations)
best_route_ga, best_fitness_ga = ga.run()
total_distance_ga = calculate_total_distance(best_route_ga)
picked_items_ga, total_profit_ga, total_weight_ga = solve_knapsack(best_route_ga)
print_solution(best_route_ga, total_distance_ga, picked_items_ga, total_profit_ga, total_weight_ga)
