import random
import math
import time
import tracemalloc
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from common.data_loader import load_data

# Pliki testowe
files = ["data/280_1.txt", "data/2000_1.txt", "data/4461_1.txt"]

# Parametry algorytmu genetycznego
population_ranges = [10, 20, 40]
generation_ranges = [5, 10, 25]
mutation_rate = 0.01
crossover_rate = 0.9
stability_runs = 5

# Funkcja pomocnicza
def calculate_total_distance(route, graph):
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    return total_distance

# Klasa algorytmu genetycznego
class GeneticAlgorithm:
    def __init__(self, graph, itemset, W, Vmax, Vmin, R, population_size, mutation_rate, crossover_rate, num_generations):
        self.graph = graph
        self.itemset = itemset
        self.W = W
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.R = R
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.population = []

    def initialize_population(self):
        cities = list(self.graph.keys())
        for _ in range(self.population_size):
            route = cities[:]
            random.shuffle(route)
            items = [random.choice([0, 1]) for _ in range(len(self.itemset))]
            self.population.append((route, items))

    def calculate_value_and_weight(self, chromosome):
        route, items = chromosome
        total_profit = 0
        current_weight = 0

        for city in route:
            for item in self.itemset.get(city, []):
                item_id, profit, weight = item
                if items[item_id - 1]:
                    if current_weight + weight <= self.W:
                        current_weight += weight
                        total_profit += profit

        return total_profit, current_weight

    def fitness(self, chromosome):
        route, items = chromosome
        total_distance = calculate_total_distance(route, self.graph)
        total_profit, current_weight = self.calculate_value_and_weight(chromosome)
        speed = self.Vmax - (self.W / self.W) * (self.Vmax - self.Vmin)
        time = total_distance / speed
        return total_profit - self.R * time

    def select_parent(self):
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: self.fitness(x), reverse=True)
        return tournament[0]

    def crossover(self, parent1, parent2):
        route1, items1 = parent1
        route2, items2 = parent2

        start, end = sorted(random.sample(range(len(route1)), 2))
        child_route = [-1] * len(route1)
        child_route[start:end] = route1[start:end]
        parent2_index = 0
        for i in range(len(child_route)):
            if child_route[i] == -1:
                while route2[parent2_index] in child_route:
                    parent2_index += 1
                child_route[i] = route2[parent2_index]

        child_items = [random.choice([i1, i2]) for i1, i2 in zip(items1, items2)]
        return (child_route, child_items)

    def mutate(self, chromosome):
        route, items = chromosome
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(items) - 1)
            items[idx] = 1 - items[idx]

    def evolve(self):
        new_population = []
        for _ in range(self.population_size):
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1
            self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def run(self):
        self.initialize_population()
        for _ in tqdm(range(self.num_generations), desc="Generacje"):
            self.evolve()
        best_solution = max(self.population, key=lambda x: self.fitness(x))
        return best_solution

# Wyniki
optimal_results = []
efficiency_results = []
stability_results = {file: [] for file in files}
memory_results = []

for file in files:
    print(f"\nPrzetwarzanie pliku: {file}")
    graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)

    # Optymalność rozwiązania
    for population_size in population_ranges:
        for num_generations in generation_ranges:
            ga = GeneticAlgorithm(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, population_size, mutation_rate, crossover_rate, num_generations)
            start_time = time.time()
            best_solution = ga.run()
            execution_time = time.time() - start_time
            best_route, best_items = best_solution
            total_distance = calculate_total_distance(best_route, graph)
            total_value, total_weight = ga.calculate_value_and_weight(best_solution)

            optimal_results.append({
                "Instancja problemu": file,
                "Liczba populacji": population_size,
                "Liczba generacji": num_generations,
                "Całkowity zysk": total_value,
                "Całkowita waga": total_weight,
                "Całkowity dystans": total_distance,
                "Czas wykonania (s)": execution_time
            })

            # Efektywność czasowa
            efficiency_results.append({
                "Instancja problemu": file,
                "Czas wykonania (s)": execution_time,
                "Liczba populacji": population_size,
                "Liczba generacji": num_generations
            })

    # Stabilność wyników
    for run in range(stability_runs):
        ga = GeneticAlgorithm(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, 30, mutation_rate, crossover_rate, 30)
        best_solution = ga.run()
        best_route, best_items = best_solution
        total_distance = calculate_total_distance(best_route, graph)
        total_value, total_weight = ga.calculate_value_and_weight(best_solution)

        stability_results[file].append({
            "Uruchomienie": run + 1,
            "Całkowity zysk": total_value,
            "Całkowita waga": total_weight,
            "Całkowity dystans": total_distance,
            "Czas wykonania (s)": execution_time,
            "Liczba populacji": 30,
            "Liczba generacji": 30
        })

    # Złożoność obliczeniowa
    tracemalloc.start()
    ga = GeneticAlgorithm(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, 30, mutation_rate, crossover_rate, 30)
    ga.run()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_results.append({
        "Instancja problemu": file,
        "Średnie zużycie pamięci (MB)": peak_memory / 10**6,
        "Liczba populacji": 30,
        "Liczba generacji": 30
    })

# Zapisywanie wyników
pd.DataFrame(optimal_results).to_excel("genetic_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel("genetic_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    pd.DataFrame(results).to_excel(f"genetic_stability_{file.split('/')[-1].split('.')[0]}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel("genetic_memory_results.xlsx", index=False)
