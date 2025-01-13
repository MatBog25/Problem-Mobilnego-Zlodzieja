import random
import math
import time
import tracemalloc
import pandas as pd
from tqdm import tqdm
from common.data_loader import load_data

# Pliki testowe
files = ["data/280_1.txt", "data/2000_1.txt", "data/4461_1.txt"]

# Parametry algorytmu PSO
particle_ranges = [10, 20, 50]  # Liczba cząstek
iteration_ranges = [10, 20, 50]  # Liczba iteracji
stability_runs = 5  # Liczba uruchomień dla stabilności
w = 0.7  # Waga bezwładności
c1 = 2.0  # Składnik poznawczy
c2 = 2.0  # Składnik społeczny

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

# Klasa PSO
class Particle:
    def __init__(self, graph, itemset, W, Vmax, Vmin, R, route):
        self.graph = graph
        self.itemset = itemset
        self.W = W
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.R = R
        self.route = self.ensure_all_cities(route)
        self.best_route = list(self.route)
        self.best_fitness, self.picked_items, self.total_weight, self.total_profit, self.final_speed = self.calculate_fitness()
        self.velocity = []

    def ensure_all_cities(self, route):
        all_cities = set(self.graph.keys())
        missing_cities = list(all_cities - set(route))
        route += missing_cities
        random.shuffle(route)
        return route

    def calculate_fitness(self):
        total_distance = calculate_total_distance(self.route, self.graph)
        total_profit = 0
        total_weight = 0
        picked_items = []

        for city in self.route:
            for item in self.itemset.get(city, []):
                item_id, profit, weight = item
                if total_weight + weight <= self.W:
                    picked_items.append((city, item_id))
                    total_weight += weight
                    total_profit += profit

        final_speed = self.Vmax - (total_weight / self.W) * (self.Vmax - self.Vmin)
        fitness = total_profit
        return fitness, picked_items, total_weight, total_profit, final_speed

    def update_velocity(self, global_best_route, w, c1, c2):
        self.velocity = []
        for i in range(len(self.route)):
            r1 = random.random()
            r2 = random.random()
            if r1 < c1 and self.route[i] in self.best_route:
                self.velocity.append(('swap', i, self.best_route.index(self.route[i])))
            if r2 < c2 and self.route[i] in global_best_route:
                self.velocity.append(('swap', i, global_best_route.index(self.route[i])))

    def update_position(self):
        new_route = list(self.route)
        for move in self.velocity:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                if i < len(new_route) and j < len(new_route):
                    new_route[i], new_route[j] = new_route[j], new_route[i]
        self.route = self.ensure_all_cities(new_route)
        new_fitness, new_picked_items, new_total_weight, new_total_profit, new_final_speed = self.calculate_fitness()
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_route = list(self.route)
            self.picked_items = new_picked_items
            self.total_weight = new_total_weight
            self.total_profit = new_total_profit
            self.final_speed = new_final_speed

class PSO:
    def __init__(self, graph, itemset, W, Vmax, Vmin, R, num_particles, w, c1, c2, num_iterations):
        self.graph = graph
        self.itemset = itemset
        self.W = W
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.R = R
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_iterations = num_iterations
        self.particles = [Particle(graph, itemset, W, Vmax, Vmin, R, self.generate_random_route()) for _ in range(num_particles)]

        self.global_best_route = self.particles[0].best_route
        self.global_best_fitness = self.particles[0].best_fitness
        self.global_best_items = self.particles[0].picked_items
        self.global_best_weight = self.particles[0].total_weight
        self.global_best_profit = self.particles[0].total_profit
        self.global_best_speed = self.particles[0].final_speed

    def generate_random_route(self):
        cities = list(self.graph.keys())
        random.shuffle(cities)
        return cities

    def run(self):
        for _ in tqdm(range(self.num_iterations), desc="Iteracje PSO"):
            for particle in self.particles:
                particle.update_velocity(self.global_best_route, self.w, self.c1, self.c2)
                particle.update_position()
                if particle.best_fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_route = particle.best_route
                    self.global_best_items = particle.picked_items
                    self.global_best_weight = particle.total_weight
                    self.global_best_profit = particle.total_profit
                    self.global_best_speed = particle.final_speed

        total_distance = calculate_total_distance(self.global_best_route, self.graph)
        return (self.global_best_route, self.global_best_fitness, self.global_best_items,
                self.global_best_weight, self.global_best_profit, self.global_best_speed, total_distance)

# Wyniki
optimal_results = []
efficiency_results = []
stability_results = {file: [] for file in files}
memory_results = []

for file in files:
    print(f"\nPrzetwarzanie pliku: {file}")
    graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)

    # Optymalność rozwiązania
    for num_particles in particle_ranges:
        for num_iterations in iteration_ranges:
            pso = PSO(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, num_particles, w, c1, c2, num_iterations)
            start_time = time.time()
            best_route, best_fitness, best_items, best_weight, best_profit, best_speed, total_distance = pso.run()
            execution_time = time.time() - start_time

            optimal_results.append({
                "Instancja problemu": file,
                "Liczba cząstek": num_particles,
                "Liczba iteracji": num_iterations,
                "Całkowity zysk": best_profit,
                "Całkowita waga": best_weight,
                "Długość trasy": total_distance,
                "Czas wykonania (s)": execution_time
            })

    # Efektywność czasowa
    for num_particles in particle_ranges:
        for num_iterations in iteration_ranges:
            pso = PSO(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, num_particles, w, c1, c2, num_iterations)
            start_time = time.time()
            _, _, _, _, _, _, _ = pso.run()
            execution_time = time.time() - start_time

            efficiency_results.append({
                "Instancja problemu": file,
                "Czas wykonania (s)": execution_time,
                "Algorytm": "PSO",
                "Parametry": f"Cząstki: {num_particles}, Iteracje: {num_iterations}"
            })

    # Stabilność wyników
    for run in range(stability_runs):
        pso = PSO(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, 10, w, c1, c2, 50)
        best_route, best_fitness, best_items, best_weight, best_profit, best_speed, total_distance = pso.run()
        stability_results[file].append({
            "Uruchomienie": run + 1,
            "Całkowity zysk": best_profit,
            "Całkowita waga": best_weight,
            "Długość trasy": total_distance,
            "Czas wykonania (s)": execution_time,
            "Parametry": "Cząstki: 10, Iteracje: 50"
        })

    # Złożoność obliczeniowa
    tracemalloc.start()
    pso = PSO(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, 10, w, c1, c2, 50)
    pso.run()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_results.append({
        "Instancja problemu": file,
        "Średnie zużycie pamięci (MB)": peak_memory / 10**6,
        "Parametry": "Cząstki: 10, Iteracje: 50"
    })

# Zapisywanie wyników
pd.DataFrame(optimal_results).to_excel("pso_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel("pso_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    pd.DataFrame(results).to_excel(f"pso_stability_{file.split('/')[-1].split('.')[0]}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel("pso_memory_results.xlsx", index=False)
