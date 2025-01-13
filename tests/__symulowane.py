import random
import math
import time
import tracemalloc
import pandas as pd
from tqdm import tqdm
from common.data_loader import load_data

# Pliki testowe
files = ["data/280_1.txt", "data/2000_1.txt", "data/4461_1.txt"]

# Parametry algorytmu
temperature_ranges = [100000, 1000000, 10000000]
cooling_rates = [0.90, 0.95, 0.99]
stopping_iterations = [10, 50, 100]
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

# Klasa Symulowanego Wyżarzania
class SimulatedAnnealing:
    def __init__(self, graph, itemset, W, Vmax, Vmin, R, initial_route, temp, alpha, stopping_temp, stopping_iter):
        self.graph = graph
        self.itemset = itemset
        self.W = W
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.R = R
        self.current_route = initial_route
        self.best_route = list(initial_route)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.current_fitness, self.current_items, self.current_weight, self.current_profit, self.current_speed = self.calculate_fitness(self.current_route)
        self.best_fitness = self.current_fitness
        self.best_items = self.current_items
        self.best_weight = self.current_weight
        self.best_profit = self.current_profit
        self.best_speed = self.current_speed

    def calculate_fitness(self, route):
        total_distance = calculate_total_distance(route, self.graph)
        total_profit = 0
        total_weight = 0
        picked_items = []

        for city in route:
            for item in self.itemset.get(city, []):
                item_id, profit, weight = item
                if total_weight + weight <= self.W:
                    picked_items.append((city, item_id))
                    total_weight += weight
                    total_profit += profit

        speed = self.Vmax - (total_weight / self.W) * (self.Vmax - self.Vmin)
        fitness = total_profit
        return fitness, picked_items, total_weight, total_profit, speed

    def anneal(self):
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = list(self.current_route)
            random.shuffle(candidate)
            candidate_fitness, _, _, _, _ = self.calculate_fitness(candidate)
            if candidate_fitness > self.current_fitness:
                self.current_route = candidate
                self.current_fitness = candidate_fitness
            self.temp *= self.alpha
            self.iteration += 1

        total_distance = calculate_total_distance(self.best_route, self.graph)
        return self.best_route, self.best_fitness, self.best_items, self.best_weight, self.best_profit, total_distance

# Wyniki
optimal_results = []
efficiency_results = []
stability_results = {file: [] for file in files}
memory_results = []

for file in files:
    print(f"\nPrzetwarzanie pliku: {file}")
    graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)

    # Optymalność rozwiązania
    for initial_temp in temperature_ranges:
        for cooling_rate in cooling_rates:
            for stopping_iter in stopping_iterations:
                initial_route = list(graph.keys())
                random.shuffle(initial_route)
                sa = SimulatedAnnealing(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, initial_route, initial_temp, cooling_rate, 1, stopping_iter)
                start_time = time.time()
                best_route, best_fitness, best_items, best_weight, best_profit, total_distance = sa.anneal()
                execution_time = time.time() - start_time

                optimal_results.append({
                    "Instancja problemu": file,
                    "Temperatura początkowa": initial_temp,
                    "Współczynnik chłodzenia": cooling_rate,
                    "Liczba iteracji": stopping_iter,
                    "Całkowity zysk": best_profit,
                    "Całkowita waga": best_weight,
                    "Długość trasy": total_distance,
                    "Czas wykonania (s)": execution_time
                })

    # Efektywność czasowa
    for initial_temp in temperature_ranges:
        for cooling_rate in cooling_rates:
            for stopping_iter in stopping_iterations:
                initial_route = list(graph.keys())
                random.shuffle(initial_route)
                sa = SimulatedAnnealing(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, initial_route, initial_temp, cooling_rate, 1, stopping_iter)
                start_time = time.time()
                sa.anneal()
                execution_time = time.time() - start_time

                efficiency_results.append({
                    "Instancja problemu": file,
                    "Czas wykonania (s)": execution_time,
                    "Algorytm": "Symulowane Wyżarzanie",
                    "Parametry": f"Temp: {initial_temp}, Cooling: {cooling_rate}, Iter: {stopping_iter}"
                })

    # Stabilność wyników
    for run in range(stability_runs):
        initial_route = list(graph.keys())
        random.shuffle(initial_route)
        sa = SimulatedAnnealing(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, initial_route, 10000, 0.95, 1, 50)
        start_time = time.time()
        best_route, best_fitness, best_items, best_weight, best_profit, total_distance = sa.anneal()
        execution_time = time.time() - start_time
        stability_results[file].append({
            "Uruchomienie": run + 1,
            "Całkowity zysk": best_profit,
            "Całkowita waga": best_weight,
            "Długość trasy": total_distance,
            "Czas wykonania (s)": execution_time,
            "Parametry": "Temp: 10000, Cooling: 0.95, Iter: 50"
        })

    # Złożoność obliczeniowa
    tracemalloc.start()
    initial_route = list(graph.keys())
    random.shuffle(initial_route)
    sa = SimulatedAnnealing(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, initial_route, 10000, 0.95, 1, 50)
    sa.anneal()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_results.append({
        "Instancja problemu": file,
        "Średnie zużycie pamięci (MB)": peak_memory / 10**6,
        "Parametry": "Temp: 10000, Cooling: 0.95, Iter: 50"
    })

# Zapisywanie wyników
pd.DataFrame(optimal_results).to_excel("sa_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel("sa_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    pd.DataFrame(results).to_excel(f"sa_stability_{file.split('/')[-1].split('.')[0]}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel("sa_memory_results.xlsx", index=False)
