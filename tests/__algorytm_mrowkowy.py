import random
import math
import time
import tracemalloc
import pandas as pd
from tqdm import tqdm
from common.data_loader import load_data

# Pliki testowe
files = ["data/280_1.txt", "data/2000_1.txt", "data/4461_1.txt"]

# Parametry ACO
evaporation_rate = 0.5
pheromone_deposit = 10
stability_runs = 5
ant_ranges = [5, 10, 20]
iteration_ranges = [10, 20, 40]
alpha_beta_pairs = [(0.1, 0.2), (1, 2)]

# Wyniki
optimal_results = []
efficiency_results = []
stability_results = {file: [] for file in files}
memory_results = []

class ACO:
    def __init__(self, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit, graph, itemset, W, Vmax, Vmin, R):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.graph = graph
        self.itemset = itemset
        self.W = W
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.R = R

    def initialize_pheromone(self, num_cities):
        return {i: {j: 1.0 for j in range(1, num_cities + 1) if i != j} for i in range(1, num_cities + 1)}

    def run(self):
        num_cities = len(self.graph)
        pheromone = self.initialize_pheromone(num_cities)
        best_route = None
        best_distance = float('inf')
        best_items = []
        best_profit = 0
        best_total_weight = 0
        best_time = float('inf')

        for iteration in tqdm(range(self.num_iterations), desc="Iteracje ACO"):
            ants = [Ant(self.alpha, self.beta, num_cities, self.graph, self.itemset, self.W) for _ in range(self.num_ants)]
            for ant in ants:
                ant.travel(pheromone)
                if ant.distance_traveled == float('inf'):
                    continue

                speed = max(self.Vmin, self.Vmax - (ant.current_weight / self.W) * (self.Vmax - self.Vmin))
                time = ant.distance_traveled / speed
                cost = self.R * time

                if ant.total_profit > best_profit:
                    best_profit = ant.total_profit
                    best_route = ant.route
                    best_distance = ant.distance_traveled
                    best_items = ant.items_picked
                    best_total_weight = ant.current_weight
                    best_time = time

            for i in pheromone:
                for j in pheromone[i]:
                    pheromone[i][j] *= (1 - self.evaporation_rate)

            for ant in ants:
                if ant.distance_traveled < float('inf') and ant.total_profit > 0:
                    for i in range(len(ant.route) - 1):
                        city1 = ant.route[i]
                        city2 = ant.route[i + 1]
                        pheromone[city1][city2] += self.pheromone_deposit / ant.distance_traveled
                        pheromone[city2][city1] += self.pheromone_deposit / ant.distance_traveled

        return best_route, best_distance, best_items, best_profit, best_total_weight, best_time


class Ant:
    def __init__(self, alpha, beta, num_cities, graph, itemset, W):
        self.alpha = alpha
        self.beta = beta
        self.num_cities = num_cities
        self.graph = graph
        self.itemset = itemset
        self.W = W
        self.route = []
        self.visited = set()
        self.items_picked = []
        self.distance_traveled = 0
        self.total_profit = 0
        self.current_weight = 0

    def select_next_city(self, current_city, pheromone):
        probabilities = []
        for dest, dist in self.graph[current_city]:
            if dest not in self.visited and dist > 0:
                tau = pheromone[current_city][dest] ** self.alpha
                eta = (1.0 / dist) ** self.beta
                probabilities.append((dest, tau * eta))

        if not probabilities:
            return None

        total = sum(prob for _, prob in probabilities)
        probabilities = [(city, prob / total) for city, prob in probabilities]
        r = random.random()
        for city, prob in probabilities:
            r -= prob
            if r <= 0:
                return city
        return probabilities[-1][0]

    def pick_items(self, current_city):
        items = sorted(self.itemset.get(current_city, []), key=lambda x: x[1] / x[2], reverse=True)
        for item_id, profit, weight in items:
            if self.current_weight + weight <= self.W:
                self.items_picked.append(item_id)
                self.total_profit += profit
                self.current_weight += weight

    def travel(self, pheromone):
        self.route = [random.randint(1, self.num_cities)]
        self.visited = set(self.route)
        self.distance_traveled = 0

        while len(self.route) < self.num_cities:
            current_city = self.route[-1]
            self.pick_items(current_city)
            next_city = self.select_next_city(current_city, pheromone)
            if next_city is None:
                self.distance_traveled = float('inf')
                return
            self.route.append(next_city)
            self.visited.add(next_city)
            self.distance_traveled += next(dist for dest, dist in self.graph[current_city] if dest == next_city)

        start_city = self.route[0]
        last_city = self.route[-1]
        self.distance_traveled += next(dist for dest, dist in self.graph[last_city] if dest == start_city)


# Testy z wykorzystaniem tqdm dla wizualizacji postępu
for file in files:
    print(f"\nPrzetwarzanie pliku: {file}")
    graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)

    for alpha, beta in alpha_beta_pairs:
        # Optymalność rozwiązania
        for num_ants in ant_ranges:
            for num_iterations in iteration_ranges:
                aco = ACO(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit, graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio)
                start_time = time.time()
                best_route, best_distance, best_items, best_profit, best_total_weight, best_time = aco.run()
                execution_time = time.time() - start_time

                optimal_results.append({
                    "Instancja problemu": file,
                    "Alpha": alpha,
                    "Beta": beta,
                    "Liczba mrówek": num_ants,
                    "Liczba iteracji": num_iterations,
                    "Całkowity zysk": best_profit,
                    "Całkowita waga": best_total_weight,
                    "Całkowity dystans": best_distance,
                    "Czas wykonania (s)": execution_time
                })

                # Efektywność czasowa
                efficiency_results.append({
                    "Instancja problemu": file,
                    "Alpha": alpha,
                    "Beta": beta,
                    "Czas wykonania (s)": execution_time,
                    "Liczba mrówek": num_ants,
                    "Liczba iteracji": num_iterations
                })

    # Stabilność wyników
    for run in range(stability_runs):
        start_time = time.time()  # Rozpocznij pomiar czasu
        aco = ACO(10, 30, alpha, beta, evaporation_rate, pheromone_deposit, graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio)
        best_route, best_distance, best_items, best_profit, best_total_weight, best_time = aco.run()
        execution_time = time.time() - start_time  # Oblicz czas wykonania

        stability_results[file].append({
            "Uruchomienie": run + 1,
            "Alpha": alpha,
            "Beta": beta,
            "Całkowity zysk": best_profit,
            "Całkowita waga": best_total_weight,
            "Całkowity dystans": best_distance,
            "Czas wykonania (s)": execution_time,  # Zamiast best_time, wykorzystujemy zmierzony czas
            "Liczba mrówek": 10,
            "Liczba iteracji": 30
        })


        # Złożoność obliczeniowa
        tracemalloc.start()
        aco = ACO(10, 30, alpha, beta, evaporation_rate, pheromone_deposit, graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio)
        aco.run()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_results.append({
            "Instancja problemu": file,
            "Alpha": alpha,
            "Beta": beta,
            "Średnie zużycie pamięci (MB)": peak_memory / 10**6,
            "Liczba mrówek": 10,
            "Liczba iteracji": 30
        })

# Zapisywanie wyników
pd.DataFrame(optimal_results).to_excel("aco_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel("aco_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    pd.DataFrame(results).to_excel(f"aco_stability_{file.split('/')[-1].split('.')[0]}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel("aco_memory_results.xlsx", index=False)
