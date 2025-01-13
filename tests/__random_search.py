import random
import time
import tracemalloc
import pandas as pd
from tqdm import tqdm
from common.data_loader import load_data

# Pliki testowe (instancje problemu)
files = ["data/280_1.txt", "data/2000_1.txt", "data/4461_1.txt"]

# Parametry algorytmu
iteration_ranges = [100, 200, 500, 1000]  # Liczba iteracji
stability_runs = 5  # Liczba uruchomień dla stabilności

# Funkcje pomocnicze
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

def solve_knapsack(route, itemset, W):
    picked_items = []
    total_profit = 0
    total_weight = 0

    for city in route:
        for item in itemset.get(city, []):
            item_id, profit, weight = item
            if total_weight + weight <= W:
                picked_items.append((city, item_id))
                total_weight += weight
                total_profit += profit

    return picked_items, total_profit, total_weight

# Random Search klasa
class RandomSearch:
    def __init__(self, graph, itemset, W, iterations):
        self.graph = graph
        self.itemset = itemset
        self.W = W
        self.iterations = iterations

    def generate_random_route(self):
        cities = list(self.graph.keys())
        random.shuffle(cities)
        return cities

    def run(self):
        best_route = self.generate_random_route()
        best_picked_items, best_total_profit, best_total_weight = solve_knapsack(best_route, self.itemset, self.W)
        best_distance = calculate_total_distance(best_route, self.graph)

        for _ in tqdm(range(self.iterations), desc="Random Search Iteracje"):
            new_route = self.generate_random_route()
            picked_items, total_profit, total_weight = solve_knapsack(new_route, self.itemset, self.W)
            total_distance = calculate_total_distance(new_route, self.graph)

            if total_profit > best_total_profit:
                best_route = new_route
                best_picked_items = picked_items
                best_total_profit = total_profit
                best_total_weight = total_weight
                best_distance = total_distance

        return best_route, best_distance, best_picked_items, best_total_profit, best_total_weight

# Wyniki
optimal_results = []
efficiency_results = []
stability_results = {file: [] for file in files}
memory_results = []

for file in files:
    print(f"\nPrzetwarzanie pliku: {file}")
    graph, itemset, knapsack_capacity, _, _, _ = load_data(file)

    # Optymalność rozwiązania
    for iterations in iteration_ranges:
        rs = RandomSearch(graph, itemset, knapsack_capacity, iterations)
        start_time = time.time()
        _, total_distance, _, total_profit, total_weight = rs.run()
        execution_time = time.time() - start_time
        optimal_results.append({
            "Instancja problemu": file,
            "Liczba iteracji": iterations,
            "Całkowity zysk": total_profit,
            "Całkowity dystans": total_distance,
            "Czas wykonania (s)": execution_time,
            "Algorytm": "Random Search"
        })

        # Efektywność czasowa
        efficiency_results.append({
            "Instancja problemu": file,
            "Czas wykonania (s)": execution_time,
            "Algorytm": "Random Search",
            "Parametry": f"Liczba iteracji: {iterations}"
        })

    # Stabilność wyników
    for run in range(stability_runs):
        rs = RandomSearch(graph, itemset, knapsack_capacity, 100)  # Stała liczba iteracji dla stabilności
        _, total_distance, _, total_profit, total_weight = rs.run()
        stability_results[file].append({
            "Uruchomienie": run + 1,
            "Całkowity zysk": total_profit,
            "Całkowity dystans": total_distance,
            "Czas wykonania (s)": execution_time,
            "Waga przedmiotów": total_weight,
            "Parametry": "Liczba iteracji: 100"
        })

    # Złożoność obliczeniowa (średnie zużycie pamięci)
    tracemalloc.start()
    rs = RandomSearch(graph, itemset, knapsack_capacity, 100)
    rs.run()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_results.append({
        "Instancja problemu": file,
        "Średnie zużycie pamięci (MB)": peak_memory / 10**6,
        "Parametry": "Liczba iteracji: 100"
    })

# Zapisywanie wyników
# Optymalność rozwiązania
optimal_df = pd.DataFrame(optimal_results)
optimal_df.to_excel("random_search_optimal_results.xlsx", index=False)

# Efektywność czasowa
efficiency_df = pd.DataFrame(efficiency_results)
efficiency_df.to_excel("random_search_efficiency_results.xlsx", index=False)

# Stabilność wyników
for file, results in stability_results.items():
    stability_df = pd.DataFrame(results)
    stability_df.to_excel(f"random_search_stability_{file.split('/')[-1].split('.')[0]}.xlsx", index=False)

# Złożoność obliczeniowa
memory_df = pd.DataFrame(memory_results)
memory_df.to_excel("random_search_memory_results.xlsx", index=False)
