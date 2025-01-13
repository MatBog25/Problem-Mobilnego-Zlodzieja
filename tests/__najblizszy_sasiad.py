import random
import time
import tracemalloc
import pandas as pd
from common.data_loader import load_data

# Pliki testowe
files = ["data/280_1.txt", "data/2000_1.txt", "data/4461_1.txt"]

# Funkcje pomocnicze
def calculate_time_and_cost(distance, total_weight, W, Vmax, Vmin, R):
    """Oblicz czas podróży."""
    speed = Vmax - (total_weight / W) * (Vmax - Vmin)
    time = distance / speed
    return time

def solve_knapsack(route, itemset, W):
    """Rozwiąż problem plecakowy dla danej trasy."""
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

class NearestNeighbor:
    def __init__(self, graph):
        self.graph = graph
        self.route = []
        self.visited = set()
        self.total_distance = 0

    def find_route(self, start):
        """Znajdź trasę najkrótszego sąsiada."""
        current_city = start
        self.route.append(current_city)
        self.visited.add(current_city)

        while len(self.visited) < len(self.graph):
            next_city = None
            shortest_distance = float("inf")
            for dest, dist in self.graph[current_city]:
                if dest not in self.visited and dist < shortest_distance:
                    next_city = dest
                    shortest_distance = dist
            if next_city is None:
                break
            self.route.append(next_city)
            self.visited.add(next_city)
            self.total_distance += shortest_distance
            current_city = next_city

        # Dodaj powrót do miasta początkowego
        if len(self.route) == len(self.graph):
            start_city = self.route[0]
            last_city = self.route[-1]
            for dest, dist in self.graph[last_city]:
                if dest == start_city:
                    self.route.append(start_city)
                    self.total_distance += dist
                    break

    def run(self):
        """Uruchom algorytm Najbliższego Sąsiada."""
        start = random.choice(list(self.graph.keys()))
        self.find_route(start)
        return self.route, self.total_distance

# Wyniki
optimal_results = []
efficiency_results = []
stability_results = {file: [] for file in files}
memory_results = []

for file in files:
    print(f"\nPrzetwarzanie pliku: {file}")
    graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)

    # Optymalność rozwiązania
    nn = NearestNeighbor(graph)
    start_time = time.time()
    best_route, total_distance = nn.run()
    picked_items, total_profit, total_weight = solve_knapsack(best_route, itemset, knapsack_capacity)
    execution_time = time.time() - start_time

    optimal_results.append({
        "Instancja problemu": file,
        "Całkowity zysk": total_profit,
        "Całkowity dystans": total_distance,
        "Czas wykonania (s)": execution_time,
        "Całkowita waga": total_weight
    })

    # Efektywność czasowa
    efficiency_results.append({
        "Instancja problemu": file,
        "Czas wykonania (s)": execution_time,
        "Algorytm": "Najbliższy Sąsiad"
    })

    # Stabilność wyników
    for run in range(5):  # Stabilność na 5 uruchomieniach
        nn = NearestNeighbor(graph)
        start_time = time.time()
        best_route, total_distance = nn.run()
        picked_items, total_profit, total_weight = solve_knapsack(best_route, itemset, knapsack_capacity)
        execution_time = time.time() - start_time

        stability_results[file].append({
            "Uruchomienie": run + 1,
            "Całkowity zysk": total_profit,
            "Całkowity dystans": total_distance,
            "Czas wykonania (s)": execution_time,
            "Całkowita waga": total_weight
        })

    # Złożoność obliczeniowa
    tracemalloc.start()
    nn = NearestNeighbor(graph)
    nn.run()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_results.append({
        "Instancja problemu": file,
        "Średnie zużycie pamięci (MB)": peak_memory / 10**6
    })

# Zapisywanie wyników
pd.DataFrame(optimal_results).to_excel("nearest_neighbor_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel("nearest_neighbor_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    pd.DataFrame(results).to_excel(f"nearest_neighbor_stability_{file.split('/')[-1].split('.')[0]}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel("nearest_neighbor_memory_results.xlsx", index=False)
