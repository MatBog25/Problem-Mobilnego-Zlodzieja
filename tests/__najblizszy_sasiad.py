import random
import time
import tracemalloc
import pandas as pd
import os
from common.data_loader import load_data  # Funkcja wczytująca dane

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/50_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio
# Obliczanie v_w - spadek prędkości w funkcji ciężaru plecaka
v_w = (Vmax - Vmin) / W

def calculate_total_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    # Dodaj odległość powrotu do miasta początkowego
    start_city = route[0]
    last_city = route[-1]
    for dest, dist in graph[last_city]:
        if dest == start_city:
            total_distance += dist
            break
    return total_distance

def calculate_travel_time(route, weights_at_cities):
    """Oblicza czas podróży zgodnie z funkcją celu."""
    total_time = 0
    
    # Obliczanie czasu podróży między kolejnymi miastami (od x_1 do x_n)
    for i in range(len(route) - 1):
        current_city = route[i]
        next_city = route[i + 1]
        
        # Znajdź odległość między miastami
        distance = 0
        for dest, dist in graph[current_city]:
            if dest == next_city:
                distance = dist
                break
        
        # Oblicz prędkość na podstawie aktualnej wagi plecaka
        current_weight = weights_at_cities[i]
        speed = Vmax - current_weight * v_w  # Dokładnie zgodnie z funkcją celu
        
        # Dodaj czas podróży
        total_time += distance / speed
    
    # Dodaj czas powrotu do miasta początkowego (od x_n do x_1)
    last_city = route[-1]
    start_city = route[0]
    
    # Znajdź odległość powrotu
    return_distance = 0
    for dest, dist in graph[last_city]:
        if dest == start_city:
            return_distance = dist
            break
    
    # Oblicz prędkość na podstawie wagi plecaka po odwiedzeniu wszystkich miast
    last_weight = weights_at_cities[-1]
    return_speed = Vmax - last_weight * v_w  # Dokładnie zgodnie z funkcją celu
    
    # Dodaj czas powrotu
    total_time += return_distance / return_speed
    
    return total_time

def calculate_objective_function(route, picked_items, total_profit):
    """Oblicza wartość funkcji celu zgodnie z podanym wzorem matematycznym."""
    # Obliczanie wag plecaka w każdym mieście
    weights_at_cities = [0] * len(route)
    current_weight = 0
    
    # Dla każdego miasta w trasie
    for i, city in enumerate(route):
        # Dodaj wagę przedmiotów zabranych w tym mieście
        for item_city, item_id in picked_items:
            if item_city == city:
                # Znajdź przedmiot w itemset
                for item in itemset.get(city, []):
                    if item[0] == item_id:  # item[0] to item_id
                        current_weight += item[2]  # item[2] to waga
                        break
        
        # Zapisz aktualną wagę plecaka po opuszczeniu miasta
        weights_at_cities[i] = current_weight
    
    # Oblicz czas podróży
    travel_time = calculate_travel_time(route, weights_at_cities)
    
    # Oblicz koszt podróży
    travel_cost = R * travel_time
    
    # Oblicz wartość funkcji celu
    objective_value = total_profit - travel_cost
    
    return objective_value, travel_time, travel_cost

def solve_knapsack(route):
    """Rozwiązuje problem plecakowy dla podanej trasy."""
    picked_items = []
    total_profit = 0
    total_weight = 0

    # Sortuj przedmioty według stosunku wartości do wagi (malejąco)
    all_items = []
    for city in route:
        for item in itemset.get(city, []):
            item_id, profit, weight = item
            ratio = profit / weight if weight > 0 else 0
            all_items.append((city, item_id, profit, weight, ratio))
    
    # Sortuj przedmioty według stosunku wartości do wagi (malejąco)
    all_items.sort(key=lambda x: x[4], reverse=True)
    
    # Wybierz przedmioty z najwyższym stosunkiem wartości do wagi
    for city, item_id, profit, weight, ratio in all_items:
        if total_weight + weight <= W:
            picked_items.append((city, item_id))
            total_weight += weight
            total_profit += profit

    return picked_items, total_profit, total_weight

def greedy_item_selection(route):
    """Zachłannie wybiera przedmioty podczas przemieszczania się po trasie."""
    picked_items = []
    total_profit = 0
    total_weight = 0
    
    # Dla każdego miasta w trasie
    for city in route:
        # Sortuj przedmioty w aktualnym mieście według stosunku wartości do wagi (malejąco)
        city_items = []
        for item in itemset.get(city, []):
            item_id, profit, weight = item
            ratio = profit / weight if weight > 0 else 0
            city_items.append((item_id, profit, weight, ratio))
        
        city_items.sort(key=lambda x: x[3], reverse=True)  # Sortuj według stosunku wartości/wagi
        
        # Wybierz jak najwięcej przedmiotów z aktualnego miasta
        for item_id, profit, weight, _ in city_items:
            if total_weight + weight <= W:
                picked_items.append((city, item_id))
                total_weight += weight
                total_profit += profit
    
    return picked_items, total_profit, total_weight

# Parametry algorytmu
good_parameters = {
    "num_attempts": 5000000000,  # Liczba prób z różnymi miastami początkowymi
    "greedy_selection": True  # Użyj zachłannego wyboru przedmiotów
}

weak_parameters = {
    "num_attempts": 1000000000,  # Tylko jedna próba
    "greedy_selection": True
}

class NearestNeighbor:
    def __init__(self, num_attempts=1, greedy_selection=True):
        self.num_attempts = num_attempts
        self.greedy_selection = greedy_selection
        self.route = []
        self.visited = set()
        self.total_distance = 0

    def find_route(self, start):
        current_city = start
        self.route.append(current_city)
        self.visited.add(current_city)

        while len(self.visited) < len(graph):
            next_city = None
            shortest_distance = float("inf")
            for dest, dist in graph[current_city]:
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
        if len(self.route) == len(graph):
            start_city = self.route[0]
            last_city = self.route[-1]
            for dest, dist in graph[last_city]:
                if dest == start_city:
                    self.route.append(start_city)
                    self.total_distance += dist
                    break

    def run(self):
        best_route = None
        best_distance = float("inf")
        best_objective = float("-inf")
        best_items = None
        best_profit = 0
        best_weight = 0
        best_travel_time = 0
        best_travel_cost = 0
        
        for _ in range(self.num_attempts):
            start = random.choice(list(graph.keys()))
            self.route = []
            self.visited = set()
            self.total_distance = 0
            self.find_route(start)
            
            if self.greedy_selection:
                picked_items, total_profit, total_weight = greedy_item_selection(self.route)
            else:
                picked_items, total_profit, total_weight = solve_knapsack(self.route)
            
            objective_value, travel_time, travel_cost = calculate_objective_function(self.route, picked_items, total_profit)
            
            if objective_value > best_objective:
                best_route = list(self.route)
                best_distance = self.total_distance
                best_objective = objective_value
                best_items = picked_items
                best_profit = total_profit
                best_weight = total_weight
                best_travel_time = travel_time
                best_travel_cost = travel_cost
        
        return (best_route, best_distance, best_items, best_profit, best_weight, 
                best_objective, best_travel_time, best_travel_cost)

def print_solution(route, total_distance, picked_items, total_profit, total_weight, objective_value):
    # Oblicz czas podróży i koszt podróży
    weights_at_cities = [0] * len(route)
    current_weight = 0
    
    # Dla każdego miasta w trasie
    for i, city in enumerate(route):
        # Dodaj wagę przedmiotów zabranych w tym mieście
        for item_city, item_id in picked_items:
            if item_city == city:
                # Znajdź przedmiot w itemset
                for item in itemset.get(city, []):
                    if item[0] == item_id:  # item[0] to item_id
                        current_weight += item[2]  # item[2] to waga
                        break
        
        # Zapisz aktualną wagę plecaka po opuszczeniu miasta
        weights_at_cities[i] = current_weight
    
    # Oblicz czas podróży
    travel_time = calculate_travel_time(route, weights_at_cities)
    
    # Oblicz koszt podróży
    travel_cost = R * travel_time
    
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(travel_time))
    print("Koszt podróży: {:.2f}".format(travel_cost))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk z przedmiotów: ", total_profit)
    print("Waga przenoszona w plecaku: ", total_weight)
    print("Wartość funkcji celu: {:.2f}".format(objective_value))

# Tworzenie katalogu wynikowego
output_dir = "tests/output/Algorytm Najblizszego Sasiada"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Utworzono katalog: {output_dir}")

# Pliki testowe
files = ["data/50_1.txt"]

# Wyniki
optimal_results = []
efficiency_results = []
stability_results = {file: [] for file in files}
memory_results = []

for file in files:
    print(f"\nPrzetwarzanie pliku: {file}")
    try:
        print("Wczytywanie danych...")
        graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)
        print(f"Wczytano dane: {len(graph)} miast, {sum(len(items) for items in itemset.values())} przedmiotów")
        
        Vmax = max_speed
        Vmin = min_speed
        W = knapsack_capacity
        R = renting_ratio
        v_w = (Vmax - Vmin) / W
        
        # Optymalność rozwiązania - dobre parametry
        print("\nTestowanie optymalności rozwiązania - dobre parametry...")
        nn = NearestNeighbor(**good_parameters)
        
        start_time = time.time()
        (best_route, best_distance, best_items, best_profit, best_weight, 
         best_objective, best_travel_time, best_travel_cost) = nn.run()
        execution_time = time.time() - start_time
        
        print(f"\nZnaleziono rozwiązanie w czasie {execution_time:.2f} sekund")
        print(f"Wartość funkcji celu: {best_objective:.2f}")
        print(f"Całkowity zysk: {best_profit:.2f}")
        print(f"Całkowita waga: {best_weight:.2f}")
        
        optimal_results.append({
            "Instancja problemu": file,
            "Parametry": "Dobre",
            "Wartość funkcji celu": best_objective,
            "Całkowity zysk": best_profit,
            "Całkowita waga": best_weight,
            "Długość trasy": best_distance,
            "Czas podróży": best_travel_time,
            "Koszt podróży": best_travel_cost,
            "Czas wykonania (s)": execution_time
        })
        
        # Optymalność rozwiązania - słabe parametry
        print("\nTestowanie optymalności rozwiązania - słabe parametry...")
        nn = NearestNeighbor(**weak_parameters)
        
        start_time = time.time()
        (best_route, best_distance, best_items, best_profit, best_weight, 
         best_objective, best_travel_time, best_travel_cost) = nn.run()
        execution_time = time.time() - start_time
        
        print(f"\nZnaleziono rozwiązanie w czasie {execution_time:.2f} sekund")
        print(f"Wartość funkcji celu: {best_objective:.2f}")
        print(f"Całkowity zysk: {best_profit:.2f}")
        print(f"Całkowita waga: {best_weight:.2f}")
        
        optimal_results.append({
            "Instancja problemu": file,
            "Parametry": "Słabe",
            "Wartość funkcji celu": best_objective,
            "Całkowity zysk": best_profit,
            "Całkowita waga": best_weight,
            "Długość trasy": best_distance,
            "Czas podróży": best_travel_time,
            "Koszt podróży": best_travel_cost,
            "Czas wykonania (s)": execution_time
        })
        
        # Efektywność czasowa
        print("\nTestowanie efektywności czasowej...")
        efficiency_results.append({
            "Instancja problemu": file,
            "Parametry": "Dobre",
            "Czas wykonania (s)": execution_time
        })
        
        # Stabilność wyników
        print("\nTestowanie stabilności wyników...")
        for run in range(5):
            print(f"\nUruchomienie {run+1}/5")
            nn = NearestNeighbor(**good_parameters)
            
            start_time = time.time()
            (best_route, best_distance, best_items, best_profit, best_weight, 
             best_objective, best_travel_time, best_travel_cost) = nn.run()
            execution_time = time.time() - start_time
            
            print(f"Znaleziono rozwiązanie w czasie {execution_time:.2f} sekund")
            print(f"Wartość funkcji celu: {best_objective:.2f}")
            
            stability_results[file].append({
                "Uruchomienie": run + 1,
                "Wartość funkcji celu": best_objective,
                "Całkowity zysk": best_profit,
                "Całkowita waga": best_weight,
                "Długość trasy": best_distance,
                "Czas podróży": best_travel_time,
                "Koszt podróży": best_travel_cost,
                "Czas wykonania (s)": execution_time
            })
        
        # Złożoność pamięciowa
        print("\nTestowanie zużycia pamięci...")
        tracemalloc.start()
        nn = NearestNeighbor(**good_parameters)
        nn.run()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_results.append({
            "Instancja problemu": file,
            "Parametry": "Dobre",
            "Zużycie pamięci (MB)": peak_memory / 10**6
        })
        
        # Test pamięci dla słabych parametrów
        tracemalloc.start()
        nn = NearestNeighbor(**weak_parameters)
        nn.run()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_results.append({
            "Instancja problemu": file,
            "Parametry": "Słabe",
            "Zużycie pamięci (MB)": peak_memory / 10**6
        })
        
    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku {file}:")
        print(f"Typ błędu: {type(e).__name__}")
        print(f"Treść błędu: {str(e)}")
        import traceback
        print("Pełny ślad błędu:")
        print(traceback.format_exc())

# Zapisywanie wyników
print("\nZapisywanie wyników do plików Excel...")
pd.DataFrame(optimal_results).to_excel(f"{output_dir}/nn_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel(f"{output_dir}/nn_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    file_name = file.split('/')[-1].split('.')[0]
    pd.DataFrame(results).to_excel(f"{output_dir}/nn_stability_{file_name}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel(f"{output_dir}/nn_memory_results.xlsx", index=False)
print("Zapisano pliki wynikowe.")

print("\nTestowanie zakończone!")
