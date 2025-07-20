import random
import time
import tracemalloc
import pandas as pd
import os
from common.data_loader import load_data

graph = None
itemset = None

Vmax = None
Vmin = None
W = None
R = None
v_w = None

def calculate_total_distance(route):
    """Oblicza całkowitą odległość dla podanej trasy."""
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

def calculate_travel_time(route, weights_at_cities):
    """Oblicza czas podróży zgodnie z funkcją celu."""
    total_time = 0
    
    for i in range(len(route) - 1):
        current_city = route[i]
        next_city = route[i + 1]
        
        distance = 0
        for dest, dist in graph[current_city]:
            if dest == next_city:
                distance = dist
                break
        
        current_weight = weights_at_cities[i]
        speed = Vmax - current_weight * v_w
        total_time += distance / speed
    
    last_city = route[-1]
    start_city = route[0]
    
    return_distance = 0
    for dest, dist in graph[last_city]:
        if dest == start_city:
            return_distance = dist
            break
    
    last_weight = weights_at_cities[-1]
    return_speed = Vmax - last_weight * v_w
    total_time += return_distance / return_speed
    
    return total_time

def calculate_objective_function(route, picked_items, total_profit):
    """Oblicza wartość funkcji celu zgodnie z podanym wzorem matematycznym."""
    weights_at_cities = [0] * len(route)
    current_weight = 0
    
    for i, city in enumerate(route):
        for item_city, item_id in picked_items:
            if item_city == city:
                for item in itemset.get(city, []):
                    if item[0] == item_id:
                        current_weight += item[2]
                        break
        
        weights_at_cities[i] = current_weight
    
    travel_time = calculate_travel_time(route, weights_at_cities)
    travel_cost = R * travel_time
    objective_value = total_profit - travel_cost
    
    return objective_value, travel_time, travel_cost

def solve_knapsack(route):
    """Rozwiązuje problem plecakowy dla podanej trasy, wybierając przedmioty losowo."""
    picked_items = []
    total_profit = 0
    total_weight = 0

    all_items = []
    for city in route:
        for item in itemset.get(city, []):
            item_id, profit, weight = item
            all_items.append((city, item_id, profit, weight))
    
    random.shuffle(all_items)
    
    for city, item_id, profit, weight in all_items:
        if total_weight + weight <= W:
            picked_items.append((city, item_id))
            total_weight += weight
            total_profit += profit

    return picked_items, total_profit, total_weight

good_parameters = {
    "iterations": 100000
}

weak_parameters = {
    "iterations": 1000
}

class RandomSearch:
    def __init__(self, iterations):
        self.iterations = iterations

    def generate_random_route(self):
        """Generuje losową trasę, która zawiera wszystkie miasta."""
        cities = list(graph.keys())
        random.shuffle(cities)
        return cities

    def run(self):
        best_route = self.generate_random_route()
        best_picked_items, best_total_profit, best_total_weight = solve_knapsack(best_route)
        best_objective, best_time, best_cost = calculate_objective_function(best_route, best_picked_items, best_total_profit)
        best_distance = calculate_total_distance(best_route)

        for _ in range(self.iterations):
            new_route = self.generate_random_route()
            picked_items, total_profit, total_weight = solve_knapsack(new_route)
            objective_value, travel_time, travel_cost = calculate_objective_function(new_route, picked_items, total_profit)
            total_distance = calculate_total_distance(new_route)

            if objective_value > best_objective:
                best_route = new_route
                best_objective = objective_value
                best_picked_items = picked_items
                best_total_profit = total_profit
                best_total_weight = total_weight
                best_distance = total_distance
                best_time = travel_time
                best_cost = travel_cost

        return best_route, best_distance, best_picked_items, best_total_profit, best_total_weight, best_time, best_cost, best_objective

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time, total_cost, objective_value):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Koszt podróży: {:.2f}".format(total_cost))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk z przedmiotów: ", total_profit)
    print("Waga przenoszona w plecaku: ", total_weight)
    print("Wartość funkcji celu: {:.2f}".format(objective_value))

output_dir = "tests/output/Algorytm Random Search"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Utworzono katalog: {output_dir}")

files = ["data/50_1.txt", "data/280_1.txt", "data/500_1.txt"] 

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
        
        print("\nTestowanie optymalności rozwiązania - dobre parametry...")
        rs = RandomSearch(**good_parameters)
        
        start_time = time.time()
        (best_route, best_distance, best_items, best_profit, best_weight, 
         best_time, best_cost, best_objective) = rs.run()
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
            "Czas podróży": best_time,
            "Koszt podróży": best_cost,
            "Czas wykonania (s)": execution_time
        })
        
        print("\nTestowanie efektywności czasowej...")
        efficiency_results.append({
            "Instancja problemu": file,
            "Parametry": "Dobre",
            "Czas wykonania (s)": execution_time
        })

        print("\nTestowanie optymalności rozwiązania - słabe parametry...")
        rs = RandomSearch(**weak_parameters)
        
        start_time = time.time()
        (best_route, best_distance, best_items, best_profit, best_weight, 
         best_time, best_cost, best_objective) = rs.run()
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
            "Czas podróży": best_time,
            "Koszt podróży": best_cost,
            "Czas wykonania (s)": execution_time
        })
        
        print("\nTestowanie stabilności wyników...")
        for run in range(5):
            print(f"\nUruchomienie {run+1}/5")
            rs = RandomSearch(**good_parameters)
            
            start_time = time.time()
            (best_route, best_distance, best_items, best_profit, best_weight, 
             best_time, best_cost, best_objective) = rs.run()
            execution_time = time.time() - start_time
            
            print(f"Znaleziono rozwiązanie w czasie {execution_time:.2f} sekund")
            print(f"Wartość funkcji celu: {best_objective:.2f}")
            
            stability_results[file].append({
                "Uruchomienie": run + 1,
                "Wartość funkcji celu": best_objective,
                "Całkowity zysk": best_profit,
                "Całkowita waga": best_weight,
                "Długość trasy": best_distance,
                "Czas podróży": best_time,
                "Koszt podróży": best_cost,
                "Czas wykonania (s)": execution_time
            })
        
        print("\nTestowanie zużycia pamięci...")
        tracemalloc.start()
        rs = RandomSearch(**good_parameters)
        rs.run()
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Aktualne zużycie pamięci: {current / 1024:.2f} KB")
        print(f"Szczytowe zużycie pamięci: {peak_memory / 1024:.2f} KB")
        
        memory_results.append({
            "Instancja problemu": file,
            "Parametry": "Dobre",
            "Zużycie pamięci (KB)": peak_memory / 1024
        })
        
        tracemalloc.start()
        rs = RandomSearch(**weak_parameters)
        rs.run()
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Aktualne zużycie pamięci (słabe parametry): {current / 1024:.2f} KB")
        print(f"Szczytowe zużycie pamięci (słabe parametry): {peak_memory / 1024:.2f} KB")
        
        memory_results.append({
            "Instancja problemu": file,
            "Parametry": "Słabe",
            "Zużycie pamięci (KB)": peak_memory / 1024
        })
        
    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku {file}:")
        print(f"Typ błędu: {type(e).__name__}")
        print(f"Treść błędu: {str(e)}")
        import traceback
        print("Pełny ślad błędu:")
        print(traceback.format_exc())

print("\nZapisywanie wyników do plików Excel...")
pd.DataFrame(optimal_results).to_excel(f"{output_dir}/rs_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel(f"{output_dir}/rs_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    file_name = file.split('/')[-1].split('.')[0]
    pd.DataFrame(results).to_excel(f"{output_dir}/rs_stability_{file_name}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel(f"{output_dir}/rs_memory_results.xlsx", index=False)
print("Zapisano pliki wynikowe.")

print("\nTestowanie zakończone!")
