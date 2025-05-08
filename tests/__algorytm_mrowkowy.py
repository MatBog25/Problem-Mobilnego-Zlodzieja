import random
import math
import time
import tracemalloc
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
import sys

# Dodaj katalog główny do ścieżki Pythona
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_loader import load_data

# Parametry problemu
Vmax = None  # Będą ustawione po wczytaniu danych
Vmin = None
W = None
R = None
v_w = None  # Będzie obliczone po wczytaniu danych

class Ant:
    def __init__(self, alpha, beta, num_cities, graph, itemset):
        self.alpha = alpha
        self.beta = beta
        self.num_cities = num_cities
        self.graph = graph
        self.itemset = itemset
        self.route = []
        self.visited = set()
        self.items_picked = []
        self.distance_traveled = 0
        self.total_profit = 0
        self.current_weight = 0
        self.weights_at_cities = []  # Lista wag plecaka w każdym mieście

    def select_next_city(self, current_city, pheromone):
        probabilities = []
        for dest, dist in self.graph[current_city]:
            if dest not in self.visited and dist > 0:  # Unikamy odległości 0
                tau = abs(pheromone[current_city][dest]) ** self.alpha  # Używamy wartości bezwzględnej
                eta = (1.0 / dist) ** self.beta
                probabilities.append((dest, tau * eta))

        if not probabilities:
            return None

        total = sum(prob for _, prob in probabilities)
        if total == 0:
            return random.choice([dest for dest, _ in self.graph[current_city] if dest not in self.visited])
            
        probabilities = [(city, prob / total) for city, prob in probabilities]
        r = random.random()
        for city, prob in probabilities:
            r -= prob
            if r <= 0:
                return city
        return probabilities[-1][0]

    def pick_items(self, current_city):
        """Wybiera przedmioty z bieżącego miasta, uwzględniając wpływ na czas podróży."""
        items = self.itemset.get(current_city, [])
        available_items = []
        
        # Oblicz pozostałą odległość do przejechania
        remaining_distance = 0
        current_city_idx = self.route.index(current_city)
        for i in range(current_city_idx, len(self.route) - 1):
            current = self.route[i]
            next_city = self.route[i + 1]
            for dest, dist in self.graph[current]:
                if dest == next_city:
                    remaining_distance += dist
                    break
        
        # Dodaj odległość powrotu do miasta początkowego
        last_city = self.route[-1]
        start_city = self.route[0]
        for dest, dist in self.graph[last_city]:
            if dest == start_city:
                remaining_distance += dist
                break
        
        # Oblicz efektywną wartość każdego przedmiotu
        for item_id, profit, weight in items:
            # Oblicz wpływ przedmiotu na prędkość
            speed_without_item = Vmax
            speed_with_item = max(Vmin, Vmax - (self.current_weight + weight) * v_w)
            
            # Oblicz czas podróży bez przedmiotu i z przedmiotem
            time_without_item = remaining_distance / speed_without_item
            time_with_item = remaining_distance / speed_with_item
            
            # Oblicz dodatkowy czas
            additional_time = time_with_item - time_without_item
            
            # Oblicz koszt transportu
            transport_cost = additional_time * R
            
            # Oblicz efektywną wartość przedmiotu
            effective_profit = profit - transport_cost
            
            # Oblicz efektywny stosunek wartości do wagi
            effective_ratio = effective_profit / weight if weight > 0 else 0
            
            # Dodaj tylko przedmioty z pozytywnym wpływem na funkcję celu
            if effective_profit > 0:
                available_items.append((item_id, profit, weight, effective_ratio, effective_profit))
        
        # Sortuj przedmioty według efektywnego stosunku wartości do wagi
        available_items.sort(key=lambda x: x[3], reverse=True)
        
        # Wybierz przedmioty
        for item_id, profit, weight, _, _ in available_items:
            if self.current_weight + weight <= W:
                self.items_picked.append((current_city, item_id))
                self.total_profit += profit
                self.current_weight += weight

    def calculate_travel_time(self):
        """Oblicza czas podróży zgodnie z funkcją celu."""
        total_time = 0
        
        # Obliczanie czasu podróży między kolejnymi miastami
        for i in range(len(self.route) - 1):
            current_city = self.route[i]
            next_city = self.route[i + 1]
            
            # Znajdź odległość między miastami
            distance = 0
            for dest, dist in self.graph[current_city]:
                if dest == next_city:
                    distance = dist
                    break
            
            # Oblicz prędkość na podstawie aktualnej wagi plecaka
            current_weight = self.weights_at_cities[i]
            speed = Vmax - current_weight * v_w
            
            # Dodaj czas podróży
            total_time += distance / speed
        
        # Dodaj czas powrotu do miasta początkowego
        last_city = self.route[-1]
        start_city = self.route[0]
        
        # Znajdź odległość powrotu
        return_distance = 0
        for dest, dist in self.graph[last_city]:
            if dest == start_city:
                return_distance = dist
                break
        
        # Oblicz prędkość na podstawie wagi plecaka po odwiedzeniu wszystkich miast
        last_weight = self.weights_at_cities[-1]
        return_speed = Vmax - last_weight * v_w
        
        # Dodaj czas powrotu
        total_time += return_distance / return_speed
        
        return total_time

    def calculate_objective_value(self):
        """Oblicza wartość funkcji celu."""
        travel_time = self.calculate_travel_time()
        travel_cost = R * travel_time
        objective_value = self.total_profit - travel_cost
        return objective_value, travel_time, travel_cost

    def travel(self, pheromone):
        self.route = [random.randint(1, self.num_cities)]
        self.visited = set(self.route)
        self.distance_traveled = 0
        self.weights_at_cities = [0]  # Początkowa waga plecaka

        while len(self.route) < self.num_cities:
            current_city = self.route[-1]
            self.pick_items(current_city)
            self.weights_at_cities.append(self.current_weight)  # Zapisz aktualną wagę plecaka
            
            next_city = self.select_next_city(current_city, pheromone)
            if next_city is None:
                self.distance_traveled = float('inf')
                return
            self.route.append(next_city)
            self.visited.add(next_city)
            self.distance_traveled += next(dist for dest, dist in self.graph[current_city] if dest == next_city)

        # Powrót do miasta początkowego
        start_city = self.route[0]
        last_city = self.route[-1]
        self.distance_traveled += next(dist for dest, dist in self.graph[last_city] if dest == start_city)
        
        # Oblicz wartość funkcji celu
        self.objective_value, self.travel_time, self.travel_cost = self.calculate_objective_value()

class ACO:
    def __init__(self, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit, graph, itemset):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.graph = graph
        self.itemset = itemset

    def initialize_pheromone(self, num_cities):
        return {i: {j: 1.0 for j in range(1, num_cities + 1) if i != j} for i in range(1, num_cities + 1)}

    def run(self):
        num_cities = len(self.graph)
        pheromone = self.initialize_pheromone(num_cities)
        best_route = None
        best_distance = float('inf')
        best_items = []
        best_profit = 0
        best_time = float('inf')
        best_total_weight = 0
        best_objective_value = float('-inf')
        best_travel_cost = 0

        for iteration in range(self.num_iterations):
            ants = [Ant(self.alpha, self.beta, num_cities, self.graph, self.itemset) for _ in range(self.num_ants)]
            for ant in ants:
                ant.travel(pheromone)
                if ant.distance_traveled == float('inf'):
                    continue

                # Aktualizacja najlepszego wyniku na podstawie wartości funkcji celu
                if ant.objective_value > best_objective_value:
                    best_objective_value = ant.objective_value
                    best_profit = ant.total_profit
                    best_route = ant.route
                    best_distance = ant.distance_traveled
                    best_items = ant.items_picked
                    best_time = ant.travel_time
                    best_total_weight = ant.current_weight
                    best_travel_cost = ant.travel_cost

            # Aktualizacja feromonów
            for i in pheromone:
                for j in pheromone[i]:
                    pheromone[i][j] *= (1 - self.evaporation_rate)

            for ant in ants:
                if ant.distance_traveled < float('inf') and ant.objective_value > float('-inf'):
                    # Aktualizacja feromonów na podstawie wartości funkcji celu
                    for i in range(len(ant.route) - 1):
                        city1 = ant.route[i]
                        city2 = ant.route[i + 1]
                        pheromone[city1][city2] += self.pheromone_deposit * ant.objective_value
                        pheromone[city2][city1] += self.pheromone_deposit * ant.objective_value

        return best_route, best_distance, best_items, best_profit, best_total_weight, best_time, best_objective_value, best_travel_cost

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time, objective_value, travel_cost):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Koszt podróży: {:.2f}".format(travel_cost))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk z przedmiotów: {:.2f}".format(total_profit))
    print("Całkowita waga przedmiotów: ", total_weight)
    print("Wartość funkcji celu: {:.2f}".format(objective_value))

# Pliki testowe
files = ["data/50_1.txt"]  # Poprawiona nazwa pliku

# Parametry algorytmu mrówkowego
good_parameters = {
    "num_ants": 1000,  # Zmniejszamy liczbę mrówek
    "num_iterations": 100,  # Zmniejszamy liczbę iteracji
    "alpha": 1.0,
    "beta": 1.0,
    "evaporation_rate": 0.5,
    "pheromone_deposit": 10
}

weak_parameters = {
    "num_ants": 50,
    "num_iterations": 50,
    "alpha": 1.0,
    "beta": 1.0,
    "evaporation_rate": 0.5,
    "pheromone_deposit": 10
}

# Tworzenie katalogu wynikowego
output_dir = "tests/output/Algorytm Mrowkowy"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Utworzono katalog: {output_dir}")

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
        print(f"Parametry: W={knapsack_capacity}, Vmax={max_speed}, Vmin={min_speed}, R={renting_ratio}")
        
        Vmax = max_speed
        Vmin = min_speed
        W = knapsack_capacity
        R = renting_ratio
        v_w = (max_speed - min_speed) / knapsack_capacity
        print(f"v_w = {v_w}")
        
        # Optymalność rozwiązania - dobre parametry
        print("\nTestowanie optymalności rozwiązania - dobre parametry...")
        aco = ACO(
            num_ants=good_parameters["num_ants"],
            num_iterations=good_parameters["num_iterations"],
            alpha=good_parameters["alpha"],
            beta=good_parameters["beta"],
            evaporation_rate=good_parameters["evaporation_rate"],
            pheromone_deposit=good_parameters["pheromone_deposit"],
            graph=graph,
            itemset=itemset
        )
        
        start_time = time.time()
        best_route, best_distance, best_items, best_profit, best_total_weight, best_time, best_objective_value, best_travel_cost = aco.run()
        execution_time = time.time() - start_time
        
        print(f"\nZnaleziono rozwiązanie w czasie {execution_time:.2f} sekund")
        print(f"Wartość funkcji celu: {best_objective_value:.2f}")
        print(f"Całkowity zysk: {best_profit:.2f}")
        print(f"Całkowita waga: {best_total_weight:.2f}")
        print(f"Znaleziona trasa: {best_route}")
        print(f"Wybrane przedmioty: {best_items}")
        
        optimal_results.append({
            "Instancja problemu": file,
            "Parametry": "Dobre",
            "Wartość funkcji celu": best_objective_value,
            "Całkowity zysk": best_profit,
            "Całkowita waga": best_total_weight,
            "Długość trasy": best_distance,
            "Czas podróży": best_time,
            "Koszt podróży": best_travel_cost,
            "Czas wykonania (s)": execution_time
        })
        
        # Optymalność rozwiązania - słabe parametry
        print("\nTestowanie optymalności rozwiązania - słabe parametry...")
        aco = ACO(
            num_ants=weak_parameters["num_ants"],
            num_iterations=weak_parameters["num_iterations"],
            alpha=weak_parameters["alpha"],
            beta=weak_parameters["beta"],
            evaporation_rate=weak_parameters["evaporation_rate"],
            pheromone_deposit=weak_parameters["pheromone_deposit"],
            graph=graph,
            itemset=itemset
        )
        
        start_time = time.time()
        best_route, best_distance, best_items, best_profit, best_total_weight, best_time, best_objective_value, best_travel_cost = aco.run()
        execution_time = time.time() - start_time
        
        print(f"\nZnaleziono rozwiązanie w czasie {execution_time:.2f} sekund")
        print(f"Wartość funkcji celu: {best_objective_value:.2f}")
        print(f"Całkowity zysk: {best_profit:.2f}")
        print(f"Całkowita waga: {best_total_weight:.2f}")
        print(f"Znaleziona trasa: {best_route}")
        print(f"Wybrane przedmioty: {best_items}")
        
        optimal_results.append({
            "Instancja problemu": file,
            "Parametry": "Słabe",
            "Wartość funkcji celu": best_objective_value,
            "Całkowity zysk": best_profit,
            "Całkowita waga": best_total_weight,
            "Długość trasy": best_distance,
            "Czas podróży": best_time,
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
            aco = ACO(
                num_ants=good_parameters["num_ants"],
                num_iterations=good_parameters["num_iterations"],
                alpha=good_parameters["alpha"],
                beta=good_parameters["beta"],
                evaporation_rate=good_parameters["evaporation_rate"],
                pheromone_deposit=good_parameters["pheromone_deposit"],
                graph=graph,
                itemset=itemset
            )
            
            start_time = time.time()
            best_route, best_distance, best_items, best_profit, best_total_weight, best_time, best_objective_value, best_travel_cost = aco.run()
            execution_time = time.time() - start_time
            
            print(f"Znaleziono rozwiązanie w czasie {execution_time:.2f} sekund")
            print(f"Wartość funkcji celu: {best_objective_value:.2f}")
            print(f"Znaleziona trasa: {best_route}")
            
            stability_results[file].append({
                "Uruchomienie": run + 1,
                "Wartość funkcji celu": best_objective_value,
                "Całkowity zysk": best_profit,
                "Całkowita waga": best_total_weight,
                "Długość trasy": best_distance,
                "Czas podróży": best_time,
                "Koszt podróży": best_travel_cost,
                "Czas wykonania (s)": execution_time
            })
        
        # Złożoność pamięciowa
        print("\nTestowanie zużycia pamięci...")
        try:
            # Test dla dobrych parametrów
            tracemalloc.start()
            aco = ACO(
                num_ants=good_parameters["num_ants"],
                num_iterations=good_parameters["num_iterations"],
                alpha=good_parameters["alpha"],
                beta=good_parameters["beta"],
                evaporation_rate=good_parameters["evaporation_rate"],
                pheromone_deposit=good_parameters["pheromone_deposit"],
                graph=graph,
                itemset=itemset
            )
            aco.run()
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_results.append({
                "Instancja problemu": file,
                "Parametry": "Dobre",
                "Zużycie pamięci (MB)": peak_memory / 10**6
            })
            
            # Czyszczenie pamięci
            del aco
            import gc
            gc.collect()
            
            # Test dla słabych parametrów
            tracemalloc.start()
            aco = ACO(
                num_ants=weak_parameters["num_ants"],
                num_iterations=weak_parameters["num_iterations"],
                alpha=weak_parameters["alpha"],
                beta=weak_parameters["beta"],
                evaporation_rate=weak_parameters["evaporation_rate"],
                pheromone_deposit=weak_parameters["pheromone_deposit"],
                graph=graph,
                itemset=itemset
            )
            aco.run()
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_results.append({
                "Instancja problemu": file,
                "Parametry": "Słabe",
                "Zużycie pamięci (MB)": peak_memory / 10**6
            })
            
            # Czyszczenie pamięci
            del aco
            gc.collect()
            
        except Exception as e:
            print(f"Błąd podczas testowania pamięci: {str(e)}")
            memory_results.append({
                "Instancja problemu": file,
                "Parametry": "Dobre",
                "Zużycie pamięci (MB)": f"Błąd: {str(e)}"
            })
            memory_results.append({
                "Instancja problemu": file,
                "Parametry": "Słabe",
                "Zużycie pamięci (MB)": f"Błąd: {str(e)}"
            })
        finally:
            # Upewnij się, że tracemalloc jest zatrzymany
            try:
                tracemalloc.stop()
            except:
                pass
    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku {file}:")
        print(f"Typ błędu: {type(e).__name__}")
        print(f"Treść błędu: {str(e)}")
        import traceback
        print("Pełny ślad błędu:")
        print(traceback.format_exc())

# Zapisywanie wyników
print("\nZapisywanie wyników do plików Excel...")
pd.DataFrame(optimal_results).to_excel(f"{output_dir}/aco_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel(f"{output_dir}/aco_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    file_name = file.split('/')[-1].split('.')[0]
    pd.DataFrame(results).to_excel(f"{output_dir}/aco_stability_{file_name}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel(f"{output_dir}/aco_memory_results.xlsx", index=False)
print("Zapisano pliki wynikowe.")

print("\nTestowanie zakończone!")

if __name__ == "__main__":
    # Przykład użycia
    print("Uruchamianie testów algorytmu mrówkowego...")
    # Testy zostały już wykonane powyżej
