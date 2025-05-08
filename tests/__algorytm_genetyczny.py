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

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, num_generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.population = []
        self.best_fitness_history = []

    def initialize_population(self, graph, itemset, W, Vmax, Vmin, R, v_w):
        """Inicjalizuje populację losowymi trasami i wektorami wyboru przedmiotów."""
        self.graph = graph
        self.itemset = itemset
        self.W = W
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.R = R
        self.v_w = v_w
        
        cities = list(graph.keys())
        
        # Użyj różnych strategii inicjalizacji dla różnorodności populacji
        for i in range(self.population_size):
            if i < self.population_size // 3:
                # Strategia 1: Losowa trasa
                route = cities[:]
                random.shuffle(route)
            elif i < 2 * self.population_size // 3:
                # Strategia 2: Trasa oparta na odległości (najbliższy sąsiad)
                route = self.nearest_neighbor_route(cities)
            else:
                # Strategia 3: Trasa oparta na wartości przedmiotów
                route = self.value_based_route(cities)
                
            # Inicjalizacja wektora przedmiotów z preferencją dla przedmiotów o wysokim stosunku wartości do wagi
            items = self.initialize_items_based_on_value_weight_ratio()
            
            self.population.append((route, items))
    
    def nearest_neighbor_route(self, cities):
        """Tworzy trasę używając algorytmu najbliższego sąsiada."""
        unvisited = cities[:]
        route = [unvisited.pop(0)]  # Zacznij od pierwszego miasta
        
        while unvisited:
            current = route[-1]
            # Znajdź najbliższe nieodwiedzone miasto
            next_city = min(unvisited, key=lambda city: self.get_distance(current, city))
            route.append(next_city)
            unvisited.remove(next_city)
            
        return route
    
    def value_based_route(self, cities):
        """Tworzy trasę opartą na wartości przedmiotów w miastach."""
        # Oblicz średnią wartość przedmiotów w każdym mieście
        city_values = {}
        for city in cities:
            total_value = 0
            item_count = 0
            for item in self.itemset.get(city, []):
                _, profit, _ = item
                total_value += profit
                item_count += 1
            city_values[city] = total_value / max(1, item_count)
        
        # Posortuj miasta według wartości przedmiotów (malejąco)
        sorted_cities = sorted(cities, key=lambda city: city_values[city], reverse=True)
        return sorted_cities
    
    def get_distance(self, city1, city2):
        """Zwraca odległość między dwoma miastami."""
        for dest, dist in self.graph[city1]:
            if dest == city2:
                return dist
        return float('inf')
    
    def initialize_items_based_on_value_weight_ratio(self):
        """Inicjalizuje wektor przedmiotów z preferencją dla przedmiotów o wysokim stosunku wartości do wagi."""
        items = [0] * len(self.itemset)
        
        # Oblicz stosunek wartości do wagi dla każdego przedmiotu
        item_ratios = []
        for city, city_items in self.itemset.items():
            for item in city_items:
                item_id, profit, weight = item
                ratio = profit / weight if weight > 0 else 0
                item_ratios.append((item_id, ratio))
        
        # Posortuj przedmioty według stosunku wartości do wagi (malejąco)
        item_ratios.sort(key=lambda x: x[1], reverse=True)
        
        # Wybierz przedmioty o najwyższym stosunku wartości do wagi
        current_weight = 0
        for item_id, _ in item_ratios:
            # Znajdź wagę przedmiotu
            for city_items in self.itemset.values():
                for item in city_items:
                    if item[0] == item_id:
                        _, _, weight = item
                        if current_weight + weight <= self.W:
                            items[item_id - 1] = 1
                            current_weight += weight
                        break
                if items[item_id - 1] == 1:
                    break
        
        return items

    def calculate_travel_time(self, route, weights_at_cities):
        """Oblicza czas podróży zgodnie z funkcją celu."""
        total_time = 0
        
        # Obliczanie czasu podróży między kolejnymi miastami
        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            
            # Znajdź odległość między miastami
            distance = self.get_distance(current_city, next_city)
            
            # Oblicz prędkość na podstawie aktualnej wagi plecaka
            current_weight = weights_at_cities[i]
            speed = self.Vmax - current_weight * self.v_w  # Dokładnie zgodnie z funkcją celu
            
            # Dodaj czas podróży
            total_time += distance / speed
        
        # Dodaj czas powrotu do miasta początkowego
        last_city = route[-1]
        start_city = route[0]
        
        # Znajdź odległość powrotu
        return_distance = self.get_distance(last_city, start_city)
        
        # Oblicz prędkość na podstawie wagi plecaka po odwiedzeniu wszystkich miast
        last_weight = weights_at_cities[-1]
        return_speed = self.Vmax - last_weight * self.v_w  # Dokładnie zgodnie z funkcją celu
        
        # Dodaj czas powrotu
        total_time += return_distance / return_speed
        
        return total_time

    def fitness(self, chromosome):
        """Oblicza funkcję celu dla danego chromosomu."""
        route, items = chromosome
        total_distance = 0
        total_profit = 0
        current_weight = 0
        weights_at_cities = [0]  # Początkowa waga plecaka
        picked_items = []

        # Najpierw oblicz odległość całej trasy
        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            total_distance += self.get_distance(current_city, next_city)

        # Dodaj dystans powrotny do miasta startowego
        total_distance += self.get_distance(route[-1], route[0])

        # Teraz wybierz przedmioty, uwzględniając wpływ na prędkość
        for i, current_city in enumerate(route):
            # Wybierz przedmioty z aktualnego miasta
            for item in self.itemset.get(current_city, []):
                item_id, profit, weight = item
                if items[item_id - 1]:  # Jeśli przedmiot jest wybrany
                    if current_weight + weight <= self.W:  # Sprawdzenie przed dodaniem przedmiotu
                        picked_items.append((current_city, item_id))
                        current_weight += weight
                        total_profit += profit
            
            # Zapisz aktualną wagę plecaka po opuszczeniu miasta
            weights_at_cities.append(current_weight)

        # Oblicz czas podróży zgodnie z funkcją celu
        travel_time = self.calculate_travel_time(route, weights_at_cities)
        
        # Oblicz koszt podróży
        travel_cost = self.R * travel_time
        
        # Oblicz wartość funkcji celu
        fitness = total_profit - travel_cost
        
        return fitness, picked_items, current_weight, total_profit, travel_time, travel_cost, total_distance

    def select_parent(self):
        """Turniejowy wybór rodziców."""
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: self.fitness(x)[0], reverse=True)
        return tournament[0]

    def crossover(self, parent1, parent2):
        """Krzyżowanie tras i wektorów wyboru przedmiotów."""
        route1, items1 = parent1
        route2, items2 = parent2

        # Krzyżowanie tras - użyj Order Crossover (OX)
        if random.random() < 0.5:
            # Order Crossover (OX)
            start, end = sorted(random.sample(range(len(route1)), 2))
            child_route = [-1] * len(route1)
            child_route[start:end] = route1[start:end]
            
            # Wypełnij pozostałe pozycje z drugiego rodzica
            j = end
            for i in range(len(route2)):
                if route2[i] not in child_route:
                    if j == len(child_route):
                        j = 0
                    child_route[j] = route2[i]
                    j += 1
        else:
            # Partially Mapped Crossover (PMX)
            start, end = sorted(random.sample(range(len(route1)), 2))
            child_route = [-1] * len(route1)
            
            # Skopiuj środkowy fragment z pierwszego rodzica
            child_route[start:end] = route1[start:end]
            
            # Utwórz mapowanie
            mapping = {}
            for i in range(start, end):
                mapping[route1[i]] = route2[i]
            
            # Wypełnij pozostałe pozycje
            for i in range(len(child_route)):
                if i < start or i >= end:
                    city = route2[i]
                    while city in mapping:
                        if city in mapping:
                            city = mapping[city]
                        else:
                            # Jeśli nie ma mapowania, wybierz losowe miasto
                            remaining_cities = [c for c in self.graph.keys() if c not in child_route]
                            if remaining_cities:
                                city = random.choice(remaining_cities)
                            else:
                                break
                    if city not in child_route:
                        child_route[i] = city

        # Krzyżowanie wektorów przedmiotów - użyj jednopunktowego krzyżowania
        crossover_point = random.randint(0, min(len(items1), len(items2)) - 1)
        child_items = items1[:crossover_point] + items2[crossover_point:]

        return (child_route, child_items)

    def mutate(self, chromosome):
        """Mutacja trasy i wektora przedmiotów."""
        route, items = chromosome
        
        # Mutacja trasy - użyj różnych operatorów mutacji
        if random.random() < self.mutation_rate:
            mutation_type = random.random()
            
            if mutation_type < 0.33:
                # Swap mutation - zamień dwa miasta
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]
            elif mutation_type < 0.66:
                # Reverse mutation - odwróć fragment trasy
                start, end = sorted(random.sample(range(len(route)), 2))
                route[start:end+1] = reversed(route[start:end+1])
            else:
                # Insert mutation - wstaw miasto w nowe miejsce
                city = route.pop(random.randint(0, len(route) - 1))
                route.insert(random.randint(0, len(route)), city)

        # Mutacja wyboru przedmiotów
        if random.random() < self.mutation_rate:
            # Mutacja bitowa - zmień kilka bitów
            num_mutations = random.randint(1, min(3, len(items)))
            for _ in range(num_mutations):
                idx = random.randint(0, len(items) - 1)
                items[idx] = 1 - items[idx]  # Przełącz 0 na 1 lub 1 na 0

    def evolve(self):
        """Ewoluuje populację."""
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

    def run(self, graph, itemset, W, Vmax, Vmin, R, v_w):
        """Uruchamia algorytm genetyczny."""
        self.initialize_population(graph, itemset, W, Vmax, Vmin, R, v_w)
        best_fitness_so_far = float('-inf')
        no_improvement_count = 0
        best_solution = None
        
        for generation in range(self.num_generations):
            self.evolve()
            
            # Znajdź najlepsze rozwiązanie w bieżącej generacji
            current_best = max(self.population, key=lambda x: self.fitness(x)[0])
            current_fitness, _, _, _, _, _, _ = self.fitness(current_best)
            
            # Zapisz historię najlepszych wartości funkcji celu
            self.best_fitness_history.append(current_fitness)
            
            # Sprawdź, czy znaleziono lepsze rozwiązanie
            if current_fitness > best_fitness_so_far:
                best_fitness_so_far = current_fitness
                best_solution = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Jeśli nie ma poprawy przez 20 generacji, zwiększ wskaźnik mutacji
            if no_improvement_count > 20:
                self.mutation_rate = min(0.5, self.mutation_rate * 1.5)
                no_improvement_count = 0
        
        # Jeśli nie znaleziono lepszego rozwiązania, użyj najlepszego z ostatniej generacji
        if best_solution is None:
            best_solution = max(self.population, key=lambda x: self.fitness(x)[0])
            
        return best_solution

# Pliki testowe
files = ["data/50_1.txt"]

# Parametry algorytmu genetycznego
good_parameters = {
    "population_size": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "num_generations": 100
}

weak_parameters = {
    "population_size": 20,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "num_generations": 20
}

# Tworzenie katalogu wynikowego
output_dir = "tests/output/Algorytm Genetyczny"
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
    graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)
    v_w = (max_speed - min_speed) / knapsack_capacity  # Obliczanie v_w
    
    # Optymalność rozwiązania - dobre parametry
    print("\nTestowanie optymalności rozwiązania - dobre parametry...")
    ga = GeneticAlgorithm(
        population_size=good_parameters["population_size"],
        mutation_rate=good_parameters["mutation_rate"],
        crossover_rate=good_parameters["crossover_rate"],
        num_generations=good_parameters["num_generations"]
    )
    
    start_time = time.time()
    best_solution = ga.run(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, v_w)
    execution_time = time.time() - start_time
    
    best_fitness, picked_items, total_weight, total_profit, travel_time, travel_cost, total_distance = ga.fitness(best_solution)
    
    optimal_results.append({
        "Instancja problemu": file,
        "Parametry": "Dobre",
        "Wartość funkcji celu": best_fitness,
        "Całkowity zysk": total_profit,
        "Całkowita waga": total_weight,
        "Długość trasy": total_distance,
        "Czas podróży": travel_time,
        "Koszt podróży": travel_cost,
        "Czas wykonania (s)": execution_time
    })
    
    # Optymalność rozwiązania - słabe parametry
    print("\nTestowanie optymalności rozwiązania - słabe parametry...")
    ga = GeneticAlgorithm(
        population_size=weak_parameters["population_size"],
        mutation_rate=weak_parameters["mutation_rate"],
        crossover_rate=weak_parameters["crossover_rate"],
        num_generations=weak_parameters["num_generations"]
    )
    
    start_time = time.time()
    best_solution = ga.run(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, v_w)
    execution_time = time.time() - start_time
    
    best_fitness, picked_items, total_weight, total_profit, travel_time, travel_cost, total_distance = ga.fitness(best_solution)
    
    optimal_results.append({
        "Instancja problemu": file,
        "Parametry": "Słabe",
        "Wartość funkcji celu": best_fitness,
        "Całkowity zysk": total_profit,
        "Całkowita waga": total_weight,
        "Długość trasy": total_distance,
        "Czas podróży": travel_time,
        "Koszt podróży": travel_cost,
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
        print(f"Uruchomienie {run+1}/5")
        ga = GeneticAlgorithm(
            population_size=good_parameters["population_size"],
            mutation_rate=good_parameters["mutation_rate"],
            crossover_rate=good_parameters["crossover_rate"],
            num_generations=good_parameters["num_generations"]
        )
        
        start_time = time.time()
        best_solution = ga.run(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, v_w)
        execution_time = time.time() - start_time
        
        best_fitness, picked_items, total_weight, total_profit, travel_time, travel_cost, total_distance = ga.fitness(best_solution)
        
        stability_results[file].append({
            "Uruchomienie": run + 1,
            "Wartość funkcji celu": best_fitness,
            "Całkowity zysk": total_profit,
            "Całkowita waga": total_weight,
            "Długość trasy": total_distance,
            "Czas podróży": travel_time,
            "Koszt podróży": travel_cost,
            "Czas wykonania (s)": execution_time
        })
    
    # Złożoność pamięciowa
    print("\nTestowanie zużycia pamięci...")
    tracemalloc.start()
    ga = GeneticAlgorithm(
        population_size=good_parameters["population_size"],
        mutation_rate=good_parameters["mutation_rate"],
        crossover_rate=good_parameters["crossover_rate"],
        num_generations=good_parameters["num_generations"]
    )
    ga.run(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, v_w)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_results.append({
        "Instancja problemu": file,
        "Parametry": "Dobre",
        "Zużycie pamięci (MB)": peak_memory / 10**6
    })
    
    # Test pamięci dla słabych parametrów
    tracemalloc.start()
    ga = GeneticAlgorithm(
        population_size=weak_parameters["population_size"],
        mutation_rate=weak_parameters["mutation_rate"],
        crossover_rate=weak_parameters["crossover_rate"],
        num_generations=weak_parameters["num_generations"]
    )
    ga.run(graph, itemset, knapsack_capacity, max_speed, min_speed, renting_ratio, v_w)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_results.append({
        "Instancja problemu": file,
        "Parametry": "Słabe",
        "Zużycie pamięci (MB)": peak_memory / 10**6
    })

# Zapisywanie wyników
print("\nZapisywanie wyników do plików Excel...")
pd.DataFrame(optimal_results).to_excel(f"{output_dir}/ga_optimal_results.xlsx", index=False)
pd.DataFrame(efficiency_results).to_excel(f"{output_dir}/ga_efficiency_results.xlsx", index=False)
for file, results in stability_results.items():
    file_name = file.split('/')[-1].split('.')[0]
    pd.DataFrame(results).to_excel(f"{output_dir}/ga_stability_{file_name}.xlsx", index=False)
pd.DataFrame(memory_results).to_excel(f"{output_dir}/ga_memory_results.xlsx", index=False)
print("Zapisano pliki wynikowe.")

print("\nTestowanie zakończone!")

if __name__ == "__main__":
    # Przykład użycia
    print("Uruchamianie testów algorytmu genetycznego...")
    # Testy zostały już wykonane powyżej
