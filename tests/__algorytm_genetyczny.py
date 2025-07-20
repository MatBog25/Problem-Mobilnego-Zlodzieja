import random
import math
import time
import tracemalloc
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
import sys

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
        
        for i in range(self.population_size):
            if i < self.population_size // 3:
                route = cities[:]
                random.shuffle(route)
            elif i < 2 * self.population_size // 3:
                route = self.nearest_neighbor_route(cities)
            else:
                route = self.value_based_route(cities)
            items = self.initialize_items_based_on_value_weight_ratio()
            
            self.population.append((route, items))
    
    def nearest_neighbor_route(self, cities):
        """Tworzy trasę używając algorytmu najbliższego sąsiada."""
        unvisited = cities[:]
        route = [unvisited.pop(0)]
        
        while unvisited:
            current = route[-1]
            next_city = min(unvisited, key=lambda city: self.get_distance(current, city))
            route.append(next_city)
            unvisited.remove(next_city)
            
        return route
    
    def value_based_route(self, cities):
        """Tworzy trasę opartą na wartości przedmiotów w miastach."""
        city_values = {}
        for city in cities:
            total_value = 0
            item_count = 0
            for item in self.itemset.get(city, []):
                _, profit, _ = item
                total_value += profit
                item_count += 1
            city_values[city] = total_value / max(1, item_count)
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
        
        item_ratios = []
        for city, city_items in self.itemset.items():
            for item in city_items:
                item_id, profit, weight = item
                ratio = profit / weight if weight > 0 else 0
                item_ratios.append((item_id, ratio))
        item_ratios.sort(key=lambda x: x[1], reverse=True)
        
        current_weight = 0
        for item_id, _ in item_ratios:
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
        
        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            
            distance = self.get_distance(current_city, next_city)
            
            current_weight = weights_at_cities[i]
            speed = self.Vmax - current_weight * self.v_w
            
            total_time += distance / speed

        last_city = route[-1]
        start_city = route[0]
        
        return_distance = self.get_distance(last_city, start_city)
        
        last_weight = weights_at_cities[-1]
        return_speed = self.Vmax - last_weight * self.v_w
        total_time += return_distance / return_speed
        
        return total_time

    def fitness(self, chromosome):
        """Oblicza funkcję celu dla danego chromosomu."""
        route, items = chromosome
        total_distance = 0
        total_profit = 0
        current_weight = 0
        weights_at_cities = [0]
        picked_items = []

        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            total_distance += self.get_distance(current_city, next_city)

        total_distance += self.get_distance(route[-1], route[0])

        for i, current_city in enumerate(route):
            for item in self.itemset.get(current_city, []):
                item_id, profit, weight = item
                if items[item_id - 1]:
                    if current_weight + weight <= self.W:
                        picked_items.append((current_city, item_id))
                        current_weight += weight
                        total_profit += profit
            
            weights_at_cities.append(current_weight)

        travel_time = self.calculate_travel_time(route, weights_at_cities)
        travel_cost = self.R * travel_time
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

        if random.random() < 0.5:
            start, end = sorted(random.sample(range(len(route1)), 2))
            child_route = [-1] * len(route1)
            child_route[start:end] = route1[start:end]
            
            j = end
            for i in range(len(route2)):
                if route2[i] not in child_route:
                    if j == len(child_route):
                        j = 0
                    child_route[j] = route2[i]
                    j += 1
        else:
            start, end = sorted(random.sample(range(len(route1)), 2))
            child_route = [-1] * len(route1)
            child_route[start:end] = route1[start:end]
            
            mapping = {}
            for i in range(start, end):
                mapping[route1[i]] = route2[i]
            
            for i in range(len(child_route)):
                if i < start or i >= end:
                    city = route2[i]
                    while city in mapping:
                        if city in mapping:
                            city = mapping[city]
                        else:
                            remaining_cities = [c for c in self.graph.keys() if c not in child_route]
                            if remaining_cities:
                                city = random.choice(remaining_cities)
                            else:
                                break
                    if city not in child_route:
                        child_route[i] = city

        crossover_point = random.randint(0, min(len(items1), len(items2)) - 1)
        child_items = items1[:crossover_point] + items2[crossover_point:]

        return (child_route, child_items)

    def mutate(self, chromosome):
        """Mutacja trasy i wektora przedmiotów."""
        route, items = chromosome
        
        if random.random() < self.mutation_rate:
            mutation_type = random.random()
            
            if mutation_type < 0.33:
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]
            elif mutation_type < 0.66:
                start, end = sorted(random.sample(range(len(route)), 2))
                route[start:end+1] = reversed(route[start:end+1])
            else:
                city = route.pop(random.randint(0, len(route) - 1))
                route.insert(random.randint(0, len(route)), city)

        if random.random() < self.mutation_rate:
            num_mutations = random.randint(1, min(3, len(items)))
            for _ in range(num_mutations):
                idx = random.randint(0, len(items) - 1)
                items[idx] = 1 - items[idx]

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
            current_best = max(self.population, key=lambda x: self.fitness(x)[0])
            current_fitness, _, _, _, _, _, _ = self.fitness(current_best)
            self.best_fitness_history.append(current_fitness)
            
            if current_fitness > best_fitness_so_far:
                best_fitness_so_far = current_fitness
                best_solution = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count > 20:
                self.mutation_rate = min(0.5, self.mutation_rate * 1.5)
                no_improvement_count = 0
        
        if best_solution is None:
            best_solution = max(self.population, key=lambda x: self.fitness(x)[0])
            
        return best_solution

files = ["data/50_1.txt", "data/280_1.txt", "data/500_1.txt"]
good_parameters = {
    "population_size": 1000,
    "mutation_rate": 0.1,
    "crossover_rate": 0.9,
    "num_generations": 200
}

weak_parameters = {
    "population_size": 50,
    "mutation_rate": 0.1,
    "crossover_rate": 0.9,
    "num_generations": 50
}

output_dir = "tests/output/Algorytm Genetyczny"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Utworzono katalog: {output_dir}")

optimal_results = []
efficiency_results = []
stability_results = {file: [] for file in files}
memory_results = []

for file in files:
    print(f"\nPrzetwarzanie pliku: {file}")
    graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)
    v_w = (max_speed - min_speed) / knapsack_capacity
    
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
    
    print("\nTestowanie efektywności czasowej...")
    efficiency_results.append({
        "Instancja problemu": file,
        "Parametry": "Dobre",
        "Czas wykonania (s)": execution_time
    })
    
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
    print("Uruchamianie testów algorytmu genetycznego...")