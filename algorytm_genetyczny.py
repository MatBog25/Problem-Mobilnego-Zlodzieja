from common.data_loader import load_data
import random
import math

graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/280_1.txt")

Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio
v_w = (Vmax - Vmin) / W

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, num_generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.population = []
        self.best_fitness_history = []

    def initialize_population(self):
        """Inicjalizuje populację losowymi trasami i wektorami wyboru przedmiotów."""
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
            for item in itemset.get(city, []):
                _, profit, _ = item
                total_value += profit
                item_count += 1
            city_values[city] = total_value / max(1, item_count)
        
        sorted_cities = sorted(cities, key=lambda city: city_values[city], reverse=True)
        return sorted_cities
    
    def get_distance(self, city1, city2):
        """Zwraca odległość między dwoma miastami."""
        for dest, dist in graph[city1]:
            if dest == city2:
                return dist
        return float('inf')
    
    def initialize_items_based_on_value_weight_ratio(self):
        """Inicjalizuje wektor przedmiotów z preferencją dla przedmiotów o wysokim stosunku wartości do wagi."""
        items = [0] * len(itemset)
        
        item_ratios = []
        for city, city_items in itemset.items():
            for item in city_items:
                item_id, profit, weight = item
                ratio = profit / weight if weight > 0 else 0
                item_ratios.append((item_id, ratio))
        
        item_ratios.sort(key=lambda x: x[1], reverse=True)
        
        current_weight = 0
        for item_id, _ in item_ratios:
            for city_items in itemset.values():
                for item in city_items:
                    if item[0] == item_id:
                        _, _, weight = item
                        if current_weight + weight <= W:
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
            for dest, dist in graph[current_city]:
                if dest == next_city:
                    total_distance += dist
                    break

        for dest, dist in graph[route[-1]]:
            if dest == route[0]:
                total_distance += dist
                break

        for i, current_city in enumerate(route):
            for item in itemset.get(current_city, []):
                item_id, profit, weight = item
                if items[item_id - 1]:
                    if current_weight + weight <= W:
                        picked_items.append((current_city, item_id))
                        current_weight += weight
                        total_profit += profit
            
            weights_at_cities.append(current_weight)

        travel_time = self.calculate_travel_time(route, weights_at_cities)
        travel_cost = R * travel_time
        fitness = total_profit - travel_cost
        
        return fitness, picked_items, current_weight, total_profit, travel_time, travel_cost

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
                        city = mapping[city]
                    child_route[i] = city

        crossover_point = random.randint(0, len(items1) - 1)
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
            num_mutations = random.randint(1, 3)
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

    def run(self):
        """Uruchamia algorytm genetyczny."""
        self.initialize_population()
        best_fitness_so_far = float('-inf')
        no_improvement_count = 0
        best_solution = None
        
        for generation in range(self.num_generations):
            self.evolve()
            
            current_best = max(self.population, key=lambda x: self.fitness(x)[0])
            current_fitness, _, _, _, _, _ = self.fitness(current_best)
            
            self.best_fitness_history.append(current_fitness)
            
            if current_fitness > best_fitness_so_far:
                best_fitness_so_far = current_fitness
                best_solution = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if (generation + 1) % 10 == 0:
                print(f"Generacja {generation + 1}/{self.num_generations}, Najlepsza wartość funkcji celu: {best_fitness_so_far:.2f}")
            
            if no_improvement_count > 20:
                print("Brak poprawy przez 20 generacji, zwiększam wskaźnik mutacji...")
                self.mutation_rate = min(0.5, self.mutation_rate * 1.5)
                no_improvement_count = 0
            
            if best_fitness_so_far > 0 and generation > 50:
                print(f"Znaleziono rozwiązanie z dodatnią wartością funkcji celu: {best_fitness_so_far:.2f}")
                break
        
        if best_solution is None:
            best_solution = max(self.population, key=lambda x: self.fitness(x)[0])
            
        return best_solution

population_size = 40
mutation_rate = 0.1
crossover_rate = 0.9
num_generations = 50

print("Uruchamianie algorytmu genetycznego...")
ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, num_generations)
best_solution = ga.run()
best_route, best_items = best_solution
best_fitness, picked_items, total_weight, total_profit, travel_time, travel_cost = ga.fitness(best_solution)

from collections import defaultdict

def calculate_total_distance(route):
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

total_distance = calculate_total_distance(best_route)

print("Najlepsza trasa:", best_route)
print("Całkowita odległość:", total_distance)
print("Czas podróży: {:.2f} jednostek czasu".format(travel_time))
print("Koszt podróży: {:.2f}".format(travel_cost))
print("Złodziej powinien zabrać następujące przedmioty:")
for city, item in picked_items:
    print(f"Miasto {city}: Przedmiot {item}")
print("Całkowity zysk z przedmiotów: {:.2f}".format(total_profit))
print("Waga przenoszona w plecaku:", total_weight)
print("Wartość funkcji celu: {:.2f}".format(best_fitness))
