from common.data_loader import load_data  # Wcześniej zaimplementowana funkcja
import random
import math

# Wczytaj dane z pliku za pomocą funkcji load_data, która zwraca graf, przedmioty, i parametry problemu
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/5miast.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, num_generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.population = []

    def initialize_population(self):
        """Inicjalizuje populację losowymi trasami i wektorami wyboru przedmiotów."""
        cities = list(graph.keys())
        for _ in range(self.population_size):
            route = cities[:]
            random.shuffle(route)
            items = [random.choice([0, 1]) for _ in range(len(itemset))]
            self.population.append((route, items))

    def fitness(self, chromosome):
        """Oblicza funkcję celu dla danego chromosomu."""
        route, items = chromosome
        total_distance = 0
        total_profit = 0
        current_weight = 0

        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            for dest, dist in graph[current_city]:
                if dest == next_city:
                    total_distance += dist
                    break

            # Uwzględnij przedmioty w aktualnym mieście
            for item in itemset.get(current_city, []):
                item_id, profit, weight = item
                if items[item_id - 1]:  # Jeśli przedmiot jest wybrany
                    if current_weight + weight <= W:  # Sprawdzenie przed dodaniem przedmiotu
                        current_weight += weight
                        total_profit += profit

        # Dodaj dystans powrotny do miasta startowego
        for dest, dist in graph[route[-1]]:
            if dest == route[0]:
                total_distance += dist
                break

        # Oblicz funkcję celu
        speed = Vmax - (current_weight / W) * (Vmax - Vmin)
        time = total_distance / speed
        return total_profit - R * time

    def select_parent(self):
        """Turniejowy wybór rodziców."""
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: self.fitness(x), reverse=True)
        return tournament[0]

    def crossover(self, parent1, parent2):
        """Krzyżowanie tras i wektorów wyboru przedmiotów."""
        route1, items1 = parent1
        route2, items2 = parent2

        # Krzyżowanie tras
        start, end = sorted(random.sample(range(len(route1)), 2))
        child_route = [-1] * len(route1)
        child_route[start:end] = route1[start:end]
        parent2_index = 0
        for i in range(len(child_route)):
            if child_route[i] == -1:
                while route2[parent2_index] in child_route:
                    parent2_index += 1
                child_route[i] = route2[parent2_index]

        # Krzyżowanie wektorów przedmiotów
        child_items = [random.choice([i1, i2]) for i1, i2 in zip(items1, items2)]

        return (child_route, child_items)

    def mutate(self, chromosome):
        """Mutacja trasy i wektora przedmiotów."""
        route, items = chromosome

        # Mutacja trasy
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]

        # Mutacja wyboru przedmiotów
        if random.random() < self.mutation_rate:
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

    def run(self):
        """Uruchamia algorytm genetyczny."""
        self.initialize_population()
        for _ in range(self.num_generations):
            self.evolve()
        best_solution = max(self.population, key=lambda x: self.fitness(x))
        return best_solution

# Parametry algorytmu genetycznego
population_size = 100
mutation_rate = 0.01
crossover_rate = 0.9
num_generations = 100

# Uruchomienie algorytmu genetycznego
print("Uruchamianie algorytmu genetycznego...")
ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, num_generations)
best_solution = ga.run()
best_route, best_items = best_solution
best_fitness = ga.fitness(best_solution)

# Oblicz dodatkowe dane
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

picked_items = defaultdict(list)
total_profit = 0
total_weight = 0

for city, item_decision in enumerate(best_items):
    for item in itemset.get(city + 1, []):
        item_id, profit, weight = item
        if item_decision and total_weight + weight <= W:  # Upewnij się, że nie przekraczamy limitu
            picked_items[city + 1].append(item_id)
            total_profit += profit
            total_weight += weight

print("Najlepsza trasa:", best_route)
print("Najlepsze przedmioty (dla każdego miasta):")
for city, items in picked_items.items():
    print(f"Miasto {city}: {items}")
print("Całkowita wartość przedmiotów:", total_profit)
print("Całkowita waga przedmiotów:", total_weight)
print("Całkowity dystans trasy:", calculate_total_distance(best_route))
print("Najlepsza wartość funkcji celu:", best_fitness)
