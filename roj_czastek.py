import random
import math
from common.data_loader import load_data  # Wcześniej zaimplementowana funkcja

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/5miast.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

class Particle:
    def __init__(self, route):
        self.route = self.ensure_all_cities(route)
        self.best_route = list(self.route)
        self.best_fitness = fitness(self.route, [])
        self.velocity = []

    def ensure_all_cities(self, route):
        """Zapewnia, że trasa zawiera wszystkie miasta."""
        all_cities = set(graph.keys())
        missing_cities = list(all_cities - set(route))
        route += missing_cities  # Dodaj brakujące miasta
        random.shuffle(route)  # Losowo permutuj trasę
        return route

    def update_velocity(self, global_best_route, w, c1, c2):
        self.velocity = []
        for i in range(len(self.route)):
            r1 = random.random()
            r2 = random.random()
            if r1 < c1 and self.route[i] in self.best_route:
                self.velocity.append(('swap', i, self.best_route.index(self.route[i])))
            if r2 < c2 and self.route[i] in global_best_route:
                self.velocity.append(('swap', i, global_best_route.index(self.route[i])))

    def update_position(self):
        new_route = list(self.route)
        for move in self.velocity:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                if i < len(new_route) and j < len(new_route):
                    new_route[i], new_route[j] = new_route[j], new_route[i]
        self.route = self.ensure_all_cities(new_route)  # Uzupełnij brakujące miasta
        new_fitness = fitness(self.route, [])
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_route = list(self.route)

def fitness(route, items):
    distance = 0
    for i in range(len(route) - 1):
        if not any(dest == route[i + 1] for dest, dist in graph[route[i]]):
            return 0  # Niepoprawna trasa
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                distance += dist
                break

    total_weight = sum(itemset[item][1] for item in items if item != -1)
    total_value = sum(itemset[item][0] for item in items if item != -1)
    time = distance / ((Vmax + Vmin) / 2)

    if total_weight > W:
        return 0
    return total_value - R * time

def solve_knapsack(route):
    """Rozwiązuje problem plecakowy dla podanej trasy."""
    distances = [0] * len(route)
    for i in range(1, len(route)):
        for dest, dist in graph[route[i - 1]]:
            if dest == route[i]:
                distances[i] = distances[i - 1] + dist
                break

    if distances[-1] == 0:
        distances[-1] = 1  # Zapobieganie dzieleniu przez zero

    finalitemset = []
    time = distances[-1] * 2 * (Vmax + Vmin)
    
    for key, value in itemset.items():
        for item in value:
            item_id, profit, weight = item
            for city in route:
                if city == item_id:  # Sprawdzamy, czy miasto jest na trasie
                    route_index = route.index(city)
                    if distances[route_index] == 0:
                        distances[route_index] = 1  # Zapobieganie dzieleniu przez zero
                    score = int(profit - (0.25 * profit * (distances[route_index] / distances[-1])) - (R * time * weight / W))
                    finalitemset.append([item_id, city, weight, score, profit])

    finalitemset.sort(key=lambda x: int(x[3]), reverse=True)

    wc = 0
    picked_items = []
    totalprof = 0

    for item in finalitemset:
        if wc + item[2] <= W:
            picked_items.append(item)
            wc += item[2]
            totalprof += item[4]

    result = {}
    for item in picked_items:
        if item[1] not in result:
            result[item[1]] = []
        result[item[1]].append(item[0])

    fin = [[city, sorted(items)] for city, items in result.items()]
    fin.sort(key=lambda x: x[0])

    return fin, totalprof, wc

class PSO:
    def __init__(self, num_particles, w, c1, c2, num_iterations):
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_iterations = num_iterations
        self.particles = []

        for _ in range(num_particles):
            route = self.generate_random_route()
            self.particles.append(Particle(route))

        self.global_best_route = self.particles[0].best_route
        self.global_best_fitness = self.particles[0].best_fitness

    def generate_random_route(self):
        cities = list(graph.keys())
        random.shuffle(cities)
        return cities

    def run(self):
        for _ in range(self.num_iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_route, self.w, self.c1, self.c2)
                particle.update_position()
                if particle.best_fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_route = particle.best_route

        return self.global_best_route, self.global_best_fitness

def print_solution(route, total_distance, picked_items, total_profit, total_weight):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Złodziej powinien zabrać następujące przedmioty:")
    for i in picked_items:
        print("Miasto : " + str(i[0]) + "   Przedmioty : " + ', '.join(str(e) for e in i[1]))
    print("Całkowity zysk : " + str(total_profit))
    print("Waga przenoszona w plecaku : " + str(total_weight))

def calculate_total_distance(route):
    """Oblicza całkowitą odległość dla podanej trasy."""
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    # Dodaj odległość powrotu do miasta startowego
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    return total_distance

# Parametry PSO dla dużego problemu
num_particles = 5000  # Więcej cząstek, by lepiej eksplorować przestrzeń
w = 0.7  # Waga bezwładności
c1 = 2.0  # Składnik poznawczy
c2 = 2.0  # Składnik społeczny
num_iterations = 100  # Więcej iteracji

# Testowanie algorytmu PSO
print("Uruchamianie algorytmu PSO...")
pso = PSO(num_particles, w, c1, c2, num_iterations)
best_route_pso, best_fitness_pso = pso.run()
total_distance_pso = calculate_total_distance(best_route_pso)
picked_items_pso, total_profit_pso, total_weight_pso = solve_knapsack(best_route_pso)
print_solution(best_route_pso, total_distance_pso, picked_items_pso, total_profit_pso, total_weight_pso)
