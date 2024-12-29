import random
import math

from common.data_loader import load_data

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/5miast.txt")

# Ustaw parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

class Solution:
    def __init__(self, route, items):
        self.route = route
        self.items = items
        self.fitness = 0

def fitness(route, items):
    distance = 0
    for i in range(len(route) - 1):
        if not any(dest == route[i + 1] for dest, dist in graph[route[i]]):
            return 0  # Jesli niepoprawna trasa zwroc 0
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


class SimulatedAnnealing:
    def __init__(self, initial_route, temp, alpha, stopping_temp, stopping_iter):
        self.current_route = initial_route
        self.best_route = list(initial_route)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.current_fitness = fitness(self.current_route, [])
        self.best_fitness = self.current_fitness

    def acceptance_probability(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.current_fitness) / self.temp)

    def is_valid_route(self, route):
        for i in range(len(route) - 1):
            if not any(dest == route[i + 1] for dest, dist in graph[route[i]]):
                return False
        return True

    def generate_valid_candidate(self):
        while True:
            candidate = list(self.current_route)
            l = random.randint(2, len(candidate) - 1)
            i = random.randint(0, len(candidate) - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
            if self.is_valid_route(candidate):
                return candidate

    def accept(self, candidate):
        candidate_fitness = fitness(candidate, [])
        if candidate_fitness > self.current_fitness:
            self.current_fitness = candidate_fitness
            self.current_route = candidate
            if candidate_fitness > self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_route = candidate
        else:
            if random.random() < self.acceptance_probability(candidate_fitness):
                self.current_fitness = candidate_fitness
                self.current_route = candidate

    def anneal(self):
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = self.generate_valid_candidate()
            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1

        return self.best_route, self.best_fitness

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

# Generowanie losowej trasy początkowej
def generate_random_route():
    cities = list(graph.keys())
    current_city = 1
    route = [current_city]
    while len(route) < len(cities):
        neighbors = [dest for dest, dist in graph[current_city] if dest not in route]
        if not neighbors:
            break
        next_city = random.choice(neighbors)
        route.append(next_city)
        current_city = next_city
    return route

# Parametry optymalizacji

initial_temperature = 10000000
cooling_rate = 0.95
stopping_temperature = 1
stopping_iter = 10000

# Testowanie algorytmów

print("\nUruchamianie algorytmu symulowanego wyżarzania...")
initial_route = generate_random_route()  # Użycie losowej trasy jako początkowej
sa = SimulatedAnnealing(initial_route=initial_route, temp=initial_temperature, alpha=cooling_rate, stopping_temp=stopping_temperature, stopping_iter=stopping_iter)
best_route_sa, best_fitness_sa = sa.anneal()
total_distance_sa = calculate_total_distance(best_route_sa)
picked_items_sa, total_profit_sa, total_weight_sa = solve_knapsack(best_route_sa)
print_solution(best_route_sa, total_distance_sa, picked_items_sa, total_profit_sa, total_weight_sa)
