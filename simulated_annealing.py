import random
import math

# Definicje grafu i zestawu przedmiotów
graph = {
    1: [(2, 200), (3, 245), (16, 225)], # Białystok
    2: [(1, 200), (3, 175), (5, 178), (10, 136), (14, 272), (16, 216)], # Warszawa
    3: [(1, 245), (2, 175), (4, 178)],  # Lublin
    4: [(3, 178), (5, 157), (6, 167)],  # Rzeszów
    5: [(2, 178), (4, 157), (6, 167), (10, 154)], # Kielce
    6: [(4, 167), (5, 167), (7, 81)], # Kraków
    7: [(6, 81), (8, 117), (10, 205)], # Katowice 
    8: [(7, 117), (9, 98), (10, 222)], # Opole
    9: [(8, 98), (10, 222), (11, 184), (12, 187)], # Wrocław
    10: [(2, 136), (5, 154), (7, 205), (8, 222), (9, 222), (11, 216), (14, 217)],  # Łódź
    11: [(9, 184), (10, 216), (12, 153), (13, 266), (14, 138)],  # Poznań
    12: [(9, 187), (11, 153), (13, 213)], # Zielona Góra
    13: [(11, 266), (12, 213), (14, 259), (15, 376)], # Szczecin
    14: [(2, 272), (10, 217), (11, 138), (13, 259), (15, 167), (16, 212)], # Bydgoszcz
    15: [(13, 376), (14, 167), (16, 182)], # Gdańsk
    16: [(1, 225), (2, 216), (14, 212), (15, 182)], # Olsztyn
}

itemset = {
    1: [300, 50, (2, 6, 9, 10, 13)],  # wartość 300, waga 50, miasta: Warszawa, Kraków, Wrocław, Łódź, Szczecin
    2: [600, 25, (2, 6, 10)],         # wartość 600, waga 25, miasta: Warszawa, Kraków, Łódź
    3: [200, 100, (1, 11, 14, 16)],   # wartość 200, waga 100, miasta: Białystok, Poznań, Bydgoszcz, Olsztyn
    4: [100, 10, (3, 4, 12, 16)],     # wartość 100, waga 10, miasta: Lublin, Rzeszów, Zielona Góra, Olsztyn
    5: [250, 75, (5, 7, 8)],          # wartość 250, waga 75, miasta: Kielce, Katowice, Opole
    6: [350, 60, (2, 6, 11, 14)],     # wartość 350, waga 60, miasta: Warszawa, Kraków, Poznań, Bydgoszcz
    7: [150, 20, (4, 7, 8, 16)],      # wartość 150, waga 20, miasta: Rzeszów, Katowice, Opole, Olsztyn
    8: [500, 30, (2, 6, 10, 15)],     # wartość 500, waga 30, miasta: Warszawa, Kraków, Łódź, Gdańsk
    9: [120, 15, (3, 5, 12, 16)],     # wartość 120, waga 15, miasta: Lublin, Kielce, Zielona Góra, Olsztyn
    10: [180, 40, (1, 3, 11, 13)],    # wartość 180, waga 40, miasta: Białystok, Lublin, Poznań, Szczecin
    11: [270, 55, (6, 9, 14, 15)],    # wartość 270, waga 55, miasta: Kraków, Wrocław, Bydgoszcz, Gdańsk
}
Vmax = 15
Vmin = 5
R = 0.1
W = 500

class Solution:
    def __init__(self, route, items):
        self.route = route
        self.items = items
        self.fitness = 0

def fitness(route, items):
    distance = 0
    for i in range(len(route) - 1):
        if not any(dest == route[i + 1] for dest, dist in graph[route[i]]):
            return 0  # Invalid route, return 0 fitness
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
        distances[-1] = 1  # Avoid division by zero

    finalitemset = []
    time = distances[-1] * 2 * (Vmax + Vmin)
    
    for key, value in itemset.items():
        for city in value[2]:
            if city in route:
                route_index = route.index(city)
                if distances[route_index] == 0:
                    distances[route_index] = 1  # Avoid division by zero
                score = int(value[0] - (0.25 * value[0] * (distances[route_index] / distances[-1])) - (R * time * value[1] / W))
                finalitemset.append([key, city, value[1], score, value[0]])

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

class GreedyAlgorithm:
    def __init__(self):
        self.visited = [0] * len(graph)
        self.shortest_path = []
        self.path_length = 0

    def path(self, i):
        self.shortest_path += [i]
        self.visited[i - 1] = 1
        min_distance = float("inf")
        next_city = None
        for edge in graph[i]:
            if self.visited[edge[0] - 1] == 0 and edge[1] < min_distance:
                min_distance = edge[1]
                next_city = edge[0]
        if next_city:
            self.path_length += min_distance
            self.path(next_city)

    def run(self):
        self.path(1)  # Start from city 1
        return self.shortest_path, self.path_length

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
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    return total_distance

# Parametry optymalizacji

# Greedy Algorithm
# - brak dodatkowych parametrów

# Simulated Annealing
initial_temperature = 10000
cooling_rate = 0.95
stopping_temperature = 1
stopping_iter = 1000

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

# Testowanie algorytmów

# Greedy Algorithm
print("Uruchamianie algorytmu zachłannego...")
greedy = GreedyAlgorithm()
best_route_greedy, total_distance_greedy = greedy.run()
picked_items_greedy, total_profit_greedy, total_weight_greedy = solve_knapsack(best_route_greedy)
print_solution(best_route_greedy, total_distance_greedy, picked_items_greedy, total_profit_greedy, total_weight_greedy)

# Simulated Annealing
print("\nUruchamianie algorytmu symulowanego wyżarzania...")
initial_route = generate_random_route()  # Użycie losowej trasy jako początkowej
print(f"Initial route: {initial_route}")
sa = SimulatedAnnealing(initial_route=initial_route, temp=initial_temperature, alpha=cooling_rate, stopping_temp=stopping_temperature, stopping_iter=stopping_iter)
best_route_sa, best_fitness_sa = sa.anneal()
total_distance_sa = calculate_total_distance(best_route_sa)
picked_items_sa, total_profit_sa, total_weight_sa = solve_knapsack(best_route_sa)
print_solution(best_route_sa, total_distance_sa, picked_items_sa, total_profit_sa, total_weight_sa)
