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

class Particle:
    def __init__(self, route):
        self.route = route
        self.best_route = list(route)
        self.best_fitness = fitness(route, [])
        self.velocity = []

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
        if is_valid_route(new_route):
            self.route = new_route
        new_fitness = fitness(self.route, [])
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_route = list(self.route)

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

def is_valid_route(route):
    for i in range(len(route) - 1):
        if not any(dest == route[i + 1] for dest, dist in graph[route[i]]):
            return False
    return True

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
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    return total_distance

# Parametry PSO
num_particles = 30
w = 0.5
c1 = 1.5
c2 = 1.5
num_iterations = 1000

# Testowanie algorytmów

# PSO
print("Uruchamianie algorytmu PSO...")
pso = PSO(num_particles, w, c1, c2, num_iterations)
best_route_pso, best_fitness_pso = pso.run()
total_distance_pso = calculate_total_distance(best_route_pso)
picked_items_pso, total_profit_pso, total_weight_pso = solve_knapsack(best_route_pso)
print_solution(best_route_pso, total_distance_pso, picked_items_pso, total_profit_pso, total_weight_pso)
