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

class Ant:
    def __init__(self, alpha, beta, num_cities):
        self.alpha = alpha
        self.beta = beta
        self.num_cities = num_cities
        self.route = []
        self.visited = set()
        self.distance_traveled = 0

    def select_next_city(self, current_city, pheromone, distances):
        probabilities = []
        for city, dist in distances[current_city]:
            if city not in self.visited:
                tau = pheromone[current_city][city] ** self.alpha
                eta = (1.0 / dist) ** self.beta
                probabilities.append((city, tau * eta))

        if not probabilities:
            return None

        total = sum(prob for _, prob in probabilities)
        probabilities = [(city, prob / total) for city, prob in probabilities]
        r = random.random()
        for city, prob in probabilities:
            r -= prob
            if r <= 0:
                return city
        return probabilities[-1][0]

    def travel(self, pheromone, distances):
        self.route = [random.randint(1, self.num_cities)]
        self.visited = set(self.route)
        self.distance_traveled = 0

        while len(self.route) < self.num_cities:
            current_city = self.route[-1]
            next_city = self.select_next_city(current_city, pheromone, distances)
            if next_city is None:
                break
            self.route.append(next_city)
            self.visited.add(next_city)
            self.distance_traveled += next(dist for city, dist in distances[current_city] if city == next_city)

        if len(self.route) < self.num_cities:
            self.distance_traveled = float('inf')

class ACO:
    def __init__(self, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit

    def initialize_pheromone(self, num_cities):
        pheromone = {}
        for i in range(1, num_cities + 1):
            pheromone[i] = {}
            for j in range(1, num_cities + 1):
                if i != j:
                    pheromone[i][j] = 1.0
        return pheromone

    def run(self, distances, num_cities):
        pheromone = self.initialize_pheromone(num_cities)
        best_route = None
        best_distance = float('inf')

        for _ in range(self.num_iterations):
            ants = [Ant(self.alpha, self.beta, num_cities) for _ in range(self.num_ants)]
            for ant in ants:
                ant.travel(pheromone, distances)
                if ant.distance_traveled < best_distance:
                    best_distance = ant.distance_traveled
                    best_route = ant.route

            for i in range(1, num_cities + 1):
                for j in range(1, num_cities + 1):
                    if i != j:
                        pheromone[i][j] *= (1 - self.evaporation_rate)

            for ant in ants:
                if ant.distance_traveled < float('inf'):
                    for i in range(len(ant.route) - 1):
                        city1 = ant.route[i]
                        city2 = ant.route[i + 1]
                        pheromone[city1][city2] += self.pheromone_deposit / ant.distance_traveled
                        pheromone[city2][city1] += self.pheromone_deposit / ant.distance_traveled

        return best_route, best_distance

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

# Parametry ACO
num_ants = 20
num_iterations = 100
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
pheromone_deposit = 100

# Testowanie algorytmów

# ACO
print("Uruchamianie algorytmu ACO...")
aco = ACO(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit)
best_route_aco, best_distance_aco = aco.run(graph, len(graph))
picked_items_aco, total_profit_aco, total_weight_aco = solve_knapsack(best_route_aco)
print_solution(best_route_aco, best_distance_aco, picked_items_aco, total_profit_aco, total_weight_aco)
