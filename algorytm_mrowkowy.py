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

def fitness(route, items):
    distance = calculate_total_distance(route)
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

def calculate_total_distance(route):
    """Oblicza całkowitą odległość dla podanej trasy."""
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    # Dodaj odległość powrotu do miasta startowego (jeśli wymagane)
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    return total_distance

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
        for dest, dist in graph[current_city]:
            if dest not in self.visited and dist > 0:  # Ignorujemy zerowe odległości
                tau = pheromone[current_city][dest] ** self.alpha
                eta = (1.0 / dist) ** self.beta
                probabilities.append((dest, tau * eta))

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
            self.distance_traveled += next(dist for dest, dist in graph[current_city] if dest == next_city)

        # Dodaj powrót do miasta początkowego
        if len(self.route) == self.num_cities:
            start_city = self.route[0]
            last_city = self.route[-1]
            self.route.append(start_city)
            self.distance_traveled += next(dist for dest, dist in graph[last_city] if dest == start_city)

        if len(self.route) < self.num_cities + 1:
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

    def run(self):
        num_cities = len(graph)
        pheromone = self.initialize_pheromone(num_cities)
        best_route = None
        best_distance = float('inf')

        for _ in range(self.num_iterations):
            ants = [Ant(self.alpha, self.beta, num_cities) for _ in range(self.num_ants)]
            for ant in ants:
                ant.travel(pheromone, graph)
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

def print_solution(route, total_distance, picked_items, total_profit, total_weight):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Złodziej powinien zabrać następujące przedmioty:")
    for i in picked_items:
        print("Miasto : " + str(i[0]) + "   Przedmioty : " + ', '.join(str(e) for e in i[1]))
    print("Całkowity zysk : " + str(total_profit))
    print("Waga przenoszona w plecaku : " + str(total_weight))

# Parametry ACO
num_ants = 100
num_iterations = 1000
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
pheromone_deposit = 200

# Testowanie algorytmu ACO
print("Uruchamianie algorytmu ACO...")
aco = ACO(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit)
best_route_aco, best_distance_aco = aco.run()
picked_items_aco, total_profit_aco, total_weight_aco = solve_knapsack(best_route_aco)
print_solution(best_route_aco, best_distance_aco, picked_items_aco, total_profit_aco, total_weight_aco)
