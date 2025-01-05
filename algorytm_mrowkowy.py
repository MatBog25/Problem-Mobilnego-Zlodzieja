import random
import math
from common.data_loader import load_data

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/33810_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

class Ant:
    def __init__(self, alpha, beta, num_cities):
        self.alpha = alpha
        self.beta = beta
        self.num_cities = num_cities
        self.route = []
        self.visited = set()
        self.items_picked = []
        self.distance_traveled = 0
        self.total_profit = 0
        self.current_weight = 0

    def select_next_city(self, current_city, pheromone):
        probabilities = []
        for dest, dist in graph[current_city]:
            if dest not in self.visited:
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

    def pick_items(self, current_city):
        for item in itemset.get(current_city, []):
            item_id, profit, weight = item
            if self.current_weight + weight <= W:  # Sprawdź pojemność plecaka
                self.items_picked.append(item_id)
                self.total_profit += profit
                self.current_weight += weight

    def travel(self, pheromone):
        self.route = [random.randint(1, self.num_cities)]
        self.visited = set(self.route)
        self.distance_traveled = 0

        while len(self.route) < self.num_cities:
            current_city = self.route[-1]
            self.pick_items(current_city)  # Wybór przedmiotów w aktualnym mieście
            next_city = self.select_next_city(current_city, pheromone)
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
        best_items = []
        best_profit = 0
        best_time = float('inf')

        for _ in range(self.num_iterations):
            ants = [Ant(self.alpha, self.beta, num_cities) for _ in range(self.num_ants)]
            for ant in ants:
                ant.travel(pheromone)
                time = ant.distance_traveled / (Vmax - (ant.current_weight / W) * (Vmax - Vmin))
                cost = R * time
                profit_with_cost = ant.total_profit - cost

                if ant.distance_traveled < best_distance and profit_with_cost > best_profit:
                    best_distance = ant.distance_traveled
                    best_route = ant.route
                    best_items = ant.items_picked
                    best_profit = profit_with_cost
                    best_time = time

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

        return best_route, best_distance, best_items, best_profit, best_time

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Całkowity zysk po uwzględnieniu kosztu wynajmu: {:.2f}".format(total_profit))
    print("Złodziej powinien zabrać następujące przedmioty:", picked_items)
    print("Całkowita waga przedmiotów: ", total_weight)

# Parametry ACO
num_ants = 10
num_iterations = 10
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
pheromone_deposit = 10

# Testowanie algorytmu ACO
print("Uruchamianie algorytmu ACO...")
aco = ACO(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit)
best_route_aco, best_distance_aco, best_items_aco, best_profit_aco, best_time_aco = aco.run()

# Oblicz wagę wszystkich przedmiotów
total_weight = 0
for city, items in itemset.items():
    for item_id, profit, weight in items:
        if item_id in best_items_aco:
            total_weight += weight

print_solution(best_route_aco, best_distance_aco, best_items_aco, best_profit_aco, total_weight, best_time_aco)
