import random
from common.data_loader import load_data
from functools import lru_cache

graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/50_1.txt")

Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio
v_w = (Vmax - Vmin) / W

def create_distance_matrix(graph, num_cities):
    """Tworzy macierz odległości dla szybszego dostępu."""
    dist_matrix = [[0] * (num_cities + 1) for _ in range(num_cities + 1)]
    for city in range(1, num_cities + 1):
        for dest, dist in graph[city]:
            dist_matrix[city][dest] = dist
    return dist_matrix

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
        self.weights_at_cities = []

    @lru_cache(maxsize=128)
    def get_distance(self, city1, city2):
        """Pobiera odległość między miastami z macierzy odległości."""
        return distance_matrix[city1][city2]

    def select_next_city(self, current_city, pheromone):
        probabilities = []
        for dest in range(1, self.num_cities + 1):
            if dest not in self.visited and dest != current_city:
                dist = self.get_distance(current_city, dest)
                if dist > 0:
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

    def calculate_remaining_distance(self, current_city_idx):
        """Oblicza pozostałą odległość do przejechania."""
        remaining_distance = 0
        
        for i in range(current_city_idx, len(self.route) - 1):
            current = self.route[i]
            next_city = self.route[i + 1]
            remaining_distance += self.get_distance(current, next_city)
        
        if current_city_idx == len(self.route) - 1:
            last_city = self.route[-1]
            start_city = self.route[0]
            remaining_distance += self.get_distance(last_city, start_city)
        
        return remaining_distance

    def pick_items(self, current_city):
        """Zoptymalizowana wersja wyboru przedmiotów."""
        items = itemset.get(current_city, [])
        if not items:
            return
            
        current_city_idx = self.route.index(current_city)
        remaining_distance = self.calculate_remaining_distance(current_city_idx)
        
        speed_without_item = max(Vmin, Vmax - self.current_weight * v_w)
        
        effective_items = []
        for item_id, profit, weight in items:
            if self.current_weight + weight <= W:
                speed_with_item = max(Vmin, Vmax - (self.current_weight + weight) * v_w)
                
                time_without_item = remaining_distance / speed_without_item
                time_with_item = remaining_distance / speed_with_item
                
                additional_time = time_with_item - time_without_item
                transport_cost = additional_time * R
                effective_profit = profit - transport_cost
                
                if effective_profit > 0:
                    effective_items.append((item_id, profit, weight, effective_profit))
        
        effective_items.sort(key=lambda x: x[3] / x[2], reverse=True)
        
        for item_id, profit, weight, _ in effective_items:
            if self.current_weight + weight <= W:
                self.items_picked.append((current_city, item_id))
                self.total_profit += profit
                self.current_weight += weight

    def calculate_travel_time(self):
        """Zoptymalizowane obliczanie czasu podróży."""
        total_time = 0
        
        for i in range(len(self.route) - 1):
            current_city = self.route[i]
            next_city = self.route[i + 1]
            distance = self.get_distance(current_city, next_city)
            current_weight = self.weights_at_cities[i]
            speed = max(Vmin, Vmax - current_weight * v_w)
            total_time += distance / speed
        
        last_city = self.route[-1]
        start_city = self.route[0]
        return_distance = self.get_distance(last_city, start_city)
        last_weight = self.weights_at_cities[-1]
        return_speed = max(Vmin, Vmax - last_weight * v_w)
        total_time += return_distance / return_speed
        
        return total_time

    def calculate_objective_value(self):
        """Oblicza wartość funkcji celu."""
        travel_time = self.calculate_travel_time()
        travel_cost = R * travel_time
        objective_value = self.total_profit - travel_cost
        return objective_value, travel_time, travel_cost

    def travel(self, pheromone):
        self.route = [random.randint(1, self.num_cities)]
        self.visited = set(self.route)
        self.distance_traveled = 0
        self.weights_at_cities = [0]

        while len(self.route) < self.num_cities:
            current_city = self.route[-1]
            self.pick_items(current_city)
            self.weights_at_cities.append(self.current_weight)
            
            next_city = self.select_next_city(current_city, pheromone)
            if next_city is None:
                self.distance_traveled = float('inf')
                return
            self.route.append(next_city)
            self.visited.add(next_city)
            self.distance_traveled += self.get_distance(current_city, next_city)

        start_city = self.route[0]
        last_city = self.route[-1]
        self.distance_traveled += self.get_distance(last_city, start_city)
        
        self.objective_value, self.travel_time, self.travel_cost = self.calculate_objective_value()

class ACO:
    def __init__(self, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit

    def initialize_pheromone(self, num_cities):
        return {i: {j: 1.0 for j in range(1, num_cities + 1) if i != j} for i in range(1, num_cities + 1)}

    def run(self):
        num_cities = len(graph)
        global distance_matrix
        distance_matrix = create_distance_matrix(graph, num_cities)
        pheromone = self.initialize_pheromone(num_cities)
        best_route = None
        best_distance = float('inf')
        best_items = []
        best_profit = 0
        best_time = float('inf')
        best_total_weight = 0
        best_objective_value = float('-inf')
        best_travel_cost = 0

        for iteration in range(self.num_iterations):
            ants = [Ant(self.alpha, self.beta, num_cities) for _ in range(self.num_ants)]
            for ant in ants:
                ant.travel(pheromone)
                if ant.distance_traveled == float('inf'):
                    continue

                if ant.objective_value > best_objective_value:
                    best_objective_value = ant.objective_value
                    best_profit = ant.total_profit
                    best_route = ant.route
                    best_distance = ant.distance_traveled
                    best_items = ant.items_picked
                    best_time = ant.travel_time
                    best_total_weight = ant.current_weight
                    best_travel_cost = ant.travel_cost

            for i in pheromone:
                for j in pheromone[i]:
                    pheromone[i][j] *= (1 - self.evaporation_rate)

            for ant in ants:
                if ant.distance_traveled < float('inf') and ant.objective_value > float('-inf'):
                    delta = self.pheromone_deposit * ant.objective_value
                    for i in range(len(ant.route) - 1):
                        city1, city2 = ant.route[i], ant.route[i + 1]
                        pheromone[city1][city2] += delta
                        pheromone[city2][city1] += delta
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteracja {iteration + 1}/{self.num_iterations}, Najlepsza wartość funkcji celu: {best_objective_value:.2f}")

        return best_route, best_distance, best_items, best_profit, best_total_weight, best_time, best_objective_value, best_travel_cost

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time, objective_value, travel_cost):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Koszt podróży: {:.2f}".format(travel_cost))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk z przedmiotów: {:.2f}".format(total_profit))
    print("Całkowita waga przedmiotów: ", total_weight)
    print("Wartość funkcji celu: {:.2f}".format(objective_value))

num_ants = 100
num_iterations = 300
alpha = 1.0
beta = 1.0
evaporation_rate = 0.9
pheromone_deposit = 1000

aco = ACO(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit)
best_route, best_distance, best_items, best_profit, best_total_weight, best_time, best_objective_value, best_travel_cost = aco.run()
print_solution(best_route, best_distance, best_items, best_profit, best_total_weight, best_time, best_objective_value, best_travel_cost)
