import random
import math

from common.data_loader import load_data

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/280_1.txt")

# Ustaw parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

class SimulatedAnnealing:
    def __init__(self, initial_route, temp, alpha, stopping_temp, stopping_iter):
        self.current_route = initial_route
        self.best_route = list(initial_route)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.current_fitness, self.current_items, self.current_weight, self.current_profit, self.current_speed = self.calculate_fitness(self.current_route)
        self.best_fitness = self.current_fitness
        self.best_items = self.current_items
        self.best_weight = self.current_weight
        self.best_profit = self.current_profit
        self.best_speed = self.current_speed

    def calculate_fitness(self, route):
        total_distance = 0
        total_profit = 0
        total_weight = 0
        picked_items = []

        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            for dest, dist in graph[current_city]:
                if dest == next_city:
                    total_distance += dist
                    break

            # Wybór przedmiotów w aktualnym mieście
            for item in itemset.get(current_city, []):
                item_id, profit, weight = item
                if total_weight + weight <= W:
                    picked_items.append((current_city, item_id))
                    total_weight += weight
                    total_profit += profit

        # Dodaj dystans powrotny do miasta startowego
        start_city = route[0]
        last_city = route[-1]
        for dest, dist in graph[last_city]:
            if dest == start_city:
                total_distance += dist
                break

        speed = Vmax - (total_weight / W) * (Vmax - Vmin)
        time = total_distance / speed
        fitness = total_profit
        return fitness, picked_items, total_weight, total_profit, speed

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
        candidate_fitness, candidate_items, candidate_weight, candidate_profit, candidate_speed = self.calculate_fitness(candidate)
        if candidate_fitness > self.current_fitness:
            self.current_fitness = candidate_fitness
            self.current_route = candidate
            self.current_items = candidate_items
            self.current_weight = candidate_weight
            self.current_profit = candidate_profit
            self.current_speed = candidate_speed
            if candidate_fitness > self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_route = candidate
                self.best_items = candidate_items
                self.best_weight = candidate_weight
                self.best_profit = candidate_profit
                self.best_speed = candidate_speed
        else:
            if random.random() < self.acceptance_probability(candidate_fitness):
                self.current_fitness = candidate_fitness
                self.current_route = candidate
                self.current_items = candidate_items
                self.current_weight = candidate_weight
                self.current_profit = candidate_profit
                self.current_speed = candidate_speed

    def anneal(self):
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = self.generate_valid_candidate()
            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1

        return self.best_route, self.best_fitness, self.best_items, self.best_weight, self.best_profit, self.best_speed

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time, speed):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Finalna prędkość złodzieja: {:.2f}".format(speed))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk : ", total_profit)
    print("Waga przenoszona w plecaku : ", total_weight)

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
best_route_sa, best_fitness_sa, best_items_sa, best_weight_sa, best_profit_sa, best_speed_sa = sa.anneal()
total_distance_sa = calculate_total_distance(best_route_sa)
total_time_sa = total_distance_sa / (Vmax - (best_weight_sa / W) * (Vmax - Vmin))
print_solution(best_route_sa, total_distance_sa, best_items_sa, best_profit_sa, best_weight_sa, total_time_sa, best_speed_sa)
