import random
import math
import time
from common.data_loader import load_data

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/280_1.txt")

# Ustaw parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio
# Obliczanie v_w - spadek prędkości w funkcji ciężaru plecaka
v_w = (Vmax - Vmin) / W

class SimulatedAnnealing:
    def __init__(self, initial_route, temp, alpha, stopping_temp, stopping_iter):
        self.current_route = initial_route
        self.best_route = list(initial_route)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1
        self.start_time = time.time()
        self.max_time = 3600  # Maksymalny czas wykonania w sekundach (1 godzina)

        # Cache dla przyspieszenia obliczeń - inicjalizacja przed użyciem
        self.distance_cache = {}
        self.weights_cache = {}

        self.current_objective, self.current_items, self.current_weight, self.current_profit, self.current_time, self.current_cost = self.calculate_objective_function(self.current_route)
        self.best_objective = self.current_objective
        self.best_items = self.current_items
        self.best_weight = self.current_weight
        self.best_profit = self.current_profit
        self.best_time = self.current_time
        self.best_cost = self.current_cost

    def get_distance(self, city1, city2):
        """Pobiera odległość między miastami z cache lub oblicza ją."""
        key = (city1, city2)
        if key in self.distance_cache:
            return self.distance_cache[key]
        
        for dest, dist in graph[city1]:
            if dest == city2:
                self.distance_cache[key] = dist
                return dist
        
        # Jeśli nie znaleziono odległości, zwróć dużą wartość
        self.distance_cache[key] = float('inf')
        return float('inf')

    def calculate_travel_time(self, route, weights_at_cities):
        """Oblicza czas podróży zgodnie z funkcją celu."""
        total_time = 0
        
        # Obliczanie czasu podróży między kolejnymi miastami (od x_1 do x_n)
        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            
            # Znajdź odległość między miastami
            distance = self.get_distance(current_city, next_city)
            
            # Oblicz prędkość na podstawie aktualnej wagi plecaka
            current_weight = weights_at_cities[i]
            speed = Vmax - current_weight * v_w  # Dokładnie zgodnie z funkcją celu
            
            # Dodaj czas podróży
            total_time += distance / speed
        
        # Dodaj czas powrotu do miasta początkowego (od x_n do x_1)
        last_city = route[-1]
        start_city = route[0]
        
        # Znajdź odległość powrotu
        return_distance = self.get_distance(last_city, start_city)
        
        # Oblicz prędkość na podstawie wagi plecaka po odwiedzeniu wszystkich miast
        last_weight = weights_at_cities[-1]
        return_speed = Vmax - last_weight * v_w  # Dokładnie zgodnie z funkcją celu
        
        # Dodaj czas powrotu
        total_time += return_distance / return_speed
        
        return total_time

    def calculate_objective_function(self, route):
        """Oblicza wartość funkcji celu zgodnie z podanym wzorem matematycznym."""
        # Obliczanie wag plecaka w każdym mieście
        weights_at_cities = [0] * len(route)
        current_weight = 0
        total_profit = 0
        picked_items = []
        
        # Dla każdego miasta w trasie
        for i, city in enumerate(route):
            # Wybór przedmiotów w aktualnym mieście
            for item in itemset.get(city, []):
                item_id, profit, weight = item
                if current_weight + weight <= W:
                    picked_items.append((city, item_id))
                    current_weight += weight
                    total_profit += profit
            
            # Zapisz aktualną wagę plecaka po opuszczeniu miasta
            weights_at_cities[i] = current_weight
        
        # Oblicz czas podróży
        travel_time = self.calculate_travel_time(route, weights_at_cities)
        
        # Oblicz koszt podróży
        travel_cost = R * travel_time
        
        # Oblicz wartość funkcji celu
        objective_value = total_profit - travel_cost
        
        return objective_value, picked_items, current_weight, total_profit, travel_time, travel_cost

    def acceptance_probability(self, candidate_objective):
        return math.exp(-abs(candidate_objective - self.current_objective) / self.temp)

    def is_valid_route(self, route):
        for i in range(len(route) - 1):
            if self.get_distance(route[i], route[i + 1]) == float('inf'):
                return False
        return self.get_distance(route[-1], route[0]) != float('inf')

    def generate_valid_candidate(self):
        """Generuje sąsiednie rozwiązanie z różnymi operatorami mutacji."""
        candidate = list(self.current_route)
        
        # Wybierz losowo operator mutacji
        operator = random.random()
        
        if operator < 0.4:  # 40% szans na odwrócenie fragmentu
            l = random.randint(2, min(5, len(candidate) - 1))  # Ograniczenie długości fragmentu
            i = random.randint(0, len(candidate) - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
        elif operator < 0.7:  # 30% szans na zamianę dwóch miast
            i = random.randint(0, len(candidate) - 1)
            j = random.randint(0, len(candidate) - 1)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        else:  # 30% szans na przesunięcie fragmentu
            l = random.randint(1, min(3, len(candidate) - 1))  # Ograniczenie długości fragmentu
            i = random.randint(0, len(candidate) - l)
            j = random.randint(0, len(candidate) - l)
            if i != j:
                fragment = candidate[i:i+l]
                del candidate[i:i+l]
                candidate[j:j] = fragment
        
        # Jeśli trasa nie jest poprawna, spróbuj ponownie
        if not self.is_valid_route(candidate):
            return self.generate_valid_candidate()
            
        return candidate

    def accept(self, candidate):
        candidate_objective, candidate_items, candidate_weight, candidate_profit, candidate_time, candidate_cost = self.calculate_objective_function(candidate)
        if candidate_objective > self.current_objective:
            self.current_objective = candidate_objective
            self.current_route = candidate
            self.current_items = candidate_items
            self.current_weight = candidate_weight
            self.current_profit = candidate_profit
            self.current_time = candidate_time
            self.current_cost = candidate_cost
            if candidate_objective > self.best_objective:
                self.best_objective = candidate_objective
                self.best_route = candidate
                self.best_items = candidate_items
                self.best_weight = candidate_weight
                self.best_profit = candidate_profit
                self.best_time = candidate_time
                self.best_cost = candidate_cost
        else:
            if random.random() < self.acceptance_probability(candidate_objective):
                self.current_objective = candidate_objective
                self.current_route = candidate
                self.current_items = candidate_items
                self.current_weight = candidate_weight
                self.current_profit = candidate_profit
                self.current_time = candidate_time
                self.current_cost = candidate_cost

    def anneal(self):
        no_improvement_count = 0
        last_best_objective = self.best_objective
        
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            # Sprawdź, czy nie przekroczono maksymalnego czasu
            if time.time() - self.start_time > self.max_time:
                print(f"Przekroczono maksymalny czas wykonania ({self.max_time} sekund).")
                break
                
            candidate = self.generate_valid_candidate()
            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1
            
            # Sprawdź, czy znaleziono lepsze rozwiązanie
            if self.best_objective > last_best_objective:
                last_best_objective = self.best_objective
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Jeśli przez długi czas nie ma poprawy, przyspiesz algorytm
            if no_improvement_count > 10000:
                self.temp *= 0.95  # Szybsze chłodzenie
                no_improvement_count = 0
                
            # Wyświetl postęp co 100000 iteracji
            if self.iteration % 100000 == 0:
                elapsed_time = time.time() - self.start_time
                print(f"Iteracja: {self.iteration}, Temperatura: {self.temp:.2f}, Najlepsza wartość: {self.best_objective:.2f}, Czas: {elapsed_time:.2f}s")

        return self.best_route, self.best_objective, self.best_items, self.best_weight, self.best_profit, self.best_time, self.best_cost

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time, total_cost, objective_value):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Koszt podróży: {:.2f}".format(total_cost))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk z przedmiotów: ", total_profit)
    print("Waga przenoszona w plecaku: ", total_weight)
    print("Wartość funkcji celu: {:.2f}".format(objective_value))

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

# Parametry optymalizacji - dostosowane do szybszego wykonania
initial_temperature = 1000000  # Zmniejszona temperatura początkowa
cooling_rate = 0.99995  # Zwiększony współczynnik chłodzenia
stopping_temperature = 0.1
stopping_iter = 1000000  # Zmniejszona liczba iteracji

# Testowanie algorytmów
print("\nUruchamianie algorytmu symulowanego wyżarzania...")
initial_route = generate_random_route()  # Użycie losowej trasy jako początkowej
sa = SimulatedAnnealing(initial_route=initial_route, temp=initial_temperature, alpha=cooling_rate, stopping_temp=stopping_temperature, stopping_iter=stopping_iter)
best_route_sa, best_objective_sa, best_items_sa, best_weight_sa, best_profit_sa, best_time_sa, best_cost_sa = sa.anneal()
total_distance_sa = calculate_total_distance(best_route_sa)
print_solution(best_route_sa, total_distance_sa, best_items_sa, best_profit_sa, best_weight_sa, best_time_sa, best_cost_sa, best_objective_sa)
