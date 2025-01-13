import random
import math
from common.data_loader import load_data  # Wcześniej zaimplementowana funkcja

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/280_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

class Particle:
    def __init__(self, route):
        self.route = self.ensure_all_cities(route)
        self.best_route = list(self.route)
        self.best_fitness, self.picked_items, self.total_weight, self.total_profit, self.final_speed = self.calculate_fitness()
        self.velocity = []

    def ensure_all_cities(self, route):
        """Zapewnia, że trasa zawiera wszystkie miasta."""
        all_cities = set(graph.keys())
        missing_cities = list(all_cities - set(route))
        route += missing_cities  # Dodaj brakujące miasta
        random.shuffle(route)  # Losowo permutuj trasę
        return route

    def calculate_fitness(self):
        total_distance = 0
        total_profit = 0
        total_weight = 0
        picked_items = []

        for i in range(len(self.route) - 1):
            current_city = self.route[i]
            next_city = self.route[i + 1]
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
        start_city = self.route[0]
        last_city = self.route[-1]
        for dest, dist in graph[last_city]:
            if dest == start_city:
                total_distance += dist
                break

        final_speed = Vmax - (total_weight / W) * (Vmax - Vmin)
        time = total_distance / final_speed
        fitness = total_profit
        return fitness, picked_items, total_weight, total_profit, final_speed

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
        new_fitness, new_picked_items, new_total_weight, new_total_profit, new_final_speed = self.calculate_fitness()
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_route = list(self.route)
            self.picked_items = new_picked_items
            self.total_weight = new_total_weight
            self.total_profit = new_total_profit
            self.final_speed = new_final_speed

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
        self.global_best_items = self.particles[0].picked_items
        self.global_best_weight = self.particles[0].total_weight
        self.global_best_profit = self.particles[0].total_profit
        self.global_best_speed = self.particles[0].final_speed

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
                    self.global_best_items = particle.picked_items
                    self.global_best_weight = particle.total_weight
                    self.global_best_profit = particle.total_profit
                    self.global_best_speed = particle.final_speed

        return (self.global_best_route, self.global_best_fitness, self.global_best_items, 
                self.global_best_weight, self.global_best_profit, self.global_best_speed)

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

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time, final_speed):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Finalna prędkość złodzieja: {:.2f}".format(final_speed))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk : ", total_profit)
    print("Waga przenoszona w plecaku : ", total_weight)

# Parametry PSO dla dużego problemu
num_particles = 1000  # Liczba cząstek
w = 0.7  # Waga bezwładności
c1 = 10.0  # Składnik poznawczy
c2 = 10.0  # Składnik społeczny
num_iterations = 1000  # Liczba iteracji

# Testowanie algorytmu PSO
print("Uruchamianie algorytmu PSO...")
pso = PSO(num_particles, w, c1, c2, num_iterations)
(best_route_pso, best_fitness_pso, best_items_pso, 
 total_weight_pso, total_profit_pso, final_speed_pso) = pso.run()

total_distance_pso = calculate_total_distance(best_route_pso)
total_time_pso = total_distance_pso / final_speed_pso
print_solution(best_route_pso, total_distance_pso, best_items_pso, total_profit_pso, total_weight_pso, total_time_pso, final_speed_pso)
