import random
import math
import time
from common.data_loader import load_data

graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/280_1.txt")

Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio
v_w = (Vmax - Vmin) / W

def efficient_two_opt(route):
    """Prosta implementacja zastępcza, która zwraca trasę bez zmian."""
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    
    return route.copy(), total_distance

def smart_item_selection(route, items_by_city):
    """Uproszczona wersja inteligentnego wyboru przedmiotów."""
    full_cycle = route + [route[0]]
    
    distances = []
    total_distance = 0
    for i in range(len(full_cycle) - 1):
        current_city = full_cycle[i]
        next_city = full_cycle[i + 1]
        found = False
        for dest, dist in graph[current_city]:
            if dest == next_city:
                distances.append(dist)
                total_distance += dist
                found = True
                break
        if not found:
            distances.append(1000)
            total_distance += 1000
    
    available_items = []
    for idx, city in enumerate(route):
        city_items = items_by_city.get(city, [])
        for item in city_items:
            item_id, profit, weight = item
            
            remaining_distance = sum(distances[idx:])
            
            speed_without_item = Vmax
            speed_with_item = max(Vmin, Vmax - weight * v_w)
            
            time_without_item = remaining_distance / speed_without_item
            time_with_item = remaining_distance / speed_with_item
            
            additional_time = time_with_item - time_without_item
            
            transport_cost = additional_time * R
            
            effective_profit = profit - transport_cost
            
            effective_ratio = effective_profit / weight if weight > 0 else 0
            
            if effective_ratio < 0:
                effective_ratio = 0
            
            available_items.append((city, item_id, profit, weight, effective_ratio, effective_profit))
    
    available_items.sort(key=lambda x: (-x[4]))
    
    picked_items = []
    total_weight = 0
    total_profit = 0
    remaining_capacity = W
    
    for item_data in available_items:
        city, item_id, profit, weight, _, effective_profit = item_data
        
        if weight <= remaining_capacity:
            if effective_profit > 0:
                picked_items.append((city, item_id))
                total_weight += weight
                total_profit += profit
                remaining_capacity -= weight
    
    return picked_items, total_weight, total_profit

class Particle:
    def __init__(self, route):
        self.route = self.ensure_all_cities(route)
        self.best_route = list(self.route)
        self.best_fitness, self.picked_items, self.total_weight, self.total_profit, self.travel_time, self.travel_cost = self.calculate_fitness()
        self.velocity = []
        self.weights_at_cities = []

    def ensure_all_cities(self, route):
        """Zapewnia, że trasa zawiera wszystkie miasta dokładnie raz."""
        all_cities = set(graph.keys())
        
        unique_route = []
        seen = set()
        for city in route:
            if city not in seen and city in all_cities:
                seen.add(city)
                unique_route.append(city)
        
        missing_cities = all_cities - seen
        if missing_cities:
            unique_route.extend(missing_cities)
        
        if len(unique_route) != len(all_cities):
            unique_route = list(all_cities)
            random.shuffle(unique_route)
            
        return unique_route

    def calculate_travel_time(self, weights_at_cities):
        """Oblicza czas podróży zgodnie z funkcją celu."""
        total_time = 0
        
        full_cycle = self.route + [self.route[0]]
        
        for i in range(len(full_cycle) - 1):
            current_city = full_cycle[i]
            next_city = full_cycle[i + 1]
            
            distance = 0
            for dest, dist in graph[current_city]:
                if dest == next_city:
                    distance = dist
                    break
            
            current_weight = weights_at_cities[i]
            speed = max(Vmin, Vmax - current_weight * v_w)
            
            total_time += distance / speed
        
        return total_time

    def calculate_fitness(self):
        """Oblicza wartość funkcji celu."""
        picked_items, total_weight, total_profit = smart_item_selection(self.route, itemset)
        
        weights_array = [0]
        current_weight = 0
        
        for city in self.route:
            for city_item, item_id in picked_items:
                if city_item == city:
                    for item in itemset.get(city, []):
                        if item[0] == item_id:
                            current_weight += item[2]
                            break
            weights_array.append(current_weight)
        
        travel_time = self.calculate_travel_time(weights_array)
        
        travel_cost = R * travel_time
        
        fitness = total_profit - travel_cost
        
        self.weights_at_cities = weights_array
        
        return fitness, picked_items, total_weight, total_profit, travel_time, travel_cost

    def update_velocity(self, global_best_route, w, c1, c2, global_best_fitness):
        """Aktualizuje prędkość cząstki."""
        self.velocity = []
        solution_gbest = list(global_best_route)
        solution_pbest = list(self.best_route)
        solution_current = list(self.route)
        
        alfa = c1
        beta = c2
        
        temp_pbest = list(solution_pbest)
        for i in range(len(solution_current)):
            if solution_current[i] != temp_pbest[i]:
                j = temp_pbest.index(solution_current[i])
                swap_operator = (i, j, alfa)
                self.velocity.append(swap_operator)
                temp_pbest[i], temp_pbest[j] = temp_pbest[j], temp_pbest[i]
        
        temp_gbest = list(solution_gbest)
        for i in range(len(solution_current)):
            if solution_current[i] != temp_gbest[i]:
                j = temp_gbest.index(solution_current[i])
                swap_operator = (i, j, beta)
                self.velocity.append(swap_operator)
                temp_gbest[i], temp_gbest[j] = temp_gbest[j], temp_gbest[i]
        
        if w > 0:
            for _ in range(max(1, int(len(self.route) * 0.05))):
                i = random.randint(0, len(self.route) - 1)
                j = random.randint(0, len(self.route) - 1)
                if i != j:
                    self.velocity.append(('swap', i, j, w))

    def update_position(self):
        """Aktualizuje pozycję cząstki."""
        if not self.velocity:
            return
            
        new_route = list(self.route)
        all_cities = set(graph.keys())
        
        for operator in self.velocity:
            if isinstance(operator, tuple) and len(operator) >= 3:
                if len(operator) == 3:
                    i, j, probability = operator
                    
                    if random.random() <= probability:
                        if 0 <= i < len(new_route) and 0 <= j < len(new_route):
                            new_route[i], new_route[j] = new_route[j], new_route[i]
                
                elif len(operator) == 4 and operator[0] == 'reverse':
                    _, i, j, probability = operator
                    
                    if random.random() <= probability:
                        if 0 <= i < len(new_route) and 0 <= j < len(new_route) and i < j:
                            new_route[i:j+1] = reversed(new_route[i:j+1])
                
                elif len(operator) == 4 and operator[0] == 'swap':
                    _, i, j, probability = operator
                    
                    if random.random() <= probability:
                        if 0 <= i < len(new_route) and 0 <= j < len(new_route):
                            new_route[i], new_route[j] = new_route[j], new_route[i]
            
            route_cities = set(new_route)
            if route_cities != all_cities or len(new_route) != len(all_cities):
                new_route = self.ensure_all_cities(new_route)
        
        self.route = new_route
        
        new_fitness, new_picked_items, new_weight, new_profit, new_time, new_cost = self.calculate_fitness()
        
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_route = list(self.route)
            self.picked_items = new_picked_items
            self.total_weight = new_weight
            self.total_profit = new_profit
            self.travel_time = new_time
            self.travel_cost = new_cost

class PSO:
    def __init__(self, num_particles, w, c1, c2, num_iterations):
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_iterations = num_iterations
        self.particles = []
        self.best_fitness_history = []
        self.stagnation_count = 0
        self.last_best_fitness = float('-inf')
        
        print(f"Inicjalizacja {num_particles} cząstek...")
        all_cities = list(graph.keys())
        
        initial_routes = []
        
        print("Generowanie różnorodnych tras początkowych...")
        for i in range(num_particles // 3 * 5):
            method = i % 5
            if method == 0:
                route = self.generate_distance_based_route()
            elif method == 1:
                route = self.generate_value_based_route()
            elif method == 2:
                route = self.generate_hybrid_route()
            elif method == 3:
                route = self.generate_insertion_route()
            else:
                route = self.generate_random_route()
                
            num_swaps = max(2, int(len(route) * 0.05))
            for _ in range(num_swaps):
                i = random.randint(0, len(route) - 1)
                j = random.randint(0, len(route) - 1)
                if i != j:
                    route[i], route[j] = route[j], route[i]
                    
            initial_routes.append(route)
            
        print("Tworzenie cząstek i ocena początkowej populacji...")
        temp_particles = [Particle(route) for route in initial_routes]
        
        temp_particles.sort(key=lambda p: p.best_fitness, reverse=True)
        self.particles = temp_particles[:num_particles]
        
        distinct_particles = []
        
        for p in self.particles:
            is_distinct = True
            for dp in distinct_particles:
                hamming_distance = sum(1 for a, b in zip(p.route, dp.route) if a != b)
                if hamming_distance < len(p.route) * 0.1:
                    is_distinct = False
                    break
                
            if is_distinct or len(distinct_particles) < num_particles * 0.5:
                distinct_particles.append(p)
                
            if len(distinct_particles) >= num_particles:
                break
        
        while len(distinct_particles) < num_particles:
            route = self.generate_random_route()
            distinct_particles.append(Particle(route))
        
        self.particles = distinct_particles

        best_particle = max(self.particles, key=lambda p: p.best_fitness)
        self.global_best_route = best_particle.best_route
        self.global_best_fitness = best_particle.best_fitness
        self.global_best_items = best_particle.picked_items
        self.global_best_weight = best_particle.total_weight
        self.global_best_profit = best_particle.total_profit
        self.global_best_time = best_particle.travel_time
        self.global_best_cost = best_particle.travel_cost
        
        self.best_fitness_history.append(self.global_best_fitness)
        
        initial_distance = calculate_total_distance(self.global_best_route)
        print(f"Początkowa długość najlepszej trasy: {initial_distance:.2f}, fitness: {self.global_best_fitness:.2f}")

    def generate_random_route(self):
        """Generuje losową trasę."""
        cities = list(graph.keys())
        random.shuffle(cities)
        return cities
        
    def generate_distance_based_route(self):
        """Generuje trasę opartą na odległości między miastami."""
        cities = list(graph.keys())
        start_city = random.choice(cities)
        route = [start_city]
        unvisited = set(cities) - {start_city}
        
        while unvisited:
            current = route[-1]
            next_cities = []
            
            for city in unvisited:
                distance = float('inf')
                for dest, dist in graph[current]:
                    if dest == city:
                        distance = dist
                        break
                next_cities.append((city, distance))
            
            next_cities.sort(key=lambda x: x[1])
            
            if next_cities:
                if random.random() < 0.9:
                    top_n = min(3, len(next_cities))
                    next_city = next_cities[random.randint(0, top_n-1)][0]
                else:
                    next_city = random.choice(next_cities)[0]
                
                route.append(next_city)
                unvisited.remove(next_city)
            else:
                break
        
        return route
    
    def generate_value_based_route(self):
        """Generuje trasę opartą na wartości przedmiotów w miastach."""
        city_values = {}
        for city, items in itemset.items():
            if items:
                total_value = sum(item[1] for item in items)
                city_values[city] = total_value / len(items)
            else:
                city_values[city] = 0
        
        all_cities = set(graph.keys())
        for city in all_cities:
            if city not in city_values:
                city_values[city] = 0
        
        city_value_pairs = [(city, city_values[city]) for city in all_cities]
        
        if random.random() < 0.5:
            city_value_pairs.sort(key=lambda x: x[1], reverse=True)
        else:
            city_value_pairs.sort(key=lambda x: x[1])
        
        route = [pair[0] for pair in city_value_pairs]
        
        if len(route) > 3:
            i = random.randint(0, len(route) - 3)
            j = random.randint(i + 1, len(route) - 1)
            route[i:j+1] = reversed(route[i:j+1])
        
        return route
        
    def generate_hybrid_route(self):
        """Generuje trasę opartą na kombinacji odległości i wartości przedmiotów."""
        cities = list(graph.keys())
        start_city = random.choice(cities)
        route = [start_city]
        unvisited = set(cities) - {start_city}
        balance = random.random()
        
        city_values = {}
        for city, items in itemset.items():
            if items:
                total_value = sum(item[1] for item in items)
                city_values[city] = total_value / len(items)
            else:
                city_values[city] = 0
        
        for city in unvisited:
            if city not in city_values:
                city_values[city] = 0
        
        max_value = max(city_values.values()) if city_values else 1
        if max_value > 0:
            for city in city_values:
                city_values[city] /= max_value
        
        while unvisited:
            current = route[-1]
            next_cities = []
            
            for city in unvisited:
                distance = float('inf')
                for dest, dist in graph[current]:
                    if dest == city:
                        distance = dist
                        break
                
                value = city_values.get(city, 0)
                
                max_distance = 10000
                norm_distance = 1 - min(distance / max_distance, 1)
                
                combined_metric = (1 - balance) * norm_distance + balance * value
                
                next_cities.append((city, combined_metric))
            
            next_cities.sort(key=lambda x: x[1], reverse=True)
            
            if next_cities:
                top_n = min(3, len(next_cities))
                next_city = next_cities[random.randint(0, top_n-1)][0]
                
                route.append(next_city)
                unvisited.remove(next_city)
            else:
                break
        
        return route
        
    def generate_insertion_route(self):
        """Generuje trasę z użyciem metody losowego wstawiania."""
        cities = list(graph.keys())
        
        if len(cities) >= 3:
            initial_cities = random.sample(cities, 3)
            route = initial_cities
            unvisited = set(cities) - set(initial_cities)
        else:
            route = cities.copy()
            unvisited = set()
        
        while unvisited:
            city = random.choice(list(unvisited))
            
            best_position = 0
            best_increase = float('inf')
            
            for i in range(len(route)):
                prev_city = route[i-1] if i > 0 else route[-1]
                next_city = route[i]
                
                dist_prev_to_city = float('inf')
                for dest, dist in graph[prev_city]:
                    if dest == city:
                        dist_prev_to_city = dist
                        break
                        
                dist_city_to_next = float('inf')
                for dest, dist in graph[city]:
                    if dest == next_city:
                        dist_city_to_next = dist
                        break
                        
                dist_prev_to_next = float('inf')
                for dest, dist in graph[prev_city]:
                    if dest == next_city:
                        dist_prev_to_next = dist
                        break
                
                increase = dist_prev_to_city + dist_city_to_next - dist_prev_to_next
                
                if increase < best_increase:
                    best_increase = increase
                    best_position = i
            
            route.insert(best_position, city)
            unvisited.remove(city)
        
        return route

    def run(self):
        """Główna pętla algorytmu PSO."""
        last_improvement_iteration = 0
        best_route_distance = float('inf')
        
        for iteration in range(self.num_iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_route, self.w, self.c1, self.c2, self.global_best_fitness)
                particle.update_position()
                
                if particle.best_fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_route = particle.best_route
                    self.global_best_items = particle.picked_items
                    self.global_best_weight = particle.total_weight
                    self.global_best_profit = particle.total_profit
                    self.global_best_time = particle.travel_time
                    self.global_best_cost = particle.travel_cost
                    last_improvement_iteration = iteration
                    self.stagnation_count = 0
                    
                    current_route_distance = calculate_total_distance(particle.best_route)
                    if current_route_distance < best_route_distance:
                        best_route_distance = current_route_distance
                        print(f"Znaleziono lepszą trasę, odległość: {best_route_distance:.2f}, funkcja celu: {self.global_best_fitness:.2f}")
            
            if self.global_best_fitness > self.last_best_fitness:
                self.last_best_fitness = self.global_best_fitness
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            
            self.best_fitness_history.append(self.global_best_fitness)
            
            if (iteration + 1) % 10 == 0:
                current_distance = calculate_total_distance(self.global_best_route)
                print(f"Iteracja {iteration + 1}/{self.num_iterations}, Wartość F.C.: {self.global_best_fitness:.2f}, Długość trasy: {current_distance:.2f}")
            
            if iteration - last_improvement_iteration > 100:
                print(f"Brak poprawy wartości funkcji celu przez {iteration - last_improvement_iteration} iteracji.")
                break
        
        return (self.global_best_route, self.global_best_fitness, self.global_best_items, 
                self.global_best_weight, self.global_best_profit, self.global_best_time, self.global_best_cost)

def calculate_total_distance(route):
    """Oblicza całkowitą odległość dla podanej trasy."""
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    return total_distance

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time, travel_cost, objective_value):
    """Wypisuje znalezione rozwiązanie."""
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Koszt podróży: {:.2f}".format(travel_cost))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk z przedmiotów: {:.2f}".format(total_profit))
    print("Waga przenoszona w plecaku: ", total_weight)
    print("Wartość funkcji celu: {:.2f}".format(objective_value))

num_particles = 20
w = 0.5
c1 = 1.0 
c2 = 1.0
num_iterations = 100

def main():
    print(f"Uruchamianie algorytmu dla problemu TTP")
    
    pso = PSO(num_particles, w, c1, c2, num_iterations)
    (best_route, best_fitness, best_items, total_weight, total_profit, total_time, travel_cost) = pso.run()

    total_distance = calculate_total_distance(best_route)
    print("\nNajlepsze znalezione rozwiązanie:")
    print_solution(best_route, total_distance, best_items, total_profit, total_weight, total_time, travel_cost, best_fitness)

if __name__ == "__main__":
    main()
