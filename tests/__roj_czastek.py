import random
import math
import time
from common.data_loader import load_data

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/50_1.txt")

# Parametry problemu
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
    # Tworzymy pełny cykl
    full_cycle = route + [route[0]]
    
    # Obliczamy odległości między kolejnymi miastami
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
    
    # Przygotowujemy listę dostępnych przedmiotów
    available_items = []
    for idx, city in enumerate(route):
        city_items = items_by_city.get(city, [])
        for item in city_items:
            item_id, profit, weight = item
            
            # Obliczamy dystans, który przedmiot będzie musiał przebyć
            remaining_distance = sum(distances[idx:])
            
            # Obliczamy wpływ przedmiotu na czas podróży
            speed_without_item = Vmax
            speed_with_item = max(Vmin, Vmax - weight * v_w)
            
            # Obliczamy czas podróży bez przedmiotu i z przedmiotem
            time_without_item = remaining_distance / speed_without_item
            time_with_item = remaining_distance / speed_with_item
            
            # Obliczenie dodatkowego czasu
            additional_time = time_with_item - time_without_item
            
            # Obliczamy koszt transportu
            transport_cost = additional_time * R
            
            # Efektywna wartość przedmiotu
            effective_profit = profit - transport_cost
            
            # Wartość efektywna na jednostkę wagi
            effective_ratio = effective_profit / weight if weight > 0 else 0
            
            # Zabezpieczenie przed ujemnymi wartościami
            if effective_ratio < 0:
                effective_ratio = 0
            
            # Zapisujemy przedmiot
            available_items.append((city, item_id, profit, weight, effective_ratio, effective_profit))
    
    # Sortujemy przedmioty według ich efektywnej wartości (malejąco)
    available_items.sort(key=lambda x: (-x[4]))
    
    # Wybieramy przedmioty, nie przekraczając pojemności plecaka
    picked_items = []
    total_weight = 0
    total_profit = 0
    remaining_capacity = W
    
    for item_data in available_items:
        city, item_id, profit, weight, _, effective_profit = item_data
        
        # Sprawdź, czy przedmiot zmieści się w plecaku
        if weight <= remaining_capacity:
            # Bierzemy tylko przedmioty z pozytywnym wpływem na funkcję celu
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
        
        # Usuń duplikaty, zachowując kolejność
        unique_route = []
        seen = set()
        for city in route:
            if city not in seen and city in all_cities:
                seen.add(city)
                unique_route.append(city)
        
        # Dodaj brakujące miasta
        missing_cities = all_cities - seen
        if missing_cities:
            unique_route.extend(missing_cities)
        
        # Sprawdź finalną trasę
        if len(unique_route) != len(all_cities):
            unique_route = list(all_cities)
            random.shuffle(unique_route)
            
        return unique_route

    def calculate_travel_time(self, weights_at_cities):
        """Oblicza czas podróży zgodnie z funkcją celu."""
        total_time = 0
        
        # Tworzy pełny cykl
        full_cycle = self.route + [self.route[0]]
        
        for i in range(len(full_cycle) - 1):
            current_city = full_cycle[i]
            next_city = full_cycle[i + 1]
            
            # Znajdź odległość między miastami
            distance = 0
            for dest, dist in graph[current_city]:
                if dest == next_city:
                    distance = dist
                    break
            
            # Oblicz prędkość na podstawie aktualnej wagi plecaka
            current_weight = weights_at_cities[i]
            speed = max(Vmin, Vmax - current_weight * v_w)
            
            # Dodaj czas podróży
            total_time += distance / speed
        
        return total_time

    def calculate_fitness(self):
        """Oblicza wartość funkcji celu."""
        # Użyj inteligentnego wyboru przedmiotów
        picked_items, total_weight, total_profit = smart_item_selection(self.route, itemset)
        
        # Obliczamy dokładne wagi plecaka w każdym mieście na trasie
        weights_array = [0]  # Początkowa waga przed pierwszym miastem
        current_weight = 0
        
        for city in self.route:
            # Sprawdź, czy w tym mieście zostały zebrane jakieś przedmioty
            for city_item, item_id in picked_items:
                if city_item == city:
                    # Znajdź wagę przedmiotu
                    for item in itemset.get(city, []):
                        if item[0] == item_id:
                            current_weight += item[2]
                            break
            weights_array.append(current_weight)
        
        # Oblicz czas podróży
        travel_time = self.calculate_travel_time(weights_array)
        
        # Oblicz koszt podróży
        travel_cost = R * travel_time
        
        # Oblicz wartość funkcji celu
        fitness = total_profit - travel_cost
        
        # Zapisz wagi plecaka w każdym mieście
        self.weights_at_cities = weights_array
        
        return fitness, picked_items, total_weight, total_profit, travel_time, travel_cost

    def update_velocity(self, global_best_route, w, c1, c2, global_best_fitness):
        """Aktualizuje prędkość cząstki."""
        self.velocity = []
        
        # Kopiujemy obecne rozwiązanie i najlepsze rozwiązania
        solution_gbest = list(global_best_route)
        solution_pbest = list(self.best_route)
        solution_current = list(self.route)
        
        # Określamy prawdopodobieństwa
        alfa = c1
        beta = c2
        
        # Komponenta poznawcza (pbest - current)
        temp_pbest = list(solution_pbest)
        for i in range(len(solution_current)):
            if solution_current[i] != temp_pbest[i]:
                j = temp_pbest.index(solution_current[i])
                swap_operator = (i, j, alfa)
                self.velocity.append(swap_operator)
                temp_pbest[i], temp_pbest[j] = temp_pbest[j], temp_pbest[i]
        
        # Komponenta społeczna (gbest - current)
        temp_gbest = list(solution_gbest)
        for i in range(len(solution_current)):
            if solution_current[i] != temp_gbest[i]:
                j = temp_gbest.index(solution_current[i])
                swap_operator = (i, j, beta)
                self.velocity.append(swap_operator)
                temp_gbest[i], temp_gbest[j] = temp_gbest[j], temp_gbest[i]
        
        # Komponent bezwładności (inercja)
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
        
        # Stosujemy operatory zamiany
        for operator in self.velocity:
            if isinstance(operator, tuple) and len(operator) >= 3:
                # Standardowy operator zamiany (i, j, prawdopodobieństwo)
                if len(operator) == 3:
                    i, j, probability = operator
                    
                    if random.random() <= probability:
                        if 0 <= i < len(new_route) and 0 <= j < len(new_route):
                            new_route[i], new_route[j] = new_route[j], new_route[i]
                
                # Specjalny operator typu (reverse, i, j, prawdopodobieństwo)
                elif len(operator) == 4 and operator[0] == 'reverse':
                    _, i, j, probability = operator
                    
                    if random.random() <= probability:
                        if 0 <= i < len(new_route) and 0 <= j < len(new_route) and i < j:
                            new_route[i:j+1] = reversed(new_route[i:j+1])
                
                # Specjalny operator typu (swap, i, j, prawdopodobieństwo)
                elif len(operator) == 4 and operator[0] == 'swap':
                    _, i, j, probability = operator
                    
                    if random.random() <= probability:
                        if 0 <= i < len(new_route) and 0 <= j < len(new_route):
                            new_route[i], new_route[j] = new_route[j], new_route[i]
            
            # Walidacja trasy po każdej operacji
            route_cities = set(new_route)
            if route_cities != all_cities or len(new_route) != len(all_cities):
                new_route = self.ensure_all_cities(new_route)
        
        # Zapisujemy nową trasę
        self.route = new_route
        
        # Obliczamy nową wartość funkcji celu
        new_fitness, new_picked_items, new_weight, new_profit, new_time, new_cost = self.calculate_fitness()
        
        # Aktualizujemy najlepsze rozwiązanie, jeśli nowe jest lepsze
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
        
        # Inicjalizacja cząstek
        print(f"Inicjalizacja {num_particles} cząstek...")
        all_cities = list(graph.keys())
        
        initial_routes = []
        
        # Generujemy więcej różnorodnych tras niż potrzebujemy
        print("Generowanie różnorodnych tras początkowych...")
        for i in range(num_particles // 3 * 5):  # Generujemy więcej, żeby wybrać najlepsze
            method = i % 5  # 5 różnych metod generowania tras
            if method == 0:
                # Trasa oparta na odległości
                route = self.generate_distance_based_route()
            elif method == 1:
                # Trasa oparta na wartości przedmiotów
                route = self.generate_value_based_route()
            elif method == 2:
                # Trasa oparta na kombinacji odległości i wartości
                route = self.generate_hybrid_route()
            elif method == 3:
                # Trasa oparta na losowym insercie
                route = self.generate_insertion_route()
            else:
                # Całkowicie losowa trasa
                route = self.generate_random_route()
                
            # Dodajemy małą losowość do każdej trasy (2-opt)
            num_swaps = max(2, int(len(route) * 0.05))  # 5% zamian
            for _ in range(num_swaps):
                i = random.randint(0, len(route) - 1)
                j = random.randint(0, len(route) - 1)
                if i != j:
                    route[i], route[j] = route[j], route[i]
                    
            initial_routes.append(route)
            
        # Tworzymy cząstki i oceniamy je
        print("Tworzenie cząstek i ocena początkowej populacji...")
        temp_particles = [Particle(route) for route in initial_routes]
        
        # Sortujemy według fitness i bierzemy najlepsze
        temp_particles.sort(key=lambda p: p.best_fitness, reverse=True)
        self.particles = temp_particles[:num_particles]
        
        # Zapewniamy różnorodność - sprawdzamy pary tras i upewniamy się,
        # że nie są zbyt podobne (oceniamy przez odległość Hamminga)
        distinct_particles = []
        
        for p in self.particles:
            is_distinct = True
            for dp in distinct_particles:
                # Obliczamy odległość Hamminga (liczbę różnic)
                hamming_distance = sum(1 for a, b in zip(p.route, dp.route) if a != b)
                if hamming_distance < len(p.route) * 0.1:  # Jeśli różnią się mniej niż 10%
                    is_distinct = False
                    break
                
            if is_distinct or len(distinct_particles) < num_particles * 0.5:
                distinct_particles.append(p)
                
            if len(distinct_particles) >= num_particles:
                break
        
        # Jeśli nie znaleźliśmy wystarczająco różnych, uzupełniamy losowymi
        while len(distinct_particles) < num_particles:
            route = self.generate_random_route()
            distinct_particles.append(Particle(route))
        
        self.particles = distinct_particles

        # Znajdź najlepszą cząstkę
        best_particle = max(self.particles, key=lambda p: p.best_fitness)
        self.global_best_route = best_particle.best_route
        self.global_best_fitness = best_particle.best_fitness
        self.global_best_items = best_particle.picked_items
        self.global_best_weight = best_particle.total_weight
        self.global_best_profit = best_particle.total_profit
        self.global_best_time = best_particle.travel_time
        self.global_best_cost = best_particle.travel_cost
        
        # Zapisz początkową wartość funkcji celu
        self.best_fitness_history.append(self.global_best_fitness)
        
        # Raportuj początkową długość trasy
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
                if random.random() < 0.9:  # 90% szans na wybór najbliższego miasta
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
        
        # Posortuj miasta według wartości przedmiotów
        city_value_pairs = [(city, city_values[city]) for city in all_cities]
        
        # Strategia 1: Najpierw miasta z najcenniejszymi przedmiotami
        if random.random() < 0.5:
            city_value_pairs.sort(key=lambda x: x[1], reverse=True)
        # Strategia 2: Najpierw miasta z najmniej cennymi przedmiotami
        else:
            city_value_pairs.sort(key=lambda x: x[1])
        
        route = [pair[0] for pair in city_value_pairs]
        
        # Dodaj losowość - odwróć losowy segment trasy
        if len(route) > 3:
            i = random.randint(0, len(route) - 3)
            j = random.randint(i + 1, len(route) - 1)
            route[i:j+1] = reversed(route[i:j+1])
        
        return route
        
    def generate_hybrid_route(self):
        """Generuje trasę opartą na kombinacji odległości i wartości przedmiotów."""
        # Podobnie jak najbliższy sąsiad, ale bierzemy pod uwagę również wartość przedmiotów
        cities = list(graph.keys())
        start_city = random.choice(cities)
        route = [start_city]
        unvisited = set(cities) - {start_city}
        
        # Współczynnik balansu między odległością a wartością przedmiotów (0-1)
        # 0 - tylko odległość, 1 - tylko wartość przedmiotów
        balance = random.random()
        
        # Słownik wartości przedmiotów w miastach
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
        
        # Znormalizuj wartości
        max_value = max(city_values.values()) if city_values else 1
        if max_value > 0:
            for city in city_values:
                city_values[city] /= max_value
        
        while unvisited:
            current = route[-1]
            next_cities = []
            
            for city in unvisited:
                # Składnik odległości
                distance = float('inf')
                for dest, dist in graph[current]:
                    if dest == city:
                        distance = dist
                        break
                
                # Składnik wartości przedmiotów
                value = city_values.get(city, 0)
                
                # Normalizacja odległości (mniejsza = lepsza)
                max_distance = 10000  # Przybliżona maksymalna odległość
                norm_distance = 1 - min(distance / max_distance, 1)
                
                # Połączona metryka - wyższa wartość = lepsze miasto
                combined_metric = (1 - balance) * norm_distance + balance * value
                
                next_cities.append((city, combined_metric))
            
            next_cities.sort(key=lambda x: x[1], reverse=True)
            
            if next_cities:
                # Wybierz jedno z najlepszych miast
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
        
        # Wybierz 3 losowe miasta jako początek
        if len(cities) >= 3:
            initial_cities = random.sample(cities, 3)
            route = initial_cities
            unvisited = set(cities) - set(initial_cities)
        else:
            route = cities.copy()
            unvisited = set()
        
        # Wstawiaj pozostałe miasta w losowych miejscach
        while unvisited:
            city = random.choice(list(unvisited))
            
            # Znajdź najlepsze miejsce do wstawienia
            best_position = 0
            best_increase = float('inf')
            
            for i in range(len(route)):
                # Oblicz wzrost długości trasy po wstawieniu miasta w pozycji i
                prev_city = route[i-1] if i > 0 else route[-1]
                next_city = route[i]
                
                # Znajdź odległości
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
                
                # Oblicz wzrost długości trasy
                increase = dist_prev_to_city + dist_city_to_next - dist_prev_to_next
                
                if increase < best_increase:
                    best_increase = increase
                    best_position = i
            
            # Wstaw miasto w najlepszej pozycji
            route.insert(best_position, city)
            unvisited.remove(city)
        
        return route

    def run(self):
        """Główna pętla algorytmu PSO."""
        last_improvement_iteration = 0
        best_route_distance = float('inf')
        
        for iteration in range(self.num_iterations):
            # Aktualizuj pozycje wszystkich cząstek
            for particle in self.particles:
                particle.update_velocity(self.global_best_route, self.w, self.c1, self.c2, self.global_best_fitness)
                particle.update_position()
                
                # Sprawdź, czy znaleziono lepsze globalne rozwiązanie
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
                    
                    # Sprawdź, czy poprawiono również trasę
                    current_route_distance = calculate_total_distance(particle.best_route)
                    if current_route_distance < best_route_distance:
                        best_route_distance = current_route_distance
                        print(f"Znaleziono lepszą trasę, odległość: {best_route_distance:.2f}, funkcja celu: {self.global_best_fitness:.2f}")
            
            # Sprawdź, czy nastąpiła poprawa
            if self.global_best_fitness > self.last_best_fitness:
                self.last_best_fitness = self.global_best_fitness
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            
            # Zapisz wartość funkcji celu
            self.best_fitness_history.append(self.global_best_fitness)
            
            # Wyświetl postęp co 10 iteracji
            if (iteration + 1) % 10 == 0:
                current_distance = calculate_total_distance(self.global_best_route)
                print(f"Iteracja {iteration + 1}/{self.num_iterations}, Wartość F.C.: {self.global_best_fitness:.2f}, Długość trasy: {current_distance:.2f}")
            
            # Sprawdzenie warunku zakończenia
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
    # Dodaj odległość powrotu do miasta startowego
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

# Parametry PSO
num_particles = 100
w = 0.7  # Waga bezwładności
c1 = 2  # Składnik poznawczy
c2 = 2  # Składnik społeczny
num_iterations = 200

# Główna funkcja programu
def main():
    print(f"Uruchamianie algorytmu dla problemu TTP")
    
    # Inicjalizacja i uruchomienie PSO
    pso = PSO(num_particles, w, c1, c2, num_iterations)
    (best_route, best_fitness, best_items, total_weight, total_profit, total_time, travel_cost) = pso.run()

    # Wypisz wyniki
    total_distance = calculate_total_distance(best_route)
    print("\nNajlepsze znalezione rozwiązanie:")
    print_solution(best_route, total_distance, best_items, total_profit, total_weight, total_time, travel_cost, best_fitness)

# Uruchomienie programu
if __name__ == "__main__":
    main()

def run_tests():
    # Parametry algorytmu
    good_parameters = {
        'num_particles': 50,
        'w': 0.8,
        'c1': 2.5,
        'c2': 2.5,
        'num_iterations': 300
    }
    
    weak_parameters = {
        'num_particles': 20,
        'w': 0.5,
        'c1': 1.0,
        'c2': 1.0,
        'num_iterations': 100
    }
    
    # Pliki testowe
    files = ["data/50_1.txt"]
    
    # Tworzenie katalogu na wyniki
    import os
    output_dir = "tests/output/Algorytm Roju Cząstek"
    os.makedirs(output_dir, exist_ok=True)
    
    # Listy na wyniki
    optimality_results = []
    efficiency_results = []
    stability_results = []
    memory_results = []
    
    for file in files:
        # Wczytanie danych
        graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file)
        
        # Testy optymalności
        for params_name, params in [("dobre", good_parameters), ("słabe", weak_parameters)]:
            pso = PSO(**params)
            best_route, best_fitness, picked_items, total_weight, total_profit, travel_time, travel_cost = pso.run()
            
            optimality_results.append({
                'Plik': file,
                'Parametry': params_name,
                'Wartość funkcji celu': best_fitness,
                'Czas podróży': travel_time,
                'Koszt podróży': travel_cost,
                'Długość trasy': calculate_total_distance(best_route),
                'Zysk z przedmiotów': total_profit,
                'Waga przedmiotów': total_weight
            })
        
        # Testy wydajności
        for params_name, params in [("dobre", good_parameters), ("słabe", weak_parameters)]:
            start_time = time.time()
            pso = PSO(**params)
            pso.run()
            end_time = time.time()
            
            efficiency_results.append({
                'Plik': file,
                'Parametry': params_name,
                'Czas wykonania': end_time - start_time
            })
        
        # Testy stabilności (5 uruchomień) - tylko dla dobrych parametrów
        stability_run_results = []
        for run in range(5):
            pso = PSO(**good_parameters)
            best_route, best_fitness, picked_items, total_weight, total_profit, travel_time, travel_cost = pso.run()
            
            stability_run_results.append({
                'Uruchomienie': run + 1,
                'Wartość funkcji celu': best_fitness,
                'Czas podróży': travel_time,
                'Koszt podróży': travel_cost,
                'Długość trasy': calculate_total_distance(best_route),
                'Zysk z przedmiotów': total_profit,
                'Waga przedmiotów': total_weight
            })
        
        stability_results.append({
            'Plik': file,
            'Wyniki': stability_run_results
        })
        
        # Testy złożoności pamięciowej
        for params_name, params in [("dobre", good_parameters), ("słabe", weak_parameters)]:
            import tracemalloc
            tracemalloc.start()
            pso = PSO(**params)
            pso.run()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_results.append({
                'Plik': file,
                'Parametry': params_name,
                'Aktualne użycie pamięci': current / 1024 / 1024,  # MB
                'Szczytowe użycie pamięci': peak / 1024 / 1024  # MB
            })
    
    # Zapisywanie wyników do plików Excel
    import pandas as pd
    
    # Wyniki optymalności
    pd.DataFrame(optimality_results).to_excel(
        f"{output_dir}/pso_optimal_results.xlsx",
        index=False
    )
    
    # Wyniki wydajności
    pd.DataFrame(efficiency_results).to_excel(
        f"{output_dir}/pso_efficiency_results.xlsx",
        index=False
    )
    
    # Wyniki stabilności
    for result in stability_results:
        pd.DataFrame(result['Wyniki']).to_excel(
            f"{output_dir}/pso_stability_{result['Plik'].split('/')[-1]}.xlsx",
            index=False
        )
    
    # Wyniki pamięci
    pd.DataFrame(memory_results).to_excel(
        f"{output_dir}/pso_memory_results.xlsx",
        index=False
    )

if __name__ == "__main__":
    run_tests()
