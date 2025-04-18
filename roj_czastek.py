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
# Obliczanie v_w - spadek prędkości w funkcji ciężaru plecaka
v_w = (Vmax - Vmin) / W

class Particle:
    def __init__(self, route):
        self.route = self.ensure_all_cities(route)
        self.best_route = list(self.route)
        self.best_fitness, self.picked_items, self.total_weight, self.total_profit, self.travel_time, self.travel_cost = self.calculate_fitness()
        self.velocity = []
        self.weights_at_cities = []  # Lista wag plecaka w każdym mieście

    def ensure_all_cities(self, route):
        """Zapewnia, że trasa zawiera wszystkie miasta."""
        all_cities = set(graph.keys())
        missing_cities = list(all_cities - set(route))
        route += missing_cities  # Dodaj brakujące miasta
        random.shuffle(route)  # Losowo permutuj trasę
        return route

    def calculate_travel_time(self, weights_at_cities):
        """Oblicza czas podróży zgodnie z funkcją celu."""
        total_time = 0
        
        # Obliczanie czasu podróży między kolejnymi miastami
        for i in range(len(self.route) - 1):
            current_city = self.route[i]
            next_city = self.route[i + 1]
            
            # Znajdź odległość między miastami
            distance = 0
            for dest, dist in graph[current_city]:
                if dest == next_city:
                    distance = dist
                    break
            
            # Oblicz prędkość na podstawie aktualnej wagi plecaka
            current_weight = weights_at_cities[i]
            speed = max(Vmin, Vmax - current_weight * v_w)  # Upewnij się, że prędkość nie spadnie poniżej Vmin
            
            # Dodaj czas podróży
            total_time += distance / speed
        
        # Dodaj czas powrotu do miasta początkowego
        last_city = self.route[-1]
        start_city = self.route[0]
        
        # Znajdź odległość powrotu
        return_distance = 0
        for dest, dist in graph[last_city]:
            if dest == start_city:
                return_distance = dist
                break
        
        # Oblicz prędkość na podstawie wagi plecaka po odwiedzeniu wszystkich miast
        last_weight = weights_at_cities[-1]
        return_speed = max(Vmin, Vmax - last_weight * v_w)  # Upewnij się, że prędkość nie spadnie poniżej Vmin
        
        # Dodaj czas powrotu
        total_time += return_distance / return_speed
        
        return total_time

    def pick_items(self, current_city, current_weight, max_items=2):
        """Wybiera przedmioty z bieżącego miasta zgodnie z ograniczeniami plecaka i stosunkiem zysku do wagi."""
        items = itemset.get(current_city, [])
        # Sortuj przedmioty po stosunku zysku do wagi malejąco
        items = sorted(items, key=lambda x: x[1] / x[2], reverse=True)
        
        picked_items = []
        total_profit = 0
        total_weight = current_weight
        
        # Ogranicz liczbę przedmiotów, które można zabrać z każdego miasta
        for i, (item_id, profit, weight) in enumerate(items):
            if i >= max_items:  # Maksymalnie max_items przedmiotów z każdego miasta
                break
            if total_weight + weight <= W:
                picked_items.append((current_city, item_id))
                total_weight += weight
                total_profit += profit
        
        return picked_items, total_weight, total_profit

    def calculate_fitness(self):
        total_distance = 0
        total_profit = 0
        total_weight = 0
        picked_items = []
        weights_at_cities = [0]  # Początkowa waga plecaka

        # Najpierw oblicz odległość całej trasy
        for i in range(len(self.route) - 1):
            current_city = self.route[i]
            next_city = self.route[i + 1]
            for dest, dist in graph[current_city]:
                if dest == next_city:
                    total_distance += dist
                    break

        # Dodaj dystans powrotny do miasta startowego
        start_city = self.route[0]
        last_city = self.route[-1]
        for dest, dist in graph[last_city]:
            if dest == start_city:
                total_distance += dist
                break

        # Teraz wybierz przedmioty, uwzględniając wpływ na prędkość
        for i, current_city in enumerate(self.route):
            # Wybierz przedmioty z aktualnego miasta, ograniczając liczbę przedmiotów
            city_items, new_weight, city_profit = self.pick_items(current_city, total_weight, max_items=2)
            
            # Dodaj wybrane przedmioty do listy
            picked_items.extend(city_items)
            total_profit += city_profit
            total_weight = new_weight
            
            # Zapisz aktualną wagę plecaka po opuszczeniu miasta
            weights_at_cities.append(total_weight)

        # Oblicz czas podróży zgodnie z funkcją celu
        travel_time = self.calculate_travel_time(weights_at_cities)
        
        # Oblicz koszt podróży
        travel_cost = R * travel_time
        
        # Oblicz wartość funkcji celu
        fitness = total_profit - travel_cost
        
        # Zapisz wagi plecaka w każdym mieście
        self.weights_at_cities = weights_at_cities
        
        return fitness, picked_items, total_weight, total_profit, travel_time, travel_cost

    def update_velocity(self, global_best_route, w, c1, c2):
        self.velocity = []
        
        # Dodaj więcej losowych ruchów, aby zwiększyć eksplorację
        if random.random() < 0.5:  # 50% szans na dodanie losowego ruchu
            # Dodaj kilka losowych zamian
            num_swaps = random.randint(1, 3)
            for _ in range(num_swaps):
                i = random.randint(0, len(self.route) - 1)
                j = random.randint(0, len(self.route) - 1)
                if i != j:
                    self.velocity.append(('swap', i, j))
        
        # Dodaj ruchy w kierunku najlepszego osobistego rozwiązania
        for i in range(len(self.route)):
            r1 = random.random()
            if r1 < c1 and self.route[i] in self.best_route:
                self.velocity.append(('swap', i, self.best_route.index(self.route[i])))
        
        # Dodaj ruchy w kierunku globalnego najlepszego rozwiązania
        for i in range(len(self.route)):
            r2 = random.random()
            if r2 < c2 and self.route[i] in global_best_route:
                self.velocity.append(('swap', i, global_best_route.index(self.route[i])))
        
        # Dodaj ruchy typu "reverse" (odwrócenie fragmentu trasy)
        if random.random() < 0.3:  # 30% szans na dodanie ruchu typu "reverse"
            start = random.randint(0, len(self.route) - 2)
            end = random.randint(start + 1, len(self.route) - 1)
            self.velocity.append(('reverse', start, end))

    def update_position(self):
        new_route = list(self.route)
        
        # Zastosuj wszystkie ruchy z prędkości
        for move in self.velocity:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                if i < len(new_route) and j < len(new_route):
                    new_route[i], new_route[j] = new_route[j], new_route[i]
            elif move[0] == 'reverse':
                start, end = move[1], move[2]
                if start < len(new_route) and end < len(new_route):
                    new_route[start:end+1] = reversed(new_route[start:end+1])
        
        # Upewnij się, że trasa zawiera wszystkie miasta
        self.route = self.ensure_all_cities(new_route)
        
        # Oblicz nową wartość funkcji celu
        new_fitness, new_picked_items, new_total_weight, new_total_profit, new_travel_time, new_travel_cost = self.calculate_fitness()
        
        # Aktualizuj najlepsze rozwiązanie, jeśli nowe jest lepsze
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_route = list(self.route)
            self.picked_items = new_picked_items
            self.total_weight = new_total_weight
            self.total_profit = new_total_profit
            self.travel_time = new_travel_time
            self.travel_cost = new_travel_cost

class PSO:
    def __init__(self, num_particles, w, c1, c2, num_iterations):
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_iterations = num_iterations
        self.particles = []
        self.no_improvement_count = 0
        self.best_fitness_history = []

        # Inicjalizacja cząstek z różnymi strategiami
        for _ in range(num_particles):
            if _ < num_particles // 3:
                # 1/3 cząstek z trasą opartą na odległości
                route = self.generate_distance_based_route()
            elif _ < 2 * num_particles // 3:
                # 1/3 cząstek z trasą opartą na wartości przedmiotów
                route = self.generate_value_based_route()
            else:
                # 1/3 cząstek z losową trasą
                route = self.generate_random_route()
            
            self.particles.append(Particle(route))

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

    def generate_random_route(self):
        """Generuje losową trasę."""
        cities = list(graph.keys())
        random.shuffle(cities)
        return cities

    def generate_distance_based_route(self):
        """Generuje trasę opartą na odległości między miastami."""
        cities = list(graph.keys())
        route = [random.choice(cities)]
        unvisited = set(cities) - {route[0]}
        
        while unvisited:
            current = route[-1]
            # Znajdź najbliższe nieodwiedzone miasto
            next_city = min(unvisited, key=lambda c: next(dist for dest, dist in graph[current] if dest == c))
            route.append(next_city)
            unvisited.remove(next_city)
        
        return route

    def generate_value_based_route(self):
        """Generuje trasę opartą na wartości przedmiotów w miastach."""
        # Oblicz średnią wartość przedmiotów w każdym mieście
        city_values = {}
        for city, items in itemset.items():
            if items:
                total_value = sum(item[1] for item in items)  # item[1] to wartość przedmiotu
                city_values[city] = total_value / len(items)
            else:
                city_values[city] = 0
        
        # Sortuj miasta według wartości przedmiotów
        sorted_cities = sorted(city_values.keys(), key=lambda c: city_values[c], reverse=True)
        
        # Dodaj losowość do trasy
        route = list(sorted_cities)
        random.shuffle(route)
        
        return route

    def run(self):
        for iteration in range(self.num_iterations):
            # Aktualizacja parametrów w czasie
            current_w = self.w * (1 - iteration / self.num_iterations)  # Zmniejsz wagę bezwładności w czasie
            current_c1 = self.c1 * (1 - iteration / self.num_iterations)  # Zmniejsz składnik poznawczy w czasie
            current_c2 = self.c2 * (1 + iteration / self.num_iterations)  # Zwiększ składnik społeczny w czasie
            
            # Zastosuj lokalne wyszukiwanie do najlepszych cząstek
            if iteration % 5 == 0:  # Co 5 iteracji
                # Sortuj cząstki według wartości funkcji celu
                sorted_particles = sorted(self.particles, key=lambda p: p.best_fitness, reverse=True)
                # Zastosuj lokalne wyszukiwanie do najlepszych 10% cząstek
                for i in range(min(10, len(sorted_particles) // 10)):
                    self.local_search(sorted_particles[i])
            
            # Aktualizuj pozycje wszystkich cząstek
            for particle in self.particles:
                particle.update_velocity(self.global_best_route, current_w, current_c1, current_c2)
                particle.update_position()
                if particle.best_fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_route = particle.best_route
                    self.global_best_items = particle.picked_items
                    self.global_best_weight = particle.total_weight
                    self.global_best_profit = particle.total_profit
                    self.global_best_time = particle.travel_time
                    self.global_best_cost = particle.travel_cost
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            
            # Zapisz wartość funkcji celu
            self.best_fitness_history.append(self.global_best_fitness)
            
            # Wyświetl postęp co 10 iteracji
            if (iteration + 1) % 10 == 0:
                print(f"Iteracja {iteration + 1}/{self.num_iterations}, Najlepsza wartość funkcji celu: {self.global_best_fitness:.2f}")
            
            # Jeśli nie ma poprawy przez 20 iteracji, zresetuj część cząstek
            if self.no_improvement_count > 20:
                print("Brak poprawy przez 20 iteracji, resetuję część cząstek...")
                self.reset_particles()
                self.no_improvement_count = 0
            
            # Jeśli znaleźliśmy rozwiązanie z dodatnią wartością funkcji celu, możemy zakończyć wcześniej
            if self.global_best_fitness > 0 and iteration > 50:
                print(f"Znaleziono rozwiązanie z dodatnią wartością funkcji celu: {self.global_best_fitness:.2f}")
                break

        return (self.global_best_route, self.global_best_fitness, self.global_best_items, 
                self.global_best_weight, self.global_best_profit, self.global_best_time, self.global_best_cost)
    
    def local_search(self, particle):
        """Zastosuj lokalne wyszukiwanie do cząstki."""
        improved = True
        while improved:
            improved = False
            # Próbuj zamienić każde miasto z każdym innym
            for i in range(len(particle.route)):
                for j in range(i + 1, len(particle.route)):
                    # Zapisz oryginalną trasę
                    original_route = list(particle.route)
                    # Zamień miasta
                    particle.route[i], particle.route[j] = particle.route[j], particle.route[i]
                    # Oblicz nową wartość funkcji celu
                    new_fitness, new_picked_items, new_total_weight, new_total_profit, new_travel_time, new_travel_cost = particle.calculate_fitness()
                    # Jeśli nowa wartość jest lepsza, zaakceptuj zmianę
                    if new_fitness > particle.best_fitness:
                        particle.best_fitness = new_fitness
                        particle.best_route = list(particle.route)
                        particle.picked_items = new_picked_items
                        particle.total_weight = new_total_weight
                        particle.total_profit = new_total_profit
                        particle.travel_time = new_travel_time
                        particle.travel_cost = new_travel_cost
                        improved = True
                    else:
                        # W przeciwnym razie przywróć oryginalną trasę
                        particle.route = original_route
    
    def reset_particles(self):
        """Resetuje część cząstek, aby uniknąć lokalnych minimów."""
        # Sortuj cząstki według wartości funkcji celu
        sorted_particles = sorted(self.particles, key=lambda p: p.best_fitness, reverse=True)
        
        # Zachowaj najlepsze 20% cząstek
        keep_count = max(1, len(sorted_particles) // 5)
        
        # Resetuj pozostałe cząstki
        for i in range(keep_count, len(sorted_particles)):
            if i < len(sorted_particles) // 2:
                # 50% resetowanych cząstek z trasą opartą na odległości
                route = self.generate_distance_based_route()
            else:
                # 50% resetowanych cząstek z trasą opartą na wartości przedmiotów
                route = self.generate_value_based_route()
            
            # Zastąp cząstkę nową
            self.particles[i] = Particle(route)
            
            # Aktualizuj globalne najlepsze rozwiązanie, jeśli nowa cząstka jest lepsza
            if self.particles[i].best_fitness > self.global_best_fitness:
                self.global_best_fitness = self.particles[i].best_fitness
                self.global_best_route = self.particles[i].best_route
                self.global_best_items = self.particles[i].picked_items
                self.global_best_weight = self.particles[i].total_weight
                self.global_best_profit = self.particles[i].total_profit
                self.global_best_time = self.particles[i].travel_time
                self.global_best_cost = self.particles[i].travel_cost

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

# Parametry PSO dla dużego problemu
num_particles = 10  # Liczba cząstek
w = 0.7  # Waga bezwładności
c1 = 2.0  # Składnik poznawczy
c2 = 2.0  # Składnik społeczny
num_iterations = 400  # Liczba iteracji

# Testowanie algorytmu PSO
print("Uruchamianie algorytmu PSO...")
pso = PSO(num_particles, w, c1, c2, num_iterations)
(best_route_pso, best_fitness_pso, best_items_pso, 
 total_weight_pso, total_profit_pso, total_time_pso, travel_cost_pso) = pso.run()

total_distance_pso = calculate_total_distance(best_route_pso)
print_solution(best_route_pso, total_distance_pso, best_items_pso, total_profit_pso, total_weight_pso, total_time_pso, travel_cost_pso, best_fitness_pso)
