import random
from common.data_loader import load_data  # Wcześniej zaimplementowana funkcja

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/2000_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio
# Obliczanie v_w - spadek prędkości w funkcji ciężaru plecaka
v_w = (Vmax - Vmin) / W

def calculate_total_distance(route):
    """Oblicza całkowitą odległość dla podanej trasy."""
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    # Dodaj odległość powrotu do miasta początkowego
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    return total_distance

def calculate_travel_time(route, weights_at_cities):
    """Oblicza czas podróży zgodnie z funkcją celu."""
    total_time = 0
    
    # Obliczanie czasu podróży między kolejnymi miastami (od x_1 do x_n)
    for i in range(len(route) - 1):
        current_city = route[i]
        next_city = route[i + 1]
        
        # Znajdź odległość między miastami
        distance = 0
        for dest, dist in graph[current_city]:
            if dest == next_city:
                distance = dist
                break
        
        # Oblicz prędkość na podstawie aktualnej wagi plecaka
        current_weight = weights_at_cities[i]
        speed = Vmax - current_weight * v_w  # Dokładnie zgodnie z funkcją celu
        
        # Dodaj czas podróży
        total_time += distance / speed
    
    # Dodaj czas powrotu do miasta początkowego (od x_n do x_1)
    last_city = route[-1]
    start_city = route[0]
    
    # Znajdź odległość powrotu
    return_distance = 0
    for dest, dist in graph[last_city]:
        if dest == start_city:
            return_distance = dist
            break
    
    # Oblicz prędkość na podstawie wagi plecaka po odwiedzeniu wszystkich miast
    last_weight = weights_at_cities[-1]
    return_speed = Vmax - last_weight * v_w  # Dokładnie zgodnie z funkcją celu
    
    # Dodaj czas powrotu
    total_time += return_distance / return_speed
    
    return total_time

def calculate_objective_function(route, picked_items, total_profit):
    """Oblicza wartość funkcji celu zgodnie z podanym wzorem matematycznym."""
    # Obliczanie wag plecaka w każdym mieście
    weights_at_cities = [0] * len(route)
    current_weight = 0
    
    # Dla każdego miasta w trasie
    for i, city in enumerate(route):
        # Dodaj wagę przedmiotów zabranych w tym mieście
        for item_city, item_id in picked_items:
            if item_city == city:
                # Znajdź przedmiot w itemset
                for item in itemset.get(city, []):
                    if item[0] == item_id:  # item[0] to item_id
                        current_weight += item[2]  # item[2] to waga
                        break
        
        # Zapisz aktualną wagę plecaka po opuszczeniu miasta
        weights_at_cities[i] = current_weight
    
    # Oblicz czas podróży
    travel_time = calculate_travel_time(route, weights_at_cities)
    
    # Oblicz koszt podróży
    travel_cost = R * travel_time
    
    # Oblicz wartość funkcji celu
    objective_value = total_profit - travel_cost
    
    return objective_value, travel_time, travel_cost

def solve_knapsack(route):
    """Rozwiązuje problem plecakowy dla podanej trasy."""
    picked_items = []
    total_profit = 0
    total_weight = 0

    # Sortuj przedmioty według stosunku wartości do wagi (malejąco)
    all_items = []
    for city in route:
        for item in itemset.get(city, []):
            item_id, profit, weight = item
            ratio = profit / weight if weight > 0 else 0
            all_items.append((city, item_id, profit, weight, ratio))
    
    # Sortuj przedmioty według stosunku wartości do wagi (malejąco)
    all_items.sort(key=lambda x: x[4], reverse=True)
    
    # Wybierz przedmioty z najwyższym stosunkiem wartości do wagi
    for city, item_id, profit, weight, ratio in all_items:
        if total_weight + weight <= W:
            picked_items.append((city, item_id))
            total_weight += weight
            total_profit += profit

    return picked_items, total_profit, total_weight

class RandomSearch:
    def __init__(self, iterations):
        self.iterations = iterations

    def generate_random_route(self):
        """Generuje losową trasę, która zawiera wszystkie miasta."""
        cities = list(graph.keys())
        random.shuffle(cities)
        return cities

    def run(self):
        best_route = self.generate_random_route()
        best_picked_items, best_total_profit, best_total_weight = solve_knapsack(best_route)
        best_objective, best_time, best_cost = calculate_objective_function(best_route, best_picked_items, best_total_profit)
        best_distance = calculate_total_distance(best_route)

        for _ in range(self.iterations):
            new_route = self.generate_random_route()
            picked_items, total_profit, total_weight = solve_knapsack(new_route)
            objective_value, travel_time, travel_cost = calculate_objective_function(new_route, picked_items, total_profit)
            total_distance = calculate_total_distance(new_route)

            if objective_value > best_objective:
                best_route = new_route
                best_objective = objective_value
                best_picked_items = picked_items
                best_total_profit = total_profit
                best_total_weight = total_weight
                best_distance = total_distance
                best_time = travel_time
                best_cost = travel_cost

        return best_route, best_distance, best_picked_items, best_total_profit, best_total_weight, best_time, best_cost, best_objective

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

# Parametry optymalizacji
rs_iterations = 1000 # Liczba iteracji

# Testowanie algorytmu Random Search
print("Uruchamianie algorytmu losowego przeszukiwania...")
rs = RandomSearch(iterations=rs_iterations)
best_route_rs, total_distance_rs, picked_items_rs, total_profit_rs, total_weight_rs, total_time_rs, total_cost_rs, objective_value_rs = rs.run()
print_solution(best_route_rs, total_distance_rs, picked_items_rs, total_profit_rs, total_weight_rs, total_time_rs, total_cost_rs, objective_value_rs)
