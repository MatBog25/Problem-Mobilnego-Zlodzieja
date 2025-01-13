import random
from common.data_loader import load_data  # Wcześniej zaimplementowana funkcja

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/280_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

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

def calculate_time_and_cost(distance, total_weight):
    """Oblicza czas podróży i koszt wynajmu."""
    speed = Vmax - (total_weight / W) * (Vmax - Vmin)
    time = distance / speed
    cost = R * time
    return time, cost

def solve_knapsack(route):
    """Rozwiązuje problem plecakowy dla podanej trasy."""
    picked_items = []
    total_profit = 0
    total_weight = 0

    for city in route:
        for item in itemset.get(city, []):
            item_id, profit, weight = item
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
        best_distance = calculate_total_distance(best_route)
        best_time, best_cost = calculate_time_and_cost(best_distance, best_total_weight)
        best_fitness = best_total_profit - best_cost

        for _ in range(self.iterations):
            new_route = self.generate_random_route()
            picked_items, total_profit, total_weight = solve_knapsack(new_route)
            total_distance = calculate_total_distance(new_route)
            time, cost = calculate_time_and_cost(total_distance, total_weight)
            new_fitness = total_profit - cost

            if new_fitness > best_fitness:
                best_route = new_route
                best_fitness = new_fitness
                best_picked_items = picked_items
                best_total_profit = total_profit
                best_total_weight = total_weight
                best_distance = total_distance
                best_time = time

        return best_route, best_distance, best_picked_items, best_total_profit, best_total_weight, best_time

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk : ", total_profit)
    print("Waga przenoszona w plecaku : ", total_weight)

# Parametry optymalizacji
rs_iterations = 100  # Liczba iteracji

# Testowanie algorytmu Random Search
print("Uruchamianie algorytmu losowego przeszukiwania...")
rs = RandomSearch(iterations=rs_iterations)
best_route_rs, total_distance_rs, picked_items_rs, total_profit_rs, total_weight_rs, total_time_rs = rs.run()
print_solution(best_route_rs, total_distance_rs, picked_items_rs, total_profit_rs, total_weight_rs, total_time_rs)
