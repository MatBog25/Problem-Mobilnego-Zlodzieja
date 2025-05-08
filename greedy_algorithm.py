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
    # Dodaj odległość powrotu do miasta startowego
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    return total_distance

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

class GreedyAlgorithm:
    def __init__(self):
        self.visited = set()
        self.route = []
        self.total_distance = 0

    def path(self, current_city):
        self.route.append(current_city)
        self.visited.add(current_city)

        shortest_distance = float('inf')
        next_city = None
        for dest, dist in graph[current_city]:
            if dest not in self.visited and dist < shortest_distance:
                shortest_distance = dist
                next_city = dest

        if next_city:
            self.total_distance += shortest_distance
            self.path(next_city)

    def run(self):
        start = random.choice(list(graph.keys()))
        self.path(start)
        return self.route, self.total_distance

def calculate_time_and_cost(distance, total_weight):
    """Oblicza czas podróży i koszt wynajmu."""
    speed = Vmax - (total_weight / W) * (Vmax - Vmin)
    time = distance / speed
    cost = R * time
    return time, cost

def print_solution(route, total_distance, picked_items, total_profit, total_weight):
    time, cost = calculate_time_and_cost(total_distance, total_weight)
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(time))
    print("Koszt wynajmu: {:.2f}".format(cost))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk : ", total_profit)
    print("Waga przenoszona w plecaku : ", total_weight)

# Testowanie algorytmu zachłannego
print("Uruchamianie algorytmu zachłannego...")
greedy = GreedyAlgorithm()
best_route_greedy, total_distance_greedy = greedy.run()
picked_items_greedy, total_profit_greedy, total_weight_greedy = solve_knapsack(best_route_greedy)
print_solution(best_route_greedy, total_distance_greedy, picked_items_greedy, total_profit_greedy, total_weight_greedy)
