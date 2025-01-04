import random
from common.data_loader import load_data  # Funkcja wczytująca dane

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/5miast.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

def calculate_total_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    # Dodaj odległość powrotu do miasta początkowego
    start_city = route[0]
    last_city = route[-1]
    for dest, dist in graph[last_city]:
        if dest == start_city:
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

class NearestNeighbor:
    def __init__(self):
        self.route = []
        self.visited = set()
        self.total_distance = 0

    def find_route(self, start):
        current_city = start
        self.route.append(current_city)
        self.visited.add(current_city)

        while len(self.visited) < len(graph):
            next_city = None
            shortest_distance = float("inf")
            for dest, dist in graph[current_city]:
                if dest not in self.visited and dist < shortest_distance:
                    next_city = dest
                    shortest_distance = dist
            if next_city is None:
                break
            self.route.append(next_city)
            self.visited.add(next_city)
            self.total_distance += shortest_distance
            current_city = next_city

        # Dodaj powrót do miasta początkowego
        if len(self.route) == len(graph):
            start_city = self.route[0]
            last_city = self.route[-1]
            for dest, dist in graph[last_city]:
                if dest == start_city:
                    self.route.append(start_city)
                    self.total_distance += dist
                    break

    def run(self):
        start = random.choice(list(graph.keys()))  # Losowo wybiera miasto początkowe
        self.find_route(start)
        return self.route, self.total_distance

def print_solution(route, total_distance, picked_items, total_profit, total_weight):
    time, cost = calculate_time_and_cost(total_distance, total_weight)
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(time))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for city, item in picked_items:
        print(f"Miasto {city}: Przedmiot {item}")
    print("Całkowity zysk : ", total_profit)
    print("Waga przenoszona w plecaku : ", total_weight)

# Testowanie algorytmu Najbliższego Sąsiada
print("Uruchamianie algorytmu Najbliższego Sąsiada...")
nn = NearestNeighbor()
best_route_nn, total_distance_nn = nn.run()
picked_items_nn, total_profit_nn, total_weight_nn = solve_knapsack(best_route_nn)
print_solution(best_route_nn, total_distance_nn, picked_items_nn, total_profit_nn, total_weight_nn)
