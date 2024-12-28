import random
from data_loader import load_data  # Funkcja wczytująca dane

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("280_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

def fitness(route, items):
    distance = calculate_total_distance(route)
    total_weight = sum(itemset[item][1] for item in items if item != -1)
    total_value = sum(itemset[item][0] for item in items if item != -1)
    time = distance / ((Vmax + Vmin) / 2)

    if total_weight > W:
        return 0
    return total_value - R * time

def solve_knapsack(route):
    """Rozwiązuje problem plecakowy dla podanej trasy."""
    distances = [0] * len(route)
    for i in range(1, len(route)):
        for dest, dist in graph[route[i - 1]]:
            if dest == route[i]:
                distances[i] = distances[i - 1] + dist
                break

    if distances[-1] == 0:
        distances[-1] = 1  # Zapobieganie dzieleniu przez zero

    finalitemset = []
    time = distances[-1] * 2 * (Vmax + Vmin)
    
    for key, value in itemset.items():
        for item in value:
            item_id, profit, weight = item
            for city in route:
                if city == item_id:  # Sprawdzamy, czy miasto jest na trasie
                    route_index = route.index(city)
                    if distances[route_index] == 0:
                        distances[route_index] = 1  # Zapobieganie dzieleniu przez zero
                    score = int(profit - (0.25 * profit * (distances[route_index] / distances[-1])) - (R * time * weight / W))
                    finalitemset.append([item_id, city, weight, score, profit])

    finalitemset.sort(key=lambda x: int(x[3]), reverse=True)

    wc = 0
    picked_items = []
    totalprof = 0

    for item in finalitemset:
        if wc + item[2] <= W:
            picked_items.append(item)
            wc += item[2]
            totalprof += item[4]

    result = {}
    for item in picked_items:
        if item[1] not in result:
            result[item[1]] = []
        result[item[1]].append(item[0])

    fin = [[city, sorted(items)] for city, items in result.items()]
    fin.sort(key=lambda x: x[0])

    return fin, totalprof, wc

def calculate_total_distance(route):
    """Oblicza całkowitą odległość dla podanej trasy."""
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    # Dodaj odległość powrotu do miasta startowego (jeśli wymagane)
    for dest, dist in graph[route[-1]]:
        if dest == route[0]:
            total_distance += dist
            break
    return total_distance

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

    def run(self):
        start = random.choice(list(graph.keys()))  # Losowo wybiera miasto początkowe
        self.find_route(start)
        return self.route, self.total_distance

def print_solution(route, total_distance, picked_items, total_profit, total_weight):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Złodziej powinien zabrać następujące przedmioty:")
    for i in picked_items:
        print("Miasto : " + str(i[0]) + "   Przedmioty : " + ', '.join(str(e) for e in i[1]))
    print("Całkowity zysk : " + str(total_profit))
    print("Waga przenoszona w plecaku : " + str(total_weight))

# Testowanie algorytmu Najbliższego Sąsiada
print("Uruchamianie algorytmu Najbliższego Sąsiada...")
nn = NearestNeighbor()
best_route_nn, total_distance_nn = nn.run()
picked_items_nn, total_profit_nn, total_weight_nn = solve_knapsack(best_route_nn)
print_solution(best_route_nn, total_distance_nn, picked_items_nn, total_profit_nn, total_weight_nn)
