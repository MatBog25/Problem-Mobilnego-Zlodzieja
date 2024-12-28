import random
from data_loader import load_data  # Wcześniej zaimplementowana funkcja

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("280_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

def fitness(route, items):
    distance = 0
    for i in range(len(route) - 1):
        if not any(dest == route[i + 1] for dest, dist in graph[route[i]]):
            return 0  # Niepoprawna trasa
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                distance += dist
                break

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
        best_fitness = fitness(best_route, [])

        for _ in range(self.iterations):
            new_route = self.generate_random_route()
            new_fitness = fitness(new_route, [])
            if new_fitness > best_fitness:
                best_route = new_route
                best_fitness = new_fitness

        return best_route, best_fitness

def print_solution(route, total_distance, picked_items, total_profit, total_weight):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Złodziej powinien zabrać następujące przedmioty:")
    for i in picked_items:
        print("Miasto : " + str(i[0]) + "   Przedmioty : " + ', '.join(str(e) for e in i[1]))
    print("Całkowity zysk : " + str(total_profit))
    print("Waga przenoszona w plecaku : " + str(total_weight))

def calculate_total_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    return total_distance

# Parametry optymalizacji
rs_iterations = 10000  # Dostosuj liczbę iteracji do wielkości problemu

# Testowanie algorytmu Random Search
print("Uruchamianie algorytmu losowego przeszukiwania...")
rs = RandomSearch(iterations=rs_iterations)
best_route_rs, best_fitness_rs = rs.run()
total_distance_rs = calculate_total_distance(best_route_rs)
picked_items_rs, total_profit_rs, total_weight_rs = solve_knapsack(best_route_rs)
print_solution(best_route_rs, total_distance_rs, picked_items_rs, total_profit_rs, total_weight_rs)
