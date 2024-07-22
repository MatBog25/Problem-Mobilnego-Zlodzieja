import random

# Definicje grafu i zestawu przedmiotów
graph = {
    1: [(2, 200), (3, 245), (16, 225)], # Białystok
    2: [(1, 200), (3, 175), (5, 178), (10, 136), (14, 272), (16, 216)], # Warszawa
    3: [(1, 245), (2, 175), (4, 178)],  # Lublin
    4: [(3, 178), (5, 157), (6, 167)],  # Rzeszów
    5: [(2, 178), (4, 157), (6, 167), (10, 154)], # Kielce
    6: [(4, 167), (5, 167), (7, 81)], # Kraków
    7: [(6, 81), (8, 117), (10, 205)], # Katowice 
    8: [(7, 117), (9, 98), (10, 222)], # Opole
    9: [(8, 98), (10, 222), (11, 184), (12, 187)], # Wrocław
    10: [(2, 136), (5, 154), (7, 205), (8, 222), (9, 222), (11, 216), (14, 217)],  # Łódź
    11: [(9, 184), (10, 216), (12, 153), (13, 266), (14, 138)],  # Poznań
    12: [(9, 187), (11, 153), (13, 213)], # Zielona Góra
    13: [(11, 266), (12, 213), (14, 259), (15, 376)], # Szczecin
    14: [(2, 272), (10, 217), (11, 138), (13, 259), (15, 167), (16, 212)], # Bydgoszcz
    15: [(13, 376), (14, 167), (16, 182)], # Gdańsk
    16: [(1, 225), (2, 216), (14, 212), (15, 182)], # Olsztyn
}

itemset = {
    1: [300, 50, (2, 6, 9, 10, 13)],  # wartość 300, waga 50, miasta: Warszawa, Kraków, Wrocław, Łódź, Szczecin
    2: [600, 25, (2, 6, 10)],         # wartość 600, waga 25, miasta: Warszawa, Kraków, Łódź
    3: [200, 100, (1, 11, 14, 16)],   # wartość 200, waga 100, miasta: Białystok, Poznań, Bydgoszcz, Olsztyn
    4: [100, 10, (3, 4, 12, 16)],     # wartość 100, waga 10, miasta: Lublin, Rzeszów, Zielona Góra, Olsztyn
    5: [250, 75, (5, 7, 8)],          # wartość 250, waga 75, miasta: Kielce, Katowice, Opole
    6: [350, 60, (2, 6, 11, 14)],     # wartość 350, waga 60, miasta: Warszawa, Kraków, Poznań, Bydgoszcz
    7: [150, 20, (4, 7, 8, 16)],      # wartość 150, waga 20, miasta: Rzeszów, Katowice, Opole, Olsztyn
    8: [500, 30, (2, 6, 10, 15)],     # wartość 500, waga 30, miasta: Warszawa, Kraków, Łódź, Gdańsk
    9: [120, 15, (3, 5, 12, 16)],     # wartość 120, waga 15, miasta: Lublin, Kielce, Zielona Góra, Olsztyn
    10: [180, 40, (1, 3, 11, 13)],    # wartość 180, waga 40, miasta: Białystok, Lublin, Poznań, Szczecin
    11: [270, 55, (6, 9, 14, 15)],    # wartość 270, waga 55, miasta: Kraków, Wrocław, Bydgoszcz, Gdańsk
}

Vmax = 15
Vmin = 5
R = 0.1
W = 500

def fitness(route, items):
    distance = 0
    for i in range(len(route) - 1):
        if not any(dest == route[i + 1] for dest, dist in graph[route[i]]):
            return 0  # Jesli niepoprawna trasa zwroc 0
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
    distances = [0] * len(route)
    for i in range(1, len(route)):
        for dest, dist in graph[route[i - 1]]:
            if dest == route[i]:
                distances[i] = distances[i - 1] + dist
                break

    if distances[-1] == 0:
        distances[-1] = 1  # Dzielenie przez 0

    finalitemset = []
    time = distances[-1] * 2 * (Vmax + Vmin)
    
    for key, value in itemset.items():
        for city in value[2]:
            if city in route:
                route_index = route.index(city)
                if distances[route_index] == 0:
                    distances[route_index] = 1  # Dzielenie przez 0
                score = int(value[0] - (0.25 * value[0] * (distances[route_index] / distances[-1])) - (R * time * value[1] / W))
                finalitemset.append([key, city, value[1], score, value[0]])

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

def calculate_total_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        for dest, dist in graph[route[i]]:
            if dest == route[i + 1]:
                total_distance += dist
                break
    return total_distance

# Testowanie algorytmu Najbliższego Sąsiada

print("Uruchamianie algorytmu Najbliższego Sąsiada...")
nn = NearestNeighbor()
best_route_nn, total_distance_nn = nn.run()
picked_items_nn, total_profit_nn, total_weight_nn = solve_knapsack(best_route_nn)
print_solution(best_route_nn, total_distance_nn, picked_items_nn, total_profit_nn, total_weight_nn)
