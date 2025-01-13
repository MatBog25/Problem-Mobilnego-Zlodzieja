import random
from common.data_loader import load_data

# Wczytaj dane z pliku
graph, itemset, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data("data/280_1.txt")

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
W = knapsack_capacity
R = renting_ratio

class Ant:
    def __init__(self, alpha, beta, num_cities):
        self.alpha = alpha
        self.beta = beta
        self.num_cities = num_cities
        self.route = []
        self.visited = set()
        self.items_picked = []
        self.distance_traveled = 0
        self.total_profit = 0
        self.current_weight = 0

    def select_next_city(self, current_city, pheromone):
        probabilities = []
        for dest, dist in graph[current_city]:
            if dest not in self.visited and dist > 0:  # Unikamy odległości 0
                tau = pheromone[current_city][dest] ** self.alpha
                eta = (1.0 / dist) ** self.beta
                probabilities.append((dest, tau * eta))

        if not probabilities:
            return None

        total = sum(prob for _, prob in probabilities)
        probabilities = [(city, prob / total) for city, prob in probabilities]
        r = random.random()
        for city, prob in probabilities:
            r -= prob
            if r <= 0:
                return city
        return probabilities[-1][0]

    def pick_items(self, current_city):
        """Wybiera przedmioty z bieżącego miasta zgodnie z ograniczeniami plecaka."""
        items = itemset.get(current_city, [])
        # Sortuj przedmioty po zysku do wagi malejąco
        items = sorted(items, key=lambda x: x[1] / x[2], reverse=True)
        
        for item_id, profit, weight in items:
            if self.current_weight + weight <= W:
                self.items_picked.append(item_id)
                self.total_profit += profit
                self.current_weight += weight

    def travel(self, pheromone):
        self.route = [random.randint(1, self.num_cities)]
        self.visited = set(self.route)
        self.distance_traveled = 0

        while len(self.route) < self.num_cities:
            current_city = self.route[-1]
            self.pick_items(current_city)
            next_city = self.select_next_city(current_city, pheromone)
            if next_city is None:
                self.distance_traveled = float('inf')
                return
            self.route.append(next_city)
            self.visited.add(next_city)
            self.distance_traveled += next(dist for dest, dist in graph[current_city] if dest == next_city)

        # Powrót do miasta początkowego
        start_city = self.route[0]
        last_city = self.route[-1]
        self.distance_traveled += next(dist for dest, dist in graph[last_city] if dest == start_city)

class ACO:
    def __init__(self, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit

    def initialize_pheromone(self, num_cities):
        return {i: {j: 1.0 for j in range(1, num_cities + 1) if i != j} for i in range(1, num_cities + 1)}

    def run(self):
        num_cities = len(graph)
        pheromone = self.initialize_pheromone(num_cities)
        best_route = None
        best_distance = float('inf')
        best_items = []
        best_profit = 0
        best_time = float('inf')
        best_total_weight = 0

        for iteration in range(self.num_iterations):
            ants = [Ant(self.alpha, self.beta, num_cities) for _ in range(self.num_ants)]
            for ant in ants:
                ant.travel(pheromone)
                if ant.distance_traveled == float('inf'):
                    continue

                # Oblicz czas i koszt
                speed = max(Vmin, Vmax - (ant.current_weight / W) * (Vmax - Vmin))
                time = ant.distance_traveled / speed
                cost = R * time

                # Aktualizacja najlepszego wyniku
                if ant.total_profit > best_profit:
                    best_profit = ant.total_profit
                    best_route = ant.route
                    best_distance = ant.distance_traveled
                    best_items = ant.items_picked
                    best_time = time
                    best_total_weight = ant.current_weight

            # Aktualizacja feromonów
            for i in pheromone:
                for j in pheromone[i]:
                    pheromone[i][j] *= (1 - self.evaporation_rate)

            for ant in ants:
                if ant.distance_traveled < float('inf') and ant.total_profit > 0:
                    for i in range(len(ant.route) - 1):
                        city1 = ant.route[i]
                        city2 = ant.route[i + 1]
                        pheromone[city1][city2] += self.pheromone_deposit / ant.distance_traveled
                        pheromone[city2][city1] += self.pheromone_deposit / ant.distance_traveled

        return best_route, best_distance, best_items, best_profit, best_total_weight, best_time

def print_solution(route, total_distance, picked_items, total_profit, total_weight, total_time):
    print("Najkrótsza ścieżka: ", route)
    print("Całkowita odległość: ", total_distance)
    print("Czas podróży: {:.2f} jednostek czasu".format(total_time))
    print("Złodziej powinien zabrać następujące przedmioty:")
    for item in picked_items:
        print(f"Przedmiot {item}")
    print("Całkowity zysk: {:.2f}".format(total_profit))
    print("Całkowita waga przedmiotów: ", total_weight)

# Parametry ACO
num_ants = 10
num_iterations = 1
alpha = 5.0
beta = 5.0
evaporation_rate = 0.5
pheromone_deposit = 10

# Uruchomienie algorytmu ACO
aco = ACO(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit)
best_route, best_distance, best_items, best_profit, best_total_weight, best_time = aco.run()

# Wyświetlenie wyników
print_solution(best_route, best_distance, best_items, best_profit, best_total_weight, best_time)
