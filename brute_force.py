import itertools
from data_loader import load_data  # Wcześniej zaimplementowana funkcja

def calculate_fitness(route, items):
    """Oblicza fitness dla danej trasy i zestawu przedmiotów."""
    distance = calculate_total_distance(route)
    total_weight = sum(item[2] for city in route for item in items_by_city.get(city, []) if item[0] in items)
    total_value = sum(item[1] for city in route for item in items_by_city.get(city, []) if item[0] in items)
    time = distance / ((Vmax + Vmin) / 2)

    if total_weight > W:
        return -float("inf")  # Kara za przekroczenie wagi
    return total_value - R * time

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

def generate_item_combinations(route):
    """Generuje wszystkie możliwe kombinacje przedmiotów dla podanej trasy."""
    available_items = []
    for city in route:
        if city in items_by_city:
            for item_id, profit, weight in items_by_city[city]:
                if item_id not in available_items:
                    available_items.append(item_id)
    all_combinations = []
    for r in range(len(available_items) + 1):
        all_combinations.extend(itertools.combinations(available_items, r))
    return all_combinations

def solve_bruteforce():
    """Rozwiązuje problem brute force."""
    best_fitness = -float("inf")
    best_route = None
    best_items = None

    cities = list(graph.keys())
    city_permutations = list(itertools.permutations(cities))
    total_permutations = len(city_permutations)
    print(f"Liczba wszystkich tras: {total_permutations}")

    iteration = 0
    for route in city_permutations:
        route = list(route) + [route[0]]  # Powrót do miasta startowego
        item_combinations = generate_item_combinations(route)

        for items in item_combinations:
            current_fitness = calculate_fitness(route, items)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_route = route
                best_items = items

        iteration += 1
        if iteration % (total_permutations // 100) == 0:
            print(f"Przetworzono {iteration}/{total_permutations} tras ({(iteration / total_permutations) * 100:.2f}%)")

    return best_route, best_items, best_fitness

def print_summary(route, items, fitness):
    """Wyświetla wyniki optymalizacji."""
    total_distance = calculate_total_distance(route)
    total_weight = sum(item[2] for city in route for item in items_by_city.get(city, []) if item[0] in items)
    total_value = sum(item[1] for city in route for item in items_by_city.get(city, []) if item[0] in items)
    print(f"Najlepsza trasa: {route}")
    print(f"Całkowity dystans: {total_distance}")
    print(f"Całkowita waga przedmiotów: {total_weight}")
    print(f"Całkowita wartość przedmiotów: {total_value}")
    print(f"Fitness: {fitness}")

# Wczytanie danych z pliku
file_path = "10miast.txt"  # Podaj odpowiednią ścieżkę do pliku
graph, items_by_city, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file_path)

# Parametry problemu
Vmax = max_speed
Vmin = min_speed
R = renting_ratio
W = knapsack_capacity

# Uruchomienie algorytmu
print("Uruchamianie algorytmu brute force...")
import time
start = time.perf_counter()
best_route, best_items, best_fitness = solve_bruteforce()
koniec = time.perf_counter() - start
print_summary(best_route, best_items, best_fitness)
print(f"Czas wykonania: {koniec} sekund")