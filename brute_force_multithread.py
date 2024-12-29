import itertools
import time
import math
import multiprocessing
from data_loader import load_data  # Wczytujemy wcześniej przygotowany moduł

def calculate_total_distance(route, graph):
    """Oblicza całkowitą odległość dla podanej trasy."""
    total_distance = 0
    for i in range(len(route) - 1):
        current_city = route[i]
        next_city = route[i + 1]
        for dest, dist in graph[current_city]:
            if dest == next_city:
                total_distance += dist
                break
    # Dodaj odległość powrotu do miasta startowego
    last_city = route[-1]
    first_city = route[0]
    for dest, dist in graph[last_city]:
        if dest == first_city:
            total_distance += dist
            break
    return total_distance

def generate_item_combinations(route, items_by_city):
    """Generuje wszystkie możliwe kombinacje przedmiotów dla podanej trasy."""
    available_items = []
    for city in route:
        if city in items_by_city:
            for (item_id, _, _) in items_by_city[city]:
                if item_id not in available_items:
                    available_items.append(item_id)
    all_combinations = []
    for r in range(len(available_items) + 1):
        all_combinations.extend(itertools.combinations(available_items, r))
    return all_combinations

def calculate_fitness(route, items, graph, items_by_city, W, Vmin, Vmax, R):
    """Oblicza fitness dla danej trasy i zestawu przedmiotów."""
    distance = calculate_total_distance(route, graph)
    total_weight = sum(item[2] for city in route 
                       for item in items_by_city.get(city, []) 
                       if item[0] in items)
    total_value = sum(item[1] for city in route 
                      for item in items_by_city.get(city, []) 
                      if item[0] in items)
    time = distance / ((Vmax + Vmin) / 2)

    if total_weight > W:
        return -float("inf")  # Kara za przekroczenie wagi
    return total_value - R * time

def process_permutation_chunk(chunk_of_permutations, graph, items_by_city, Vmin, Vmax, W, R):
    """
    Funkcja wywoływana w procesach potomnych.
    Otrzymuje "kawałek" listy permutacji i przetwarza go.
    """
    local_best_fitness = -float("inf")
    local_best_route = None
    local_best_items = None
    
    for route_tuple in chunk_of_permutations:
        route = list(route_tuple) + [route_tuple[0]]  # Powrót do miasta startowego
        item_combinations = generate_item_combinations(route, items_by_city)  # Przekazujemy items_by_city
        for items in item_combinations:
            current_fitness = calculate_fitness(route, items, graph, items_by_city, W, Vmin, Vmax, R)
            if current_fitness > local_best_fitness:
                local_best_fitness = current_fitness
                local_best_route = route
                local_best_items = items

    return (local_best_route, local_best_items, local_best_fitness)

def solve_bruteforce(graph, items_by_city, Vmin, Vmax, W, R):
    """Rozwiązuje problem brute force równolegle."""
    cities = list(graph.keys())
    city_permutations = list(itertools.permutations(cities))
    total_permutations = len(city_permutations)
    print(f"Liczba wszystkich tras: {total_permutations}")

    num_cores = multiprocessing.cpu_count()
    chunk_size = math.ceil(total_permutations / num_cores)
    permutation_chunks = [
        city_permutations[i : i + chunk_size] 
        for i in range(0, total_permutations, chunk_size)
    ]

    # Przygotowanie parametrów dla każdego procesu
    args = [(chunk, graph, items_by_city, Vmin, Vmax, W, R) for chunk in permutation_chunks]

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(process_permutation_chunk, args)

    best_route, best_items, best_fitness = max(results, key=lambda x: x[2])
    return best_route, best_items, best_fitness

def print_summary(route, items, fitness, graph, items_by_city):
    """Wyświetla wyniki optymalizacji."""
    total_distance = calculate_total_distance(route, graph)
    total_weight = sum(item[2] for city in route 
                       for item in items_by_city.get(city, []) 
                       if item[0] in items)
    total_value = sum(item[1] for city in route 
                      for item in items_by_city.get(city, []) 
                      if item[0] in items)
    print(f"Najlepsza trasa: {route}")
    print(f"Całkowity dystans: {total_distance}")
    print(f"Całkowita waga przedmiotów: {total_weight}")
    print(f"Całkowita wartość przedmiotów: {total_value}")
    print(f"Fitness: {fitness}")

def main():
    file_path = "12miast.txt"  # Ścieżka do pliku z danymi
    graph, items_by_city, knapsack_capacity, min_speed, max_speed, renting_ratio = load_data(file_path)

    # Parametry problemu
    Vmax = max_speed
    Vmin = min_speed
    R = renting_ratio
    W = knapsack_capacity

    print("Uruchamianie algorytmu brute force (wielordzeniowo)...")
    start = time.perf_counter()
    best_route, best_items, best_fitness = solve_bruteforce(graph, items_by_city, Vmin, Vmax, W, R)
    elapsed = time.perf_counter() - start

    # Wyświetlenie wyników
    print_summary(best_route, best_items, best_fitness, graph, items_by_city)
    print(f"Czas wykonania: {elapsed:.2f} s")

if __name__ == "__main__":
    main()
