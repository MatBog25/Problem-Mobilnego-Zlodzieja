import math

def load_data(file_path):
    graph = {}
    items_by_city = {}
    knapsack_capacity = None
    min_speed = None
    max_speed = None
    renting_ratio = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    node_coords = {}
    for line in lines:
        if line.startswith("CAPACITY OF KNAPSACK"):
            knapsack_capacity = int(line.split(":")[1].strip())
        elif line.startswith("MIN SPEED"):
            min_speed = float(line.split(":")[1].strip())
        elif line.startswith("MAX SPEED"):
            max_speed = float(line.split(":")[1].strip())
        elif line.startswith("RENTING RATIO"):
            renting_ratio = float(line.split(":")[1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            index = lines.index(line) + 1
            while not lines[index].startswith("ITEMS SECTION"):
                data = lines[index].strip().split()
                city_id = int(data[0])
                x, y = map(float, data[1:])
                node_coords[city_id] = (x, y)
                index += 1
        elif line.startswith("ITEMS SECTION"):
            index = lines.index(line) + 1
            while index < len(lines):
                data = lines[index].strip().split()
                if len(data) < 4:
                    index += 1
                    continue
                item_id, profit, weight, city = map(int, data[:4])
                if city not in items_by_city:
                    items_by_city[city] = []
                items_by_city[city].append((item_id, profit, weight))
                index += 1

    # Budowanie grafu za pomocą odległości Euklidesowych
    for city1, (x1, y1) in node_coords.items():
        graph[city1] = []
    for city1, (x1, y1) in node_coords.items():
        for city2, (x2, y2) in node_coords.items():
            if city1 != city2:
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                graph[city1].append((city2, math.ceil(distance)))
                graph[city2].append((city1, math.ceil(distance)))  # Dodaj relację odwrotną

    return graph, items_by_city, knapsack_capacity, min_speed, max_speed, renting_ratio
