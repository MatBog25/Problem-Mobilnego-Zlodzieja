import matplotlib.pyplot as plt
import math

def load_data(file_path):
    """Wczytuje dane o miastach i współrzędnych z pliku."""
    node_coords = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_reading = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                start_reading = True
                continue
            if start_reading:
                if line.strip() == "" or line.startswith("ITEMS SECTION"):
                    break
                data = line.strip().split()
                city_id = int(data[0])
                x, y = map(float, data[1:])
                node_coords[city_id] = (x, y)
    return node_coords

def visualize_route(file_path, route, title="Wybrana trasa"):
    """Funkcja do wizualizacji trasy."""
    # Wczytaj dane o współrzędnych miast
    node_coords = load_data(file_path)

    x = []
    y = []

    # Pobierz współrzędne miast w kolejności trasy
    for city in route:
        if city in node_coords:
            x.append(node_coords[city][0])
            y.append(node_coords[city][1])

    # Dodaj powrót do miasta startowego
    x.append(node_coords[route[0]][0])
    y.append(node_coords[route[0]][1])

    # Rysuj punkty miast
    plt.scatter(x, y, color='red', zorder=5, label="Miasta")

    # Zaznacz miasto startowe
    plt.scatter(x[0], y[0], color='green', s=100, zorder=6, label="Miasto startowe")

    # Rysuj połączenia między miastami
    plt.plot(x, y, color='blue', linestyle='-', linewidth=1, zorder=1, label="Trasa")

    # Dodaj etykiety miast
    for city in route:
        plt.text(node_coords[city][0], node_coords[city][1], str(city), fontsize=8, ha='right', color='black')

    # Konfiguracja wykresu
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper left")  # Przeniesienie legendy w lewy górny róg
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis("equal")
    plt.show()

# Przykład użycia
file_path = "10miast.txt"  # Plik z danymi o miastach
route =  [7, 4, 1, 5, 2, 6, 9, 10, 8, 3]
visualize_route(file_path, route, title="Najlepsza trasa")
