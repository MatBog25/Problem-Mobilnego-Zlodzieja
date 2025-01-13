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
file_path = "data/280_1.txt"  # Plik z danymi o miastach
route =  [70, 71, 72, 73, 74, 75, 76, 79, 80, 89, 109, 86, 87, 84, 85, 65, 64, 66, 67, 69, 68, 58, 57, 56, 55, 46, 47, 48, 49, 50, 51, 52, 53, 54, 39, 36, 35, 38, 37, 41, 42, 43, 60, 61, 59, 44, 45, 40, 34, 33, 32, 31, 30, 125, 124, 123, 122, 121, 120, 119, 157, 158, 159, 160, 175, 161, 162, 164, 166, 167, 168, 169, 102, 103, 104, 108, 110, 111, 114, 115, 117, 118, 62, 63, 112, 88, 83, 82, 81, 92, 99, 100, 101, 170, 172, 171, 173, 106, 107, 105, 90, 91, 96, 93, 98, 97, 95, 94, 78, 77, 113, 116, 153, 156, 152, 151, 177, 176, 181, 182, 183, 184, 185, 187, 186, 188, 189, 190, 191, 193, 144, 145, 200, 198, 197, 195, 196, 201, 199, 203, 204, 212, 211, 210, 209, 252, 257, 254, 255, 256, 258, 259, 260, 261, 275, 274, 273, 272, 271, 16, 19, 132, 131, 130, 154, 155, 136, 267, 268, 137, 266, 265, 264, 263, 262, 270, 134, 135, 269, 18, 133, 17, 21, 20, 128, 127, 126, 28, 27, 26, 22, 25, 23, 24, 14, 15, 13, 12, 11, 10, 8, 9, 7, 276, 277, 278, 279, 3, 280, 2, 242, 241, 245, 240, 246, 239, 238, 231, 229, 228, 227, 226, 225, 224, 235, 234, 233, 232, 237, 236, 244, 243, 247, 250, 251, 230, 248, 249, 5, 6, 4, 1, 208, 253, 206, 207, 213, 216, 215, 218, 217, 220, 221, 222, 219, 214, 223, 205, 142, 143, 146, 147, 148, 149, 139, 138, 140, 141, 150, 178, 179, 180, 202, 194, 192, 163, 165, 174, 29, 129]
visualize_route(file_path, route, title="Najlepsza trasa")
