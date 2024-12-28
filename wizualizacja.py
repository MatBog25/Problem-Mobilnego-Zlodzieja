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
file_path = "280_1.txt"  # Plik z danymi o miastach
route = [197, 196, 201, 198, 199, 145, 146, 143, 144, 200, 202, 203, 204, 205, 206, 207, 210, 209, 252, 255, 254, 253, 208, 212, 211, 214, 215, 218, 217, 216, 213, 221, 222, 219, 220, 223, 224, 225, 226, 227, 228, 229, 230, 251, 250, 247, 244, 241, 242, 243, 248, 278, 279, 4, 277, 276, 275, 274, 273, 272, 271, 16, 17, 18, 133, 132, 131, 130, 21, 20, 19, 134, 135, 269, 268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 256, 249, 245, 246, 231, 232, 237, 236, 235, 234, 233, 238, 239, 240, 280, 3, 1, 2, 10, 11, 12, 15, 13, 14, 24, 23, 25, 22, 26, 27, 28, 29, 32, 31, 30, 125, 124, 123, 122, 121, 120, 119, 157, 158, 159, 160, 175, 161, 162, 163, 164, 165, 188, 189, 190, 191, 192, 194, 195, 193, 186, 187, 185, 184, 183, 182, 181, 180, 179, 178, 176, 177, 151, 152, 156, 153, 155, 154, 129, 128, 127, 126, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 60, 61, 118, 62, 63, 64, 65, 66, 67, 70, 71, 72, 73, 74, 75, 77, 78, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 108, 110, 111, 114, 113, 87, 84, 85, 86, 116, 115, 117, 59, 44, 45, 46, 47, 48, 49, 52, 51, 50, 37, 53, 54, 55, 56, 57, 58, 68, 69, 83, 88, 112, 109, 89, 90, 91, 92, 93, 94, 79, 80, 81, 82, 76, 107, 106, 173, 174, 105, 169, 170, 171, 168, 167, 166, 172, 150, 149, 148, 147, 142, 141, 140, 139, 138, 137, 136, 270, 9, 8, 7, 6, 5]
visualize_route(file_path, route, title="Najlepsza trasa")
