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
route = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 68, 69, 77, 78, 186, 192, 204, 205, 206, 243, 242, 31, 32, 29, 28, 27, 26, 22, 25, 23, 24, 14, 13, 12, 11, 10, 8, 7, 9, 6, 5, 260, 259, 258, 257, 254, 253, 208, 207, 210, 209, 252, 255, 256, 249, 248, 247, 244, 241, 240, 239, 238, 231, 
232, 233, 234, 235, 236, 237, 246, 245, 250, 251, 230, 229, 228, 227, 226, 225, 224, 223, 222, 219, 218, 215, 214, 211, 212, 213, 216, 217, 220, 221, 203, 
202, 200, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 1, 280, 2, 3, 279, 278, 4, 277, 276, 275, 274, 273, 272, 271, 16, 17, 18, 19, 20, 21, 128, 127, 126, 125, 30, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261, 15, 133, 132, 131, 130, 129, 154, 155, 153, 156, 152, 151, 177, 176, 181, 180, 179, 178, 150, 149, 148, 147, 146, 145, 199, 198, 197, 194, 195, 196, 201, 193, 191, 190, 189, 188, 187, 185, 184, 183, 182, 161, 162, 163, 164, 165, 166, 167, 168, 169, 101, 100, 99, 98, 93, 94, 95, 96, 97, 92, 91, 90, 89, 109, 108, 104, 103, 102, 170, 171, 172, 173, 106, 105, 107, 174, 175, 160, 159, 158, 157, 119, 120, 121, 122, 123, 124, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 60, 61, 118, 117, 115, 114, 111, 110, 112, 88, 83, 82, 81, 80, 79, 76, 75, 74, 73, 
72, 71, 70, 67, 66, 65, 64, 63, 62, 116, 86, 85, 84, 87, 113, 59, 44]
visualize_route(file_path, route, title="Najlepsza trasa")