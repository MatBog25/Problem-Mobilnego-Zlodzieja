Wartości w grafie to kolejno:
"1:" - numer miasta na mapie
[(2,200)] - lista miast, z których mamy dostępną trasę z bierzącego. (2,200) - 2 to numer miasta, 200 to odległość z bierzącego miasta do docelowego w kilometrach.

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
