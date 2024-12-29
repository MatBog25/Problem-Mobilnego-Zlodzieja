import matplotlib.pyplot as plt
import numpy as np

# Dane liczby miast i liczby kombinacji
cities = [5, 6, 8, 10, 11]
combinations = [120, 720, 40320, 3628800, 39916800]  # Liczba kombinacji (n!)

# Czasy dla każdej metody (w sekundach)
brute_force_times = [0.02, 0.18, 42.40, 2251.82 * 8, 2251.82 * 100]  # Oszacowanie dla 11 miast
multi_threaded_times = [0.14, 0.19, 5.13, 2251.82, 2251.82 * 10]  # Oszacowanie wielowątkowe
genetic_times = [0.35, 0.40, 0.50, 0.66, 0.75]  # Oszacowanie dla genetycznego

# Wykres liczby kombinacji
plt.figure(figsize=(10, 6))
plt.plot(cities, combinations, marker='o', label="Liczba kombinacji")
plt.yscale('log')
plt.ylim(100, 10**8)  # Dostosowanie zakresu osi Y
for i, txt in enumerate(combinations):
    plt.text(cities[i], combinations[i], f"{txt}", fontsize=8, ha='right')
plt.title("Eksplozja liczby kombinacji w funkcji liczby miast")
plt.xlabel("Liczba miast")
plt.ylabel("Liczba kombinacji (logarytmicznie)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Wykres czasów działania
plt.figure(figsize=(10, 6))
plt.plot(cities[:4], brute_force_times[:4], marker='o', label="Brute Force")
plt.plot(cities, multi_threaded_times, marker='o', label="Brute Force (Wielowątkowy)")
plt.plot(cities, genetic_times, marker='o', label="Algorytm Genetyczny")
plt.yscale('log')  # Skala logarytmiczna dla osi Y
plt.ylim(0.01, 10**4)  # Dostosowanie zakresu osi Y
for i, txt in enumerate(brute_force_times[:4]):
    plt.text(cities[i], brute_force_times[i], f"{txt:.2f}s", fontsize=8, ha='right')
for i, txt in enumerate(multi_threaded_times):
    plt.text(cities[i], multi_threaded_times[i], f"{txt:.2f}s", fontsize=8, ha='left')
for i, txt in enumerate(genetic_times):
    plt.text(cities[i], genetic_times[i], f"{txt:.2f}s", fontsize=8, ha='center')
plt.title("Porównanie czasów działania w funkcji liczby miast")
plt.xlabel("Liczba miast")
plt.ylabel("Czas wykonania (sekundy, logarytmicznie)")
plt.grid(True)
plt.legend()
plt.show()
