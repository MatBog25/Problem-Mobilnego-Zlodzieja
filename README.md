# Traveling Thief Problem - Heuristic Algorithms Comparison
## Porównanie Algorytmów Heurystycznych dla Problemu mobilnego złodzieja

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## English Version

### Overview

This project implements and compares five metaheuristic algorithms for solving the Traveling Thief Problem (TTP). TTP is a multi-component optimization problem that combines the Traveling Salesman Problem (TSP) with the Knapsack Problem (KP), creating a challenging interdependent optimization scenario.

### Problem Description

The Traveling Thief Problem consists of:
- **Cities**: A set of cities with geographical coordinates
- **Items**: Each city contains items with specific profit and weight values
- **Knapsack**: The thief has a knapsack with limited capacity
- **Speed constraint**: The traveling speed decreases as the knapsack weight increases

### Objective Function

The goal is to maximize the objective function:

```
Objective = Total_Profit - Travel_Cost
```

Where:
- **Total_Profit** = Sum of profits from collected items
- **Travel_Cost** = Renting_Ratio × Travel_Time
- **Travel_Time** = Σ(distance_i / speed_i)
- **Speed** = max(min_speed, max_speed - current_weight × v_w)
- **v_w** = (max_speed - min_speed) / knapsack_capacity

### Implemented Algorithms

#### 1. Genetic Algorithm (GA)
- **File**: `algorytm_genetyczny.py`
- **Features**: 
  - Population-based evolutionary approach
  - Crossover and mutation operations
  - Multiple initialization strategies (random, nearest neighbor, value-based)
  - Intelligent item selection based on profit-to-weight ratio

#### 2. Ant Colony Optimization (ACO)
- **File**: `algorytm_mrowkowy.py`
- **Features**:
  - Pheromone-based path construction
  - Distance matrix optimization for faster computation
  - Dynamic item selection considering remaining travel distance
  - Cached distance calculations using @lru_cache

#### 3. Particle Swarm Optimization (PSO)
- **File**: `roj_czastek.py`
- **Features**:
  - Swarm intelligence approach
  - Velocity and position updates
  - Global and personal best tracking
  - Smart item selection algorithm

#### 4. Simulated Annealing (SA)
- **File**: `simulated_annealing.py`
- **Features**:
  - Temperature-based acceptance probability
  - 2-opt local search improvements
  - Caching mechanisms for performance optimization
  - Adaptive neighborhood exploration

#### 5. Random Search (RS)
- **File**: `random_search.py`
- **Features**:
  - Baseline comparison algorithm
  - Pure random route and item selection
  - Statistical sampling approach

### Project Structure

```

├── algorytm_genetyczny.py       # Genetic Algorithm implementation
├── algorytm_mrowkowy.py         # Ant Colony Optimization implementation
├── roj_czastek.py               # Particle Swarm Optimization implementation
├── simulated_annealing.py       # Simulated Annealing implementation
├── random_search.py             # Random Search implementation
├── common/                      # Shared utilities
│   ├── data_loader.py          # Data loading and parsing
│   ├── wizualizacja.py         # Visualization utilities
│   └── wykresy.py              # Plotting utilities
├── data/                       # Test instances
│   ├── 50_1.txt               # 50 cities instance
│   ├── 280_1.txt              # 280 cities instance
│   ├── 500_1.txt              # 500 cities instance
│   └── ...                    # Additional test instances
├── tests/                     # Testing modules
└── README.md                  # This file
```

### Data Format

Input files contain:
- **CAPACITY OF KNAPSACK**: Maximum weight capacity
- **MIN SPEED / MAX SPEED**: Speed constraints
- **RENTING RATIO**: Cost per time unit
- **NODE_COORD_SECTION**: City coordinates (x, y)
- **ITEMS SECTION**: Item data (id, profit, weight, city)

### Usage

```python
# Example: Running Ant Colony Optimization
from algorytm_mrowkowy import ACO

# Initialize ACO with parameters
aco = ACO(num_ants=100, num_iterations=300, alpha=1.0, beta=1.0, 
          evaporation_rate=0.9, pheromone_deposit=1000)

# Run optimization
best_route, best_distance, best_items, best_profit, \
best_weight, best_time, best_objective, travel_cost = aco.run()
```

### Performance Metrics

The algorithms are evaluated based on:
- **Objective Value**: Primary optimization criterion
- **Computation Time**: Algorithm efficiency
- **Solution Quality**: Consistency across multiple runs
- **Memory Usage**: Resource consumption

### Requirements

- Python 3.7+
- Standard libraries: `random`, `math`, `time`, `functools`
- Optional: `numpy`, `matplotlib` for visualization

---

## Wersja Polska

### Przegląd

Ten projekt implementuje i porównuje pięć algorytmów heurystycznych do rozwiązania problemu TTP. TTP to wielokomponentowy problem optymalizacyjny łączący Problem Komiwojażera (TSP) z Problemem Plecakowym (KP), tworząc wyzwanie optymalizacyjne o wzajemnie zależnych komponentach.

### Opis Problemu

TTP składa się z:
- **Miast**: Zbiór miast z współrzędnymi
- **Przedmiotów**: Każde miasto zawiera przedmioty o określonych wartościach zysku i wagi
- **Plecaka**: Złodziej posiada plecak o ograniczonej pojemności
- **Ograniczenia prędkości**: Prędkość podróży maleje wraz ze wzrostem wagi plecaka

### Funkcja Celu

Celem jest maksymalizacja funkcji celu:

```
Cel = Całkowity_Zysk - Koszt_Podróży
```

Gdzie:
- **Całkowity_Zysk** = Suma zysków z zebranych przedmiotów
- **Koszt_Podróży** = Współczynnik_Wynajmu × Czas_Podróży
- **Czas_Podróży** = Σ(odległość_i / prędkość_i)
- **Prędkość** = max(min_prędkość, max_prędkość - aktualna_waga × v_w)
- **v_w** = (max_prędkość - min_prędkość) / pojemność_plecaka

### Zaimplementowane Algorytmy

#### 1. Algorytm Genetyczny (GA)
- **Plik**: `algorytm_genetyczny.py`
- **Cechy**:
  - Populacyjne podejście ewolucyjne
  - Operacje krzyżowania i mutacji
  - Wielokrotne strategie inicjalizacji (losowa, najbliższy sąsiad, oparta na wartości)
  - Inteligentny wybór przedmiotów na podstawie stosunku zysku do wagi

#### 2. Algorytm Mrówkowy (ACO)
- **Plik**: `algorytm_mrowkowy.py`
- **Cechy**:
  - Konstrukcja ścieżek oparta na feromonach
  - Optymalizacja macierzy odległości dla szybszych obliczeń
  - Dynamiczny wybór przedmiotów uwzględniający pozostałą odległość
  - Cache'owane obliczenia odległości używając @lru_cache

#### 3. Algorytm Roju Cząstek (PSO)
- **Plik**: `roj_czastek.py`
- **Cechy**:
  - Podejście oparte na inteligencji roju
  - Aktualizacje prędkości i pozycji
  - Śledzenie globalnych i osobistych najlepszych rozwiązań
  - Inteligentny algorytm wyboru przedmiotów

#### 4. Symulowane Wyżarzanie (SA)
- **Plik**: `simulated_annealing.py`
- **Cechy**:
  - Prawdopodobieństwo akceptacji oparte na temperaturze
  - Ulepszenia lokalnego przeszukiwania 2-opt
  - Mechanizmy cache'owania
  - Adaptacyjna eksploracja sąsiedztwa

#### 5. Przeszukiwanie Losowe (RS)
- **Plik**: `random_search.py`
- **Cechy**:
  - Algorytm bazowy do porównań
  - Czysto losowy wybór trasy i przedmiotów

### Struktura Projektu

```
├── algorytm_genetyczny.py       # Implementacja Algorytmu Genetycznego
├── algorytm_mrowkowy.py         # Implementacja Algorytmu Mrówkowego
├── roj_czastek.py               # Implementacja Algorytmu Roju Cząstek
├── simulated_annealing.py       # Implementacja Symulowanego Wyżarzania
├── random_search.py             # Implementacja Przeszukiwania Losowego
├── common/                      # Współdzielone narzędzia
│   ├── data_loader.py          # Ładowanie i parsowanie danych
│   ├── wizualizacja.py         # Narzędzia wizualizacji
│   └── wykresy.py              # Narzędzia wykresów
├── data/                       # Instancje testowe
│   ├── 50_1.txt               # Instancja 50 miast
│   ├── 280_1.txt              # Instancja 280 miast
│   ├── 500_1.txt              # Instancja 500 miast
│   └── ...                    # Dodatkowe instancje testowe
├── tests/                     # Moduły testowe
└── README.md                  # Ten plik
```

### Format Danych

Pliki wejściowe zawierają:
- **CAPACITY OF KNAPSACK**: Maksymalna pojemność wagowa
- **MIN SPEED / MAX SPEED**: Ograniczenia prędkości
- **RENTING RATIO**: Koszt za jednostkę czasu
- **NODE_COORD_SECTION**: Współrzędne miast (x, y)
- **ITEMS SECTION**: Dane przedmiotów (id, zysk, waga, miasto)

### Użycie

```python
# Przykład: Uruchomienie Algorytmu Mrówkowego
from algorytm_mrowkowy import ACO

# Inicjalizacja ACO z parametrami
aco = ACO(num_ants=100, num_iterations=300, alpha=1.0, beta=1.0, 
          evaporation_rate=0.9, pheromone_deposit=1000)

# Uruchomienie optymalizacji
best_route, best_distance, best_items, best_profit, \
best_weight, best_time, best_objective, travel_cost = aco.run()
```

### Metryki Wydajności

Algorytmy są oceniane na podstawie:
- **Wartości Funkcji Celu**: Główne kryterium optymalizacji
- **Czasu Obliczeń**: Efektywność algorytmu
- **Jakości Rozwiązania**: Spójność w wielu uruchomieniach
- **Zużycia Pamięci**: Konsumpcja zasobów

### Wymagania

- Python 3.7+
- Biblioteki standardowe: `random`, `math`, `time`, `functools`
- Opcjonalne: `numpy`, `matplotlib` do wizualizacji

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{ttp_metaheuristics_2024,
  title={Traveling Thief Problem: Heuristic Algorithms Comparison},
  author={[Mateusz Boguszewski]},
  year={2025},
  url={https://github.com/MatBog25/Problem-Mobilnego-Zlodzieja}
}
```