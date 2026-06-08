# Projekt: Wykrywanie choroby serca

**Autorzy:** Weronika Kłujszo, Marta Czarnecka  
**Zbiór danych:** UCI Heart Disease Dataset (Cleveland)  

---

## Architektura projektu

### Folder `code`
* **data_preprocessing.py** – plik do przetwarzania danych dla 3 cech (wiek, cholesterol, ciśnienie) ze zbioru UCI Heart Disease Dataset
* **plots_generator.py** – plik eda, generujący i zapisujący wykresy dla danych
* **models.py** – plik zawierający 4 podstawowe modele (bez optymalizacji)

#### Główne pliki projektu:
* **combined_regular.ipynb** – plik zawierający kod z `data_preprocessing`, `plots_generator`, `models`, a także dodatkowy kod optymalizacyjny dla 3 cech zbioru
* **combined_expanded.ipynb** – plik zawierający kod z `data_preprocessing`, `plots_generator`, `models`, a także dodatkowy kod optymalizacyjny dla wszystkich 13 cech zbioru

---

### Folder `data`
* **test_scaled.csv** – dane testowe po zastosowaniu StandardScaler (kod dostępny w `data_preprocessing.py`)
* **train_scaled.csv** – dane treningowe po zastosowaniu StandardScaler (kod dostępny w `data_preprocessing.py`)

---

### Folder `plots`
* **correlation_matrix.png** – macierz korelacji dla 3 cech zbioru wraz z y=target
* **features_distribtion.png** – wykresy dystrybucji cech dla danych testowych i treningowych dla 3 cech zbioru

---

### Pozostałe pliki
* **.gitignore**
* **documentation.md** – dokładna dokumentacja projektu, opracowująca kod z `combined_regular.ipynb` oraz `combined_expanded.ipynb`
* **requirements.txt**
