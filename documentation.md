**Autorzy:** Weronika Kłujszo, Marta Czarnecka
**Temat:** Wykrywanie choroby serca 
**Zbiór danych:** UCI Heart Disease Dataset (Cleveland)  

---

## 1. Charakterystyka zbioru danych i analiza eksploracyjna

Do realizacji projektu wykorzystano  medyczny zbiór danych **UCI Heart Disease Dataset**, pochodzący z kliniki w Cleveland. Zgodnie z wytycznymi projektu, analiza oraz proces wnioskowania zostały zawężone do trzech kluczowych, ciągłych cech fizjologicznych pacjentów:
1. **Wiek (age):** Wiek pacjenta wyrażony w latach.
2. **Spoczynkowe ciśnienie krwi (trestbps):** Ciśnienie krwi zmierzone w mm Hg podczas przyjęcia pacjenta do szpitala.
3. **Cholesterol (chol):** Poziom cholesterolu w surowicy krwi pacjenta wyrażony w mg/dl.

Zmienna objaśniana (**target**) w oryginalnym zbiorze przyjmowała wartości od 0 (brak zmian chorobowych) do 4 (zaawansowane stadium choroby). Na potrzeby klasyfikacji binarnej dokonano transformacji zmiennej celowej: wartość `0` oznacza pacjenta zdrowego, natomiast wartości `1, 2, 3, 4` zostały zmapowane jako `1` (pacjent chory).

### 1.1. Przykładowe dane
Poniższa tabela przedstawia pierwsze pięć rekordów ze zbioru danych po ograniczeniu do wybranych cech (przed procesem skalowania):

| Wiek (age) | Ciśnienie krwi (trestbps) | Cholesterol (chol) | Wynik (target) |
| :---: | :---: | :---: | :---: |
| 63.0 | 145.0 | 233.0 | 0 |
| 67.0 | 160.0 | 286.0 | 1 |
| 67.0 | 120.0 | 229.0 | 1 |
| 37.0 | 130.0 | 250.0 | 0 |
| 41.0 | 130.0 | 204.0 | 0 |

### 1.2. Rozkład cech i wnioski z analizy (EDA)
W ramach przygotowania danych uruchomiono dedykowany moduł wizualizacji (`plots_generator.py`), który wygenerował wykresy rozkładów oraz macierz korelacji (zapisane w katalogu `/plots`). Na podstawie analizy rozkładów sformułowano następujące wnioski:
* **Wiek (age):** Większość pacjentów w badanej próbie znajduje się w przedziale od 50 do 65 lat. Wykres rozkładu (histogram z nałożoną krzywą gęstości) wskazuje, że wraz z wiekiem rośnie proporcja osób chorych (target=1) w stosunku do zdrowych.
* **Ciśnienie krwi (trestbps):** Rozkład przypomina rozkład normalny o lekkiej prawostronnej skośności. Największa koncentracja pomiarów przypada na przedział 120–140 mm Hg. Pacjenci z ciśnieniem powyżej 140 mm Hg częściej są kwalifikowani jako osoby z grupy ryzyka.
* **Cholesterol (chol):** Wykazuje relatywnie szeroki rozrzut wartości (od ~125 do ponad 400 mg/dl), z wyraźnym punktem ciężkości w okolicach 240 mg/dl. Obserwuje się silne nakładanie się rozkładów dla osób chorych i zdrowych

---

## 2. Przetwarzanie danych wejściowych przez algorytmy ML

Wybrane do projektu algorytmy różnią się sposobem przetwarzania danych i podejmowania decyzji klasyfikacyjnych. Poniżej opisano mechanizm, w jaki sposób każdy z nich przetwarza wektor wejściowy $X = [age, trestbps, chol]$.

### 2.1. K-NeighborsClassifier (KNN)
Algorytm KNN reprezentuje podejście leniwe (instance-based learning). Nie buduje on jawnego modelu matematycznego w fazie treningu. 
* **Przetwarzanie danych:** Każdy pacjent jest traktowany jako punkt w trójwymiarowej przestrzeni kartezjańskiej. Podczas predykcji algorytm oblicza odległość (standardowo euklidesową) między nowym pacjentem a wszystkimi punktami ze zbioru treningowego. 
* **Decyzja:** Identyfikuje $K$ najbliższych punktów (sąsiadów) i przypisuje nowej obserwacji klasę dominującą wśród tych sąsiadów. Ponieważ klasyfikacja opiera się na odległościach między punktami, konieczne było zastosowanie standaryzacji cech.

### 2.2. Support Vector Classification (SVC)
Maszyna Wektorów Nośnych dąży do znalezienia optymalnej hiperpłaszczyzny rozdzielającej dwie klasy w przestrzeni cech.
* **Przetwarzanie danych:** SVC analizuje wektory wejściowe w celu maksymalizacji tzw. marginesu geometrycznego, czyli odległości między hiperpłaszczyzną decyzyjną a najbliższymi punktami treningowymi z obu klas (wektorami nośnymi). 
* **Nieliniowość:** W wersji bazowej algorytm domyślnie mapuje dane do wyższego wymiaru za pomocą funkcji Radial Basis Function (RBF), co pozwala na elastyczne wytyczanie zakrzywionych granic decyzji w 3-wymiarowej przestrzeni wieku, ciśnienia i cholesterolu.

### 2.3. RandomForestClassifier
Las Losowy to algorytm zespołowy (ensemble learning) oparty na technice agregacji (bagging).
* **Przetwarzanie danych:** Algorytm tworzy zbiór niezależnych drzew decyzyjnych. Każde drzewo przetwarza dane wejściowe poprzez sekwencyjne podziały (splity) przestrzeni cech przy użyciu prostych nierówności (np. *czy wiek > 55 i cholesterol < 240?*). Podziały są dobierane tak, aby zmaksymalizować czystość powstałych węzłów (mierzoną współczynnikiem Giniego lub entropią).
* **Decyzja:** Każde drzewo w lesie oddaje swój niezależny głos na klasę 0 lub 1. Ostateczna prognoza jest wynikiem głosowania większościowego. Las losowy z natury jest odporny na brak skalowania danych, choć w projekcie przeskalowano dane jednolicie dla wszystkich modeli.

### 2.4. MLPClassifier
Wielowarstwowy Perceptron to sztuczna sieć neuronowa typu feed-forward.
* **Przetwarzanie danych:** Wektor wejściowy $X$ trafia do warstwy wejściowej, skąd sygnał płynie do warstw ukrytych. W każdym neuronie następuje operacja kombinacji liniowej: pomnożenie cech przez wagi ($w$), dodanie wyrazu wolnego ($b$), a następnie przepuszczenie wyniku przez nieliniową funkcję aktywacji (np. ReLU lub Tanh).
* **Uczenie:** Proces uczenia polega na propagacji w przód (Forward Propagation), obliczeniu błędu klasyfikacji na wyjściu za pomocą funkcji straty, a następnie propagacji wstecznej (Backpropagation). W tym drugim kroku wagi są modyfikowane za pomocą algorytmu spadku gradientu (np. optymalizatora Adam), aby zminimalizować błąd przy kolejnych podejściach.

---

## 3. Wstępne wyniki testowe

Przed przystąpieniem do zaawansowanej optymalizacji parametrów (zadanie dla Części II raportu), uruchomiono wszystkie cztery algorytmy na domyślnych ustawieniach pakietu `scikit-learn` w celu wyznaczenia linii bazowej (baseline).

Po podziale zbioru w stosunku 80% (trening) do 20% (test) oraz zastosowaniu `StandardScaler`, modele osiągnęły następującą dokładność (`Accuracy`) na zbiorze testowym:

| Model klasyfikacyjny | Konfiguracja bazowa (Domyślna) | Dokładność (Baseline Accuracy) |
| :--- | :--- | :--- |
| **RandomForestClassifier** | `n_estimators=100`, Gini criterion | **59.02%** |
| **Support Vector Classifier (SVC)** | `kernel='rbf'`, `C=1.0` | **62.30%** |
| **KNeighborsClassifier** | `n_neighbors=5`, metric='minkowski' | **59.02%** |
| **MLPClassifier (Sklearn)** | 1 layer (100 hidden nodes), `activation='relu'` | **57.38%** |

### Wnioski z wyników bazowych:
1. Najwyższą skutecznością na start wykazał się algorytm **SVC (62.30%)**, co sugeruje, że nieliniowe jądro RBF pozwala skuteczniej oddzielać klasy w przypadku danych, które nie są liniowo separowalne.
2. Skuteczność modeli na poziomie ok. 60% wynika bezpośrednio z ograniczenia liczby cech wejściowych do zaledwie trzech. W oryginalnym zbiorze UCI Cleveland obecne są znacznie silniejsze predyktory (np. ból w klatce piersiowej czy naczynia krwionośne). Uzyskanie wyniku przekraczającego poziom losowy (50%) z samego wieku, ciśnienia i cholesterolu potwierdza, że wybrane cechy zawierają informacje przydatne do klasyfikacji stanu pacjenta.
3. Wyniki te stanowią punkt odniesienia dla dalszej optymalizacji modeli.