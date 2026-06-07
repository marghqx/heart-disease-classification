**Autorzy:** Weronika Kłujszo, Marta Czarnecka
**Temat:** Wykrywanie choroby serca 
**Zbiór danych:** UCI Heart Disease Dataset (Cleveland)  

---

## 1. Charakterystyka zbioru danych i analiza eksploracyjna

Do realizacji projektu wykorzystano  medyczny zbiór danych **UCI Heart Disease Dataset**, pochodzący z kliniki w Cleveland. Zgodnie z wytycznymi projektu, analiza oraz proces wnioskowania zostały zawężone do sześciu kluczowych cech fizjologicznych pacjentów:
1. **Wiek (age):** Wiek pacjenta wyrażony w latach.
2. **Spoczynkowe ciśnienie krwi (trestbps):** Ciśnienie krwi zmierzone w mm Hg podczas przyjęcia pacjenta do szpitala.
3. **Cholesterol (chol):** Poziom cholesterolu w surowicy krwi pacjenta wyrażony w mg/dl.
4. **Liczba głównych naczyń (ca):** Liczba naczyń krwionośnych (0–3) zabarwionych podczas badania fluoroskopowego, wskazująca na stopień zablokowania tętnic.
5. **Typ bólu w klatce piersiowej (cp):** Rodzaj odczuwanego bólu sklasyfikowany w skali 1–4, od dusznicy typowej po brak objawów.
6. **Maksymalne tętno (thalach):** Najwyższa akcja serca osiągnięta przez pacjenta podczas próby wysiłkowej.

Zmienna objaśniana (**target**) w oryginalnym zbiorze przyjmowała wartości od 0 (brak zmian chorobowych) do 4 (zaawansowane stadium choroby). Na potrzeby klasyfikacji binarnej dokonano transformacji zmiennej celowej: wartość `0` oznacza pacjenta zdrowego, natomiast wartości `1, 2, 3, 4` zostały zmapowane jako `1` (pacjent chory).

### 1.1. Przykładowe dane
Poniższa tabela przedstawia pierwsze pięć rekordów ze zbioru danych po ograniczeniu do wybranych cech (przed procesem skalowania):

| Wiek (age) | Ciśnienie (trestbps) | Cholesterol (chol) | Naczynia (ca) | Ból (cp) | Tętno (thalach) | Wynik (target) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 63.0 | 145.0 | 233.0 | 0.0 | 1.0 | 150.0 | 0 |
| 67.0 | 160.0 | 286.0 | 3.0 | 4.0 | 108.0 | 1 |
| 67.0 | 120.0 | 229.0 | 2.0 | 4.0 | 129.0 | 1 |
| 37.0 | 130.0 | 250.0 | 0.0 | 3.0 | 187.0 | 0 |
| 41.0 | 130.0 | 204.0 | 0.0 | 2.0 | 172.0 | 0 |

### 1.2. Rozkład cech i wnioski z analizy (EDA)
W ramach przygotowania danych uruchomiono dedykowany moduł wizualizacji (`plots_generator.py`), który wygenerował wykresy rozkładów oraz macierz korelacji (zapisane w katalogu `/plots`). Na podstawie analizy rozkładów sformułowano następujące wnioski:
* **Wiek (age):** Większość pacjentów w badanej próbie znajduje się w przedziale od 50 do 65 lat. Wykres rozkładu (histogram z nałożoną krzywą gęstości) wskazuje, że wraz z wiekiem rośnie proporcja osób chorych (target=1) w stosunku do zdrowych.
* **Ciśnienie krwi (trestbps):** Rozkład przypomina rozkład normalny o lekkiej prawostronnej skośności. Największa koncentracja pomiarów przypada na przedział 120–140 mm Hg. Pacjenci z ciśnieniem powyżej 140 mm Hg częściej są kwalifikowani jako osoby z grupy ryzyka.
* **Cholesterol (chol):** Wykazuje relatywnie szeroki rozrzut wartości (od ~125 do ponad 400 mg/dl), z wyraźnym punktem ciężkości w okolicach 240 mg/dl. Obserwuje się silne nakładanie się rozkładów dla osób chorych i zdrowych
* **Liczba głównych naczyń (ca):** Wykazuje potężną moc dyskryminacyjną w procesie klasyfikacji. Analiza danych wskazuje, że pacjenci z wyższą liczbą zablokowanych naczyń krwionośnych (`ca` > 0) drastycznie częściej okazują się chorzy. Jest to statystycznie jeden z najsilniejszych predyktorów w całym zbiorze.
* **Typ bólu w klatce piersiowej (cp):** Rozkład tej cechy ujawnia, że pacjenci zgłaszający tzw. ból bezobjawowy lub ukryty (wartość 4.0) stanowią największą grupę wśród osób ze zdiagnozowaną chorobą serca. Pozwala to modelom łatwiej odróżnić poważne stany wieńcowe od bólów o innym podłożu.
* **Maksymalne tętno (thalach):** Wykres rozkładu pokazuje wyraźną zależność medyczną, osoby zdrowe osiągają znacznie wyższe tętno maksymalne podczas próby wysiłkowej. Niskie wartości `thalach` silnie korelują z obecnością schorzeń kardiologicznych.

---

## 2. Przetwarzanie danych wejściowych przez algorytmy ML

Wybrane do projektu algorytmy różnią się sposobem przetwarzania danych i podejmowania decyzji klasyfikacyjnych. Poniżej opisano mechanizm, w jaki sposób każdy z nich przetwarza rozszerzony wektor wejściowy $X = [age, trestbps, chol, ca, cp, thalach]$.

### 2.1. K-NeighborsClassifier (KNN)
Algorytm KNN reprezentuje podejście leniwe (instance-based learning). Nie buduje on jawnego modelu matematycznego w fazie treningu. 
* **Przetwarzanie danych:** Każdy pacjent jest traktowany jako punkt w sześciowymiarowej przestrzeni kartezjańskiej. Podczas predykcji algorytm oblicza odległość (standardowo euklidesową) między nowym pacjentem a wszystkimi punktami ze zbioru treningowego. 
* **Decyzja:** Identyfikuje $K$ najbliższych punktów (sąsiadów) i przypisuje nowej obserwacji klasę dominującą wśród tych sąsiadów. Ponieważ klasyfikacja opiera się na odległościach między punktami, konieczne było zastosowanie standaryzacji cech.

### 2.2. Support Vector Classification (SVC)
Maszyna Wektorów Nośnych dąży do znalezienia optymalnej hiperpłaszczyzny rozdzielającej dwie klasy w przestrzeni cech.
* **Przetwarzanie danych:** SVC analizuje wektory wejściowe w celu maksymalizacji tzw. marginesu geometrycznego, czyli odległości między hiperpłaszczyzną decyzyjną a najbliższymi punktami treningowymi z obu klas (wektorami nośnymi). 
* **Nieliniowość:** W wersji bazowej algorytm domyślnie mapuje dane do wyższego wymiaru za pomocą funkcji Radial Basis Function (RBF), co pozwala na elastyczne wytyczanie zakrzywionych granic decyzji w wielowymiarowej przestrzeni cech pacjenta.

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

Przed przystąpieniem do zaawansowanej optymalizacji parametrów, uruchomiono wszystkie cztery algorytmy na domyślnych ustawieniach pakietu `scikit-learn` w celu wyznaczenia linii bazowej (baseline).

Po podziale zbioru w stosunku 80% (trening) do 20% (test) oraz zastosowaniu `StandardScaler`, modele osiągnęły następującą dokładność (`Accuracy`) na zbiorze testowym:

| Model klasyfikacyjny | Konfiguracja bazowa (Domyślna) | Dokładność (Baseline Accuracy) |
| :--- | :--- | :--- |
| **RandomForestClassifier** | `n_estimators=100`, Gini criterion | **90.16%** |
| **Support Vector Classifier (SVC)** | `kernel='rbf'`, `C=1.0` | **88.52%** |
| **KNeighborsClassifier** | `n_neighbors=5`, metric='minkowski' | **90.16%** |
| **MLPClassifier (Sklearn)** | 1 layer (100 hidden nodes), `activation='relu'` | **85.25%** |

### Wnioski z wyników bazowych:

Najwyższą skuteczność w konfiguracji bazowej uzyskały algorytmy Random Forest oraz KNN, osiągając identyczną wartość accuracy na poziomie 90,16%. Wynik ten wskazuje, że zastosowany zestaw cech, obejmujący zarówno parametry fizjologiczne, jak i istotne wskaźniki kliniczne (m.in. liczbę zablokowanych naczyń oraz typ bólu w klatce piersiowej), dostarcza wystarczających informacji do skutecznego rozróżniania klas.

Wszystkie analizowane modele osiągnęły dokładność w przedziale od 85% do 90%, co potwierdza wysoką wartość diagnostyczną wykorzystanych zmiennych. Dobre wyniki uzyskane przy domyślnych ustawieniach parametrów sugerują istnienie wyraźnych zależności pomiędzy cechami opisującymi pacjentów a zmienną decyzyjną.
