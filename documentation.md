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

## 3. Wstępne wyniki testowe - bez optymalizacji, ze skalowaniem, na 3 CECHACH 

Przed przystąpieniem do zaawansowanej optymalizacji parametrów, uruchomiono wszystkie cztery algorytmy na domyślnych ustawieniach pakietu `scikit-learn` w celu wyznaczenia linii bazowej (baseline).

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

---

## 4. Optymalizacja architektury MLP (TensorFlow/Keras) z użyciem GPU

### 4.1. Model bazowy TensorFlow

Model bazowy zbudowano jako sieć sekwencyjną z dwiema warstwami ukrytymi (64 i 32 neurony z aktywacją ReLU) i warstwą wyjściową z sigmoidem. Sieć była uczona przez 50 epok z rozmiarem batcha 32 i optymalizatorem Adam, przy funkcji straty `binary_crossentropy`. Obliczenia były wykonywane na procesorze GPU T4 w środowisku Google Colab.

Wyniki dla obu wariantów zbioru danych:

| Wariant zbioru | Test Accuracy |
| :--- | :---: |
| 3 cechy (age, trestbps, chol) | ~65.57% |
| Wszystkie 13 cech | ~83.61% |

Różnica (~18 punktów procentowych) jednoznacznie potwierdza, że trzy wybrane cechy nie są wystarczające do skutecznej klasyfikacji, ponieważ większość informacji diagnostycznej tkwi w pozostałych atrybutach zbioru (typ bólu w klatce piersiowej, wynik EKG, tętno maksymalne itp.). Jak się później okaże, jest to przede wszystkim tętno.

Warto zaznaczyć, że nawet przy 20 000 kroków uczenia (epochs × batches), czas treningu TensorFlow na GPU jest zdecydowanie krótszy niż czas wykonania GridSearchCV — ten ostatni, ze względu na konieczność wielokrotnego dopasowywania modeli (przeszukiwanie siatki parametrów × walidacja krzyżowa), jest znacznie bardziej kosztowny obliczeniowo i potrafi działać przez bardzo długi czas nawet na stosunkowo małym zbiorze danych.

### 4.2. Przeszukiwanie przestrzeni architektur

W celu optymalnego doboru architektury sieci przetestowano kombinację następujących hiperparametrów:

* **Architektury (liczba warstw i neuronów):** `[32, 32]`, `[64]`, `[128, 64]`, `[256, 128, 64]`
* **Funkcje aktywacji:** `relu`, `sigmoid`, `tanh`
* **Optymalizatory:** `adam`, `sgd`

Łącznie przetestowano 24 kombinacje (4 architektury × 3 aktywacje × 2 optymalizatory), każda uczona przez 50 epok bez wyświetlania logów (`verbose=0`).

### 4.3. Najlepsze konfiguracje MLP

| Wariant zbioru | Najlepsza architektura | Aktywacja | Optymalizator | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| 3 cechy | `[32, 32]` | `relu` | `adam` | ~67.21% |
| Wszystkie 13 cech | `[32, 32]` | `relu` | `sgd` | **90.16%** |

**Wnioski dot. funkcji aktywacji i architektury:**

* Funkcja **ReLU** (Rectified Linear Unit) okazała się najlepsza w obydwu wariantach zbioru. Jej popularność wynika z prostoty obliczeniowej i braku problemu zanikającego gradientu, który mocno dotyka funkcję `sigmoid` w głębszych sieciach. `tanh` dawała wyniki zbliżone do ReLU, ale nieznacznie gorsze.
* Zaskakującym wynikiem jest to, że **płytsza architektura `[32, 32]`** osiągnęła najlepsze wyniki zamiast głębszej `[256, 128, 64]`. Przy stosunkowo małym zbiorze (~242 próbki treningowe) zbyt duże sieci ulegają przeuczeniu i zbyt mała liczba przykładów nie wystarcza, aby właściwie wytrenować setki parametrów.
* Optymalizator **SGD** okazał się lepszy dla pełnego zestawu cech, co może wynikać z jego silniejszego efektu regularyzacyjnego przy małych danych w porównaniu do adaptatywnego Adama.

---

## 5. Optymalizacja liczby drzew w RandomForest

W celu doboru optymalnej liczby drzew (`n_estimators`) przeprowadzono 5-krotną walidację krzyżową (cross-validation).

### Wyniki (wariant 3 cech):

Najwyższy wynik CV uzyskano dla `n_estimators = 8` lub `n_estimators = 9`, po czym krzywa stabilizuje się i nie rośnie istotnie wraz z dalszym zwiększaniem liczby drzew. Sugeruje to, że przy tak małej przestrzeni cech (3 predyktory) las szybko osiąga swoje granice możliwości i dodatkowe drzewa nie wnoszą nowych informacji.

### Wyniki (wariant wszystkich 13 cech):

Dla pełnego zestawu cech optymalna liczba drzew wynosiła **200** — krzywa CV rosła monotonicznie, co wskazuje, że bogatsza przestrzeń cech wymaga większej różnorodności drzew do efektywnego uśrednienia błędu.

**Ogólny wniosek:** Optymalną liczbę drzew należy dobierać empirycznie w zależności od złożoności problemu. Zbyt mała liczba drzew prowadzi do wysokiej wariancji predykcji, zaś zbyt duża do niepotrzebnych kosztów obliczeniowych bez poprawy jakości modelu.

---

## 6. Optymalizacja algorytmów metodą GridSearchCV

GridSearchCV przeszukuje zadaną siatkę kombinacji hiperparametrów i ocenia każdą z nich przy użyciu walidacji krzyżowej. 

Optymalizację przeprowadzono zarówno na danych nieskalowanych, jak i skalowanych (`StandardScaler`), dla każdego z czterech algorytmów.

### 6.1. Przeszukiwane siatki parametrów

**SVC:**
* `C`: [0.1, 1, 10] — parametr regularyzacji (im większy, tym mniejszy margines i mniejsze przeuczenie)
* `kernel`: ['rbf', 'linear']
* `gamma`: ['scale', 'auto']

**KNN:**
* `n_neighbors`: [3, 5, 7, 9, 11]
* `metric`: ['euclidean', 'manhattan']

**RandomForest:**
* `n_estimators`: [5, 10, 50, 100, 200] (wariant 3 cech: [5, 8, 9, 10, 20, 30, 50, 100, 200])
* `max_depth`: [None, 5, 10]
* `criterion`: ['gini', 'entropy']

**MLPClassifier (sklearn):**
* `hidden_layer_sizes`: [(50,), (100,), (100, 50)]
* `activation`: ['relu', 'tanh']
* `alpha`: [0.0001, 0.001] — siła regularyzacji L2

### 6.2. Wyniki GridSearchCV — wariant 3 cech

| Model | Najlepsze parametry | CV Accuracy | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **SVC** | `kernel='rbf'`, `C=10`, `gamma='scale'` | ~61% | ~62% |
| **SVC (scaled)** | `kernel='rbf'`, `C=10`, `gamma='scale'` | ~64% | ~67% |
| **KNN** | `n_neighbors=11`, `metric='manhattan'` | ~60% | ~60% |
| **KNN (scaled)** | `n_neighbors=11`, `metric='euclidean'` | ~63% | ~65% |
| **RandomForest** | `n_estimators=50`, `criterion='gini'`, `max_depth=None` | ~61% | ~62% |
| **RandomForest (scaled)** | jak wyżej | ~61% | ~62% |
| **MLP sklearn** | `hidden_layer_sizes=(100, 50)`, `activation='tanh'`, `alpha=0.001` | ~61% | ~62% |
| **MLP sklearn (scaled)** | `activation='relu'`, `alpha=0.0001` | ~62% | ~64% |

### 6.3. Wyniki GridSearchCV — wariant wszystkich 13 cech

| Model | Najlepsze parametry | CV Accuracy | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **SVC** | `kernel='rbf'`, `C=10`, `gamma='scale'` | ~82% | ~85% |
| **SVC (scaled)** | `kernel='rbf'`, `C=1`, `gamma='scale'` | ~84% | ~87% |
| **KNN** | `n_neighbors=7`, `metric='manhattan'` | ~76% | ~79% |
| **KNN (scaled)** | `n_neighbors=7`, `metric='euclidean'` | ~82% | ~84% |
| **RandomForest** | `n_estimators=100`, `criterion='entropy'`, `max_depth=None` | ~83% | ~85% |
| **RandomForest (scaled)** | jak wyżej | ~83% | ~85% |
| **MLP sklearn** | `hidden_layer_sizes=(100, 50)`, `activation='relu'`, `alpha=0.0001` | ~82% | ~84% |
| **MLP sklearn (scaled)** | jak wyżej | ~83% | ~85% |

### 6.4. Wnioski z optymalizacji

**Które funkcje aktywacji dają najlepsze wyniki?**

Dla MLPClassifier (sklearn) najlepsze wyniki dawała funkcja **`relu`** na danych ze skalowaniem oraz **`tanh`** na danych bez skalowania. Obydwie funkcje są silnie nieliniowe i umożliwiają sieci naukę złożonych reprezentacji. `relu` jest preferowana przy danych znormalizowanych ze względu na szybszą zbieżność, natomiast `tanh` lepiej radzi sobie gdy dane mają wartości symetryczne wokół zera (co symuluje brak skalowania).

Dla TensorFlow najlepszą funkcją aktywacji była **`relu`** — zarówno w wariancie 3 cech, jak i 13 cech.

**Które jądra SVC dają najlepsze wyniki?**

W obu wariantach zbioru jądro **`rbf`** (Radial Basis Function) dominowało nad jądrem liniowym. Wynika to z faktu, że dane medyczne nie są liniowo separowalne, więc granica decyzyjna między pacjentami chorymi i zdrowymi ma charakter zakrzywiony. Jądro RBF efektywnie mapuje dane do przestrzeni wyższego wymiaru, w której możliwe jest znalezienie lepszej hiperpłaszczyzny rozdzielającej klasy.

**Wpływ skalowania:** Skalowanie danych poprawiało wyniki wszystkich modeli wrażliwych na skalę (KNN, SVC, MLP). RandomForest był na nie praktycznie obojętny, co jest zgodne z jego naturą — drzewa decyzyjne operują na progach cech, a nie na ich wartościach bezwzględnych.

---

## 7. Analiza ważności cech (Feature Importance) dla RandomForest

Analiza ważności cech przeprowadzona na modelu RandomForest po optymalizacji GridSearchCV wykorzystuje miarę **MDI (Mean Decrease in Impurity)**, czyli średni spadek nieczystości Giniego (lub entropii) wynikający z podziałów w danym węźle dla każdej cechy, uśredniony po wszystkich drzewach w lesie.

### 7.1. Wariant 3 cech (age, trestbps, chol)

W wariancie ograniczonym do trzech cech ważności rozkładają się stosunkowo równomiernie:

| Cecha | Ważność (MDI) |
| :--- | :---: |
| **age** (wiek) | ~0.39 |
| **chol** (cholesterol) | ~0.33 |
| **trestbps** (ciśnienie) | ~0.28 |

Wiek okazał się najistotniejszym predyktorem (akurat w tym zetawieniu), co jest zgodne z wiedzą medyczną - ryzyko chorób serca znacząco rośnie z wiekiem. Cholesterol i ciśnienie mają zbliżone znaczenie, a ich relatywnie wysokie ważności wynikają też po części z faktu, że są jedynymi dostępnymi predyktorami — model „zużywa" całą przestrzeń cech.

### 7.2. Wariant wszystkich 13 cech

W pełnym zbiorze cech pojawia się wyraźna hierarchia:

| Cecha | Opis | Ważność (MDI) |
| :--- | :--- | :---: |
| **thal** | Wynik badania talasemii | ~0.18 |
| **cp** | Typ bólu w klatce piersiowej | ~0.15 |
| **ca** | Liczba naczyń krwionośnych (fluoroskopia) | ~0.14 |
| **oldpeak** | Obniżenie ST (EKG wysiłkowe) | ~0.12 |
| **thalach** | Maksymalne tętno | ~0.10 |
| **age** | Wiek | ~0.08 |
| **chol** | Cholesterol | ~0.06 |
| **trestbps** | Ciśnienie spoczynkowe | ~0.05 |
| pozostałe | sex, fbs, restecg, exang, slope | ~0.12 łącznie |

**Kluczowe wnioski:**

1. Trzy cechy z zadania (age, trestbps, chol) łącznie odpowiadają za jedynie około **19% łącznej ważności**, podczas gdy pozostałe 10 cech — za **81%**. Tłumaczy to dramatyczną różnicę w dokładności między wariantami (~60% vs ~85%).

2. Najważniejszym predyktorem okazał się wynik badania **talasemii (thal)**. Kategoria choroby hemoglobiny jest silnie powiązana z przyszłymi incydentami kardiologicznymi. Tuż za nim plasuje się **typ bólu w klatce piersiowej (cp)** - jego charakter (stabilny/niestabilny/atypowy/bezobjawowy) jest jednym z najsilniejszych klinicznych sygnałów choroby wieńcowej.

3. Wiek, cholesterol i ciśnienie,mimo, że są medycznie istotnymi czynnikami ryzyka — okazują się słabymi predyktorami w kontekście tego konkretnego zbioru klinicznego, w którym znacznie bardziej informacyjne są wyniki badań diagnostycznych (EKG, angiografia, próba wysiłkowa).

---

## 8. Podsumowanie i porównanie modeli

| Model | Wariant 3 cech (po optymalizacji) | Wariant 13 cech (po optymalizacji) |
| :--- | :---: | :---: |
| **SVC (scaled, rbf)** | ~67% | **~87%** |
| **MLP TensorFlow (GPU)** | ~67% | ~90% |
| **RandomForest** | ~62% | ~85% |
| **KNN (scaled)** | ~65% | ~84% |
| **MLP sklearn (scaled)** | ~64% | ~85% |

Najwyższą skuteczność w wariancie wszystkich 13 cech osiągnęła sieć **MLP TensorFlow (90.16%)**, a spośród modeli sklearn — **SVC z jądrem RBF (~87%)**. Uzyskane wyniki potwierdzają, że:

* Ograniczenie do 3 cech z zadania skutkuje poważnym spadkiem dokładności (o ok. 20–25 pp.), co jest bezpośrednią konsekwencją niskiej informatywności tych cech.
* Optymalizacja hiperparametrów (GridSearchCV, przeszukiwanie architektur TF) przynosi mierzalną poprawę w stosunku do konfiguracji bazowych.
* Skalowanie danych ma kluczowe znaczenie dla modeli opartych na odległościach i sieciach neuronowych.
