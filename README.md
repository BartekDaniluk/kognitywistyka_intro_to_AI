# Interactive Kognitywistyka

Interaktywne materiały do przedmiotu **Wprowadzenie do sztucznej inteligencji** (kognitywistyka, UMCS).
Zamiast rysować przestrzenie stanów na kartce – zmieniasz parametry problemu i od razu widzisz graf,
definicję formalną, tabelę przejść i rozwiązanie, a potem sam „chodzisz” po przestrzeni stanów.

Notebooki działają w **Google Colab** bez instalowania czegokolwiek – kliknij przycisk i uruchom komórki po kolei.

| Lab | Temat | Colab |
|---|---|---|
| 2 | Przestrzenie stanów: świat odkurzacza, pojemniki z wodą | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BartekDaniluk/Interactive-Kognitywistyka/blob/main/lab02_stany/01_przestrzenie_stanow.ipynb) |

## Uruchomienie lokalnie

```bash
pip install -r requirements.txt jupyterlab
jupyter lab
```

## Struktura

- `labNN_*/` – notebook(i) dla danego laboratorium + moduł Pythona z kodem (np. `lab_states.py`).
  Pierwsza (ukryta) komórka każdego notebooka w Colabie klonuje to repozytorium i importuje kod stąd.
