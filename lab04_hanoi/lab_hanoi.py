"""
lab_hanoi.py - Wieze Hanoi jako przestrzen stanow, graf (trojkat Sierpinskiego)
oraz interaktywne cwiczenie porownujace rozwiazanie studenta z algorytmami BFS i DFS.

Modul do zajec "Wprowadzenie do AI" (kognitywistyka, UMCS) – laboratorium 4.
Korzysta z klas Problem i StateSpace z lab_states.py (Lab 2) oraz funkcji
przeszukiwania z lab_search.py (Lab 3).

Szybki start w notebooku:
    import lab_hanoi as lh
    lh.show_hanoi_space(n_disks=3, n_pegs=3)   # definicja formalna + graf przestrzeni stanow
    lh.show_hanoi_bfs_dfs(n_disks=3, n_pegs=3) # porownanie drzewa BFS i DFS na grafie Hanoi
    lh.hanoi_space_app()                       # interaktywny eksplorator przestrzeni stanow Hanoi
    lh.hanoi_exercise(n_disks=3, n_pegs=3)     # interaktywne zadanie dla studenta (HTML/SVG/JS)
"""
from __future__ import annotations

import html
import itertools
import json
import math
import os
import sys
import uuid

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# Upewnij sie, ze lab02_stany i lab03_przeszukiwanie sa w sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("lab02_stany", "lab03_przeszukiwanie"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import lab_states as ls
import lab_search as lsearch
from lab_states import Problem, StateSpace, _CSS, n_steps

# Kolory krazkow (od najmniejszego #1 do najwiekszego #6)
DISK_COLORS = (
    "#e03131",  # 1 - czerwony (najmniejszy)
    "#f08c00",  # 2 - pomaranczowy
    "#2f9e44",  # 3 - zielony
    "#1971c2",  # 4 - niebieski
    "#9c36b5",  # 5 - fioletowy
    "#0ca678",  # 6 - turkusowy (najwiekszy)
)

PEG_NAMES = "ABCDEF"


# --------------------------------------------------------------------------
# Formalny model Wiez Hanoi jako podklasa Problem (z Lab 2)
# --------------------------------------------------------------------------
class Hanoi(Problem):
    """Wieze Hanoi dla n_disks krazkow (1..n, 1 = najmniejszy) i n_pegs slupkow (0..k-1).

    Stan: krotka dlugosci n_disks: (p_1, p_2, ..., p_n), gdzie p_i in {0, ..., k-1}
    oznacza numer slupka, na ktorym lezy krazek i (poniewaz mniejszy musi lezec
    na wiekszym, przypisanie krazkow do slupkow jednoznacznie wyznacza ich kolejnosc!).
    """

    def __init__(self, n_disks: int = 3, n_pegs: int = 3, initial=None, goal=None):
        assert 2 <= n_disks <= 6, "n_disks musi byc w przedziale 2..6"
        assert 2 <= n_pegs <= 5, "n_pegs musi byc w przedziale 2..5"
        self.n_disks = n_disks
        self.n_pegs = n_pegs
        if initial is None:
            initial = (0,) * n_disks
        if goal is None:
            goal = (n_pegs - 1,) * n_disks
        self.goal_state = tuple(goal)
        super().__init__(tuple(initial))
        self.node_w = max(0.88, 0.30 * n_pegs + 0.16)
        self.node_h = 0.56

    @staticmethod
    def peg_name(p: int) -> str:
        return PEG_NAMES[p]

    def top_disks(self, s: tuple) -> list[int | None]:
        """Zwraca dla kazdego slupka indeks najmniejszego (wierzcholkowego) krazka (0..n-1) lub None."""
        top: list[int | None] = [None] * self.n_pegs
        for d in range(self.n_disks - 1, -1, -1):
            top[s[d]] = d
        return top

    def pegs_representation(self, s: tuple) -> str:
        """Zwraca czytelny zapis zawartosci slupkow, np. A:[3,2,1] B:[] C:[]."""
        pegs = [[] for _ in range(self.n_pegs)]
        for d in range(self.n_disks - 1, -1, -1):
            pegs[s[d]].append(str(d + 1))
        return " ".join(f"{self.peg_name(p)}:[{','.join(pegs[p])}]" for p in range(self.n_pegs))

    def all_states(self):
        return list(itertools.product(range(self.n_pegs), repeat=self.n_disks))

    def action_list(self):
        return [(u, v) for u in range(self.n_pegs) for v in range(self.n_pegs) if u != v]

    def applicable(self, s, a) -> bool:
        u, v = a
        top = self.top_disks(s)
        if top[u] is None:
            return False
        return top[v] is None or top[u] < top[v]

    def result(self, s, a):
        u, v = a
        top = self.top_disks(s)
        du = top[u]
        assert du is not None
        s2 = list(s)
        s2[du] = v
        return tuple(s2)

    def is_goal(self, s) -> bool:
        return s == self.goal_state

    def state_label(self, s) -> str:
        return "(" + ", ".join(self.peg_name(p) for p in s) + ")"

    def action_name(self, a) -> str:
        u, v = a
        return f"{self.peg_name(u)} → {self.peg_name(v)}"

    def action_short(self, a) -> str:
        u, v = a
        return f"{self.peg_name(u)}→{self.peg_name(v)}"

    def action_color(self, a) -> str:
        u, v = a
        palette = ("#1971c2", "#2f9e44", "#e8590c", "#9c36b5", "#0ca678")
        return palette[(u + v) % len(palette)]

    def layout(self, states):
        n, k = self.n_disks, self.n_pegs
        if k == 3:
            # Dokladny uklad trojkata Sierpinskiego!
            V = [(-1.0, 0.0), (0.0, math.sqrt(3.0)), (1.0, 0.0)]
            scale_base = 1.05 if n <= 3 else 0.82
            pos = {}
            for s in states:
                perm = [0, 1, 2]
                x, y = 0.0, 0.0
                for d in range(n - 1, -1, -1):
                    p = s[d]
                    c = perm[p]
                    sc = (2 ** d) * scale_base
                    x += sc * V[c][0]
                    y += sc * V[c][1]
                    others = [i for i in range(3) if i != p]
                    perm[others[0]], perm[others[1]] = perm[others[1]], perm[others[0]]
                pos[s] = (x, y)
            return pos
        elif k == 2:
            # Uklad siatkowy (wiersze = pary wiekszych krazkow, kolumny = najmniejszy krazek)
            pos = {}
            for idx, s in enumerate(sorted(states)):
                col = s[0]
                row = sum(s[d] * (2 ** (d - 1)) for d in range(1, n))
                pos[s] = (col * 3.2, -row * 1.35)
            return pos
        else:
            # Wielokat rekurencyjny dla k >= 4
            angles = [math.pi / 2 + 2 * math.pi * i / k for i in range(k)]
            V = [(math.cos(a), math.sin(a)) for a in angles]
            pos = {}
            for s in states:
                x, y = 0.0, 0.0
                for d in range(n - 1, -1, -1):
                    p = s[d]
                    sc = (2.2 ** d) * 1.1
                    x += sc * V[p][0]
                    y += sc * V[p][1]
                pos[s] = (x, y)
            return pos

    def draw_node_content(self, ax, s, x, y, faded: bool):
        """Rysuje miniaturowe slupki z kolorowymi krazkami wewnatrz wezla grafu."""
        n, k = self.n_disks, self.n_pegs
        w, h = self.node_w, self.node_h
        alpha = 0.35 if faded else 1.0
        base_y = y - h * 0.34
        peg_h = h * 0.56
        peg_spacing = (w - 0.16) / k
        left_x = x - (w - 0.16) / 2 + peg_spacing / 2

        # Podstawa i slupki
        ax.plot(
            [x - w * 0.42, x + w * 0.42],
            [base_y, base_y],
            color="#495057",
            lw=1.0,
            zorder=5,
            alpha=alpha,
        )
        pegs = [[] for _ in range(k)]
        for d in range(n - 1, -1, -1):
            pegs[s[d]].append(d)

        disk_h = min(0.085, peg_h / (n + 0.5))
        for p in range(k):
            px = left_x + p * peg_spacing
            ax.plot([px, px], [base_y, base_y + peg_h], color="#868e96", lw=0.9, zorder=5, alpha=alpha)
            for level, d in enumerate(pegs[p]):
                dw = peg_spacing * (0.38 + 0.50 * (d + 1) / n)
                dy = base_y + level * (disk_h + 0.01)
                col = DISK_COLORS[d % len(DISK_COLORS)]
                ax.add_patch(
                    Rectangle(
                        (px - dw / 2, dy),
                        dw,
                        disk_h,
                        fc=col,
                        ec="#212529",
                        lw=0.4,
                        zorder=6,
                        alpha=alpha,
                    )
                )

        # Podpis krotki na gorze wezla
        ax.text(
            x,
            y + h * 0.33,
            self.state_label(s),
            ha="center",
            va="center",
            fontsize=6.5,
            weight="bold",
            color="#868e96" if faded else "#212529",
            zorder=7,
        )

    def draw_state(self, ax, s):
        """Rysuje duzy podglad slupkow i krazkow dla wybranego stanu."""
        n, k = self.n_disks, self.n_pegs
        ax.clear()
        ax.add_patch(Rectangle((-0.6, -0.12), k - 1 + 1.2, 0.12, fc="#dee2e6", ec="#495057", lw=1.5))
        pegs = [[] for _ in range(k)]
        for d in range(n - 1, -1, -1):
            pegs[s[d]].append(d)

        dh = min(0.22, 0.9 / (n + 0.5))
        for p in range(k):
            ax.add_patch(Rectangle((p - 0.035, 0.0), 0.07, 1.02, fc="#868e96", ec="#343a40", lw=1.0))
            ax.text(p, -0.26, self.peg_name(p), ha="center", va="top", fontsize=12, weight="bold", color="#212529")
            for level, d in enumerate(pegs[p]):
                w = 0.26 + 0.56 * (d + 1) / n
                y0 = level * (dh + 0.02)
                col = DISK_COLORS[d % len(DISK_COLORS)]
                box = FancyBboxPatch(
                    (p - w / 2, y0 + 0.01),
                    w,
                    dh,
                    boxstyle="round,pad=0.01,rounding_size=0.04",
                    fc=col,
                    ec="#212529",
                    lw=1.2,
                    zorder=4,
                )
                ax.add_patch(box)
                ax.text(
                    p,
                    y0 + dh / 2 + 0.01,
                    str(d + 1),
                    ha="center",
                    va="center",
                    fontsize=9,
                    weight="bold",
                    color="white",
                    zorder=5,
                )
        ax.set_xlim(-0.75, k - 0.25)
        ax.set_ylim(-0.45, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")

    def definition_rows(self):
        n, k = self.n_disks, self.n_pegs
        peg_list = ", ".join(self.peg_name(i) for i in range(k))
        tuple_vars = ", ".join(f"p<sub>{i+1}</sub>" for i in range(n))
        states_desc = (
            f"krotka <code>({tuple_vars})</code>, gdzie <code>p<sub>i</sub> ∈ {{{peg_list}}}</code> oznacza słupek, "
            f"na którym leży krążek <i>i</i> (1 = najmniejszy, {n} = największy).<br>"
            f"<span style='color:#495057;font-size:90%'>Ponieważ większy krążek nigdy nie może leżeć na mniejszym, "
            f"samo przypisanie krążków do słupków jednoznacznie wyznacza ich kolejność od dołu do góry! "
            f"Liczba wszystkich stanów: <b>|S| = {k}<sup>{n}</sup> = {k ** n}</b>.</span>"
        )
        rules_desc = (
            f"<b>X → Y</b> (dla X, Y ∈ {{{peg_list}}}, X ≠ Y): przełóż wierzchołkowy (najmniejszy) krążek ze słupka X na słupek Y.<br>"
            f"<b>Warunek stosowalności:</b> słupek X jest niepusty oraz (słupek Y jest pusty lub górny krążek na X jest mniejszy niż górny krążek na Y).<br>"
            f"Łącznie par słupków (potencjalnych reguł): <b>{k} · ({k} − 1) = {k * (k - 1)}</b>."
        )
        return [
            ("Stany (S)", states_desc),
            ("Stan początkowy", f"<code>{html.escape(self.state_label(self.initial))}</code> — wszystkie krążki na słupku <b>A</b> ({self.pegs_representation(self.initial)})"),
            ("Test celu", f"<code>{html.escape(self.state_label(self.goal_state))}</code> — wszystkie krążki na słupku <b>{self.peg_name(k - 1)}</b>"),
            ("Akcje i model przejścia", rules_desc),
            ("Koszt akcji", "każde przełożenie pojedynczego krążka kosztuje <b>1</b> (szukamy minimalnej liczby ruchów)"),
        ]


# --------------------------------------------------------------------------
# Wizualizacje przestrzeni stanow i porownania BFS / DFS w Pythonie
# --------------------------------------------------------------------------
def show_hanoi_space(n_disks: int = 3, n_pegs: int = 3, include_unreachable: bool = True, show_solution: bool = True):
    """Wyswietla definicje formalna oraz graf przestrzeni stanow dla Wiez Hanoi."""
    p = Hanoi(n_disks=n_disks, n_pegs=n_pegs)
    ls.show(p, include_unreachable=include_unreachable, show_solution=show_solution, edge_labels=(n_disks <= 3 and n_pegs <= 3))


def show_hanoi_bfs_dfs(n_disks: int = 3, n_pegs: int = 3):
    """Porownuje obok siebie przebieg BFS (najkrotsza sciezka) oraz DFS na grafie Wiez Hanoi."""
    from IPython.display import HTML, display

    p = Hanoi(n_disks=n_disks, n_pegs=n_pegs)
    space = StateSpace(p, include_unreachable=(n_pegs == 2))
    bfs_steps = lsearch.trace_bfs_iterative(p, detailed=False)
    dfs_steps = lsearch.trace_dfs_iterative(p, detailed=False)
    bfs_end = bfs_steps[-1]
    dfs_end = dfs_steps[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.8))
    show_lbl = n_disks <= 2 and n_pegs <= 3
    lsearch.draw_search_step(
        space,
        bfs_end,
        ax=ax1,
        edge_labels=show_lbl,
        title=f"BFS (wszerz – Kolejka FIFO) | {n_disks} kr., {n_pegs} sł.",
    )
    lsearch.draw_search_step(
        space,
        dfs_end,
        ax=ax2,
        edge_labels=show_lbl,
        title=f"DFS (w głąb – Stos LIFO) | {n_disks} kr., {n_pegs} sł.",
    )
    fig.tight_layout()
    display(fig)
    plt.close(fig)

    bfs_len = len(bfs_end.solution) if bfs_end.solution is not None else "brak (cel nieosiągalny)"
    dfs_len = len(dfs_end.solution) if dfs_end.solution is not None else "brak (cel nieosiągalny)"
    bfs_seq = p.solution_text(bfs_end.solution) if bfs_end.solution else "—"
    dfs_seq = p.solution_text(dfs_end.solution) if dfs_end.solution else "—"

    tbl = _CSS + (
        "<table class='ls'>"
        "<tr><th>Algorytm</th><th>Wszystkich stanów (k<sup>n</sup>)</th><th>Stanów osiągalnych</th>"
        "<th>Odwiedzone stany</th><th>Długość znalezionej ścieżki</th><th>Sekwencja ruchów</th></tr>"
        f"<tr><td><b>🌊 BFS (najkrótsza ścieżka)</b></td>"
        f"<td style='text-align:center'>{len(space.all_states)}</td>"
        f"<td style='text-align:center'>{len(space.reachable)}</td>"
        f"<td style='text-align:center'><b>{len(bfs_end.visited)}</b></td>"
        f"<td style='text-align:center'><b>{bfs_len}</b></td>"
        f"<td><span style='font-size:90%'>{html.escape(bfs_seq)}</span></td></tr>"
        f"<tr><td><b>🕳️ DFS (w głąb)</b></td>"
        f"<td style='text-align:center'>{len(space.all_states)}</td>"
        f"<td style='text-align:center'>{len(space.reachable)}</td>"
        f"<td style='text-align:center'><b>{len(dfs_end.visited)}</b></td>"
        f"<td style='text-align:center'><b>{dfs_len}</b></td>"
        f"<td><span style='font-size:90%'>{html.escape(dfs_seq)}</span></td></tr>"
        "</table>"
    )
    display(HTML(tbl))


class HanoiSpaceExplorerApp:
    """Interaktywny eksplorator przestrzeni stanow, tabeli przejsc i przeszukiwania BFS/DFS dla Wiez Hanoi."""

    def __init__(self):
        import ipywidgets as W

        self.W = W
        self._busy = False
        st = {"description_width": "initial"}
        self.w_disks = W.IntSlider(value=3, min=2, max=4, step=1, description="Liczba krążków (n):", style=st)
        self.w_pegs = W.IntSlider(value=3, min=2, max=4, step=1, description="Liczba słupków (k):", style=st)
        self.w_unreach = W.Checkbox(value=True, description="Pokaż też stany nieosiągalne (ważne dla k=2!)", indent=False)

        self.w_algo = W.Dropdown(
            options=[
                ("BFS – wszerz (iteracyjny, Kolejka FIFO)", "bfs_iter"),
                ("DFS – w głąb (iteracyjny, Stos LIFO)", "dfs_iter"),
                ("DFS – w głąb (rekurencyjny, Stos wywołań)", "dfs_rec"),
            ],
            value="bfs_iter",
            description="Algorytm (animacja):",
            style=st,
            layout=W.Layout(width="360px"),
        )
        self.play = W.Play(value=0, min=0, max=10, step=1, interval=750)
        self.slider = W.IntSlider(value=0, min=0, max=10, step=1, description="Krok:", style=st, layout=W.Layout(width="340px"))
        W.jslink((self.play, "value"), (self.slider, "value"))

        for w in (self.w_disks, self.w_pegs, self.w_unreach, self.w_algo):
            w.observe(self.rebuild, names="value")
        self.slider.observe(self._render_anim_step, names="value")

        self.out_graph = W.Output()
        self.out_anim = W.Output()
        self.anim_banner = W.HTML()
        self.out_def = W.Output()
        self.out_table = W.Output()

        anim_tab = W.VBox([
            W.HBox([self.w_algo, self.play, self.slider], layout=W.Layout(flex_flow="row wrap", align_items="center", grid_gap="10px")),
            self.anim_banner,
            self.out_anim,
        ])

        tabs = W.Tab(children=[self.out_graph, anim_tab, self.out_def, self.out_table])
        tabs.set_title(0, "📐 Porównanie: BFS vs DFS na grafie")
        tabs.set_title(1, "🎬 Animacja krok po kroku (Kolejka / Stos)")
        tabs.set_title(2, "📋 Definicja formalna")
        tabs.set_title(3, "🔢 Tabela przejść")

        self.widget = W.VBox([
            W.HTML("<h3 style='margin:4px 0'>🗼 Przestrzeń stanów Wież Hanoi – eksplorator</h3>"),
            W.HBox([self.w_disks, self.w_pegs, self.w_unreach], layout=W.Layout(flex_flow="row wrap", grid_gap="14px")),
            tabs,
        ])
        self.rebuild()

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def rebuild(self, *_):
        if self._busy:
            return
        self._busy = True
        try:
            from IPython.display import HTML, display

            n = self.w_disks.value
            k = self.w_pegs.value
            self.problem = Hanoi(n_disks=n, n_pegs=k)
            self.space = StateSpace(self.problem, include_unreachable=self.w_unreach.value)
            sol = ls.solve(self.problem)
            self.anim_steps = lsearch.run_trace(self.problem, algo=self.w_algo.value, detailed=False)
            max_idx = max(0, len(self.anim_steps) - 1)
            self.play.max = max_idx
            self.slider.max = max_idx
            self.slider.value = max_idx
        finally:
            self._busy = False

        with self.out_def:
            self.out_def.clear_output(wait=True)
            display(HTML(ls.definition_html(self.space, sol)))

        with self.out_table:
            self.out_table.clear_output(wait=True)
            display(HTML(ls.transition_table_html(self.space)))

        with self.out_graph:
            self.out_graph.clear_output(wait=True)
            bfs_end = lsearch.trace_bfs_iterative(self.problem, detailed=False)[-1]
            dfs_end = lsearch.trace_dfs_iterative(self.problem, detailed=False)[-1]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.8))
            show_lbl = (k ** n) <= 16
            lsearch.draw_search_step(
                self.space,
                bfs_end,
                ax=ax1,
                edge_labels=show_lbl,
                title=f"BFS (wszerz – najkrótsza ścieżka) | |S| = {k}^{n} = {k**n}",
            )
            lsearch.draw_search_step(
                self.space,
                dfs_end,
                ax=ax2,
                edge_labels=show_lbl,
                title=f"DFS (w głąb – stos LIFO) | |S| = {k}^{n} = {k**n}",
            )
            fig.tight_layout()
            display(fig)
            plt.close(fig)

        self._render_anim_step()

    def _render_anim_step(self, *_):
        if self._busy or not getattr(self, "anim_steps", None):
            return
        from IPython.display import display

        idx = min(self.slider.value, len(self.anim_steps) - 1)
        step = self.anim_steps[idx]
        algo = self.w_algo.value
        self.anim_banner.value = (
            f"<div style='background:#fff9db;border-left:4px solid #f08c00;padding:6px 10px;"
            f"margin:4px 0;border-radius:4px;color:#212529;font-size:13px'>"
            f"<b>Krok {idx} / {len(self.anim_steps) - 1} — {html.escape(step.title)}</b><br>{step.message}</div>"
        )
        with self.out_anim:
            self.out_anim.clear_output(wait=True)
            fig, (ax_g, ax_s) = plt.subplots(
                1,
                2,
                figsize=(13.5, 5.4),
                gridspec_kw={"width_ratios": [1.85, 1.0]},
            )
            show_lbl = (self.problem.n_pegs ** self.problem.n_disks) <= 16
            lsearch.draw_search_step(
                self.space,
                step,
                ax=ax_g,
                edge_labels=show_lbl,
                title=f"Graf Wież Hanoi — {lsearch.ALGO_INFO[algo]['short']}",
            )
            lsearch.draw_structure_panel(ax_s, self.problem, step, algo)
            fig.tight_layout()
            display(fig)
            plt.close(fig)


def hanoi_space_app():
    """Uruchamia interaktywny eksplorator przestrzeni stanow i grafu Wiez Hanoi."""
    return HanoiSpaceExplorerApp()


# --------------------------------------------------------------------------
# Interaktywne 3-czesciowe cwiczenie w przegladarce (HTML + SVG + JS, bez ipywidgets):
#   1. Formalizacja stanu START (krotka + dozwolone akcje X -> Y) [Lab 2]
#   2. Analiza srodkow i celow (podcel / most w trojkacie Sierpinskiego) [Newell & Simon]
#   3. Najkrotsza sciezka START -> CEL na slupkach + porownanie z BFS i DFS [Lab 3 & 4]
# Dwa tryby: "exam" (📝 Egzamin - jednorazowe zatwierdzenie 🔒) oraz "playground" (🔓 Plac zabaw).
# --------------------------------------------------------------------------
HANOI_MODES = ("exam", "playground")

_HANOI_LOGIC_JS = r"""
var HN = (function () {
  var PEGS = ["A", "B", "C", "D", "E"];
  var COLORS = ["#e03131", "#f08c00", "#2f9e44", "#1971c2", "#9c36b5", "#0ca678"];

  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function stateKey(s) { return s.join(","); }
  function stateLabel(s) {
    return "(" + s.map(function (p) { return p === null || p < 0 ? "?" : PEGS[p]; }).join(", ") + ")";
  }
  function stateEq(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  function topDisks(s, k) {
    var top = [];
    for (var p = 0; p < k; p++) top.push(-1);
    for (var d = s.length - 1; d >= 0; d--) top[s[d]] = d;
    return top;
  }

  function allActionLabels(k) {
    var out = [];
    for (var u = 0; u < k; u++) {
      for (var v = 0; v < k; v++) {
        if (u !== v) out.push({ from: u, to: v, label: PEGS[u] + "→" + PEGS[v] });
      }
    }
    return out;
  }

  function validMoves(s, k) {
    var top = topDisks(s, k), out = [];
    for (var u = 0; u < k; u++) {
      if (top[u] < 0) continue;
      var du = top[u];
      for (var v = 0; v < k; v++) {
        if (u === v) continue;
        if (top[v] < 0 || du < top[v]) {
          var s2 = s.slice();
          s2[du] = v;
          out.push({ from: u, to: v, disk: du + 1, next: s2, label: PEGS[u] + "→" + PEGS[v] });
        }
      }
    }
    return out;
  }

  function isGoal(s, k, goalState) {
    if (goalState) return stateEq(s, goalState);
    for (var i = 0; i < s.length; i++) if (s[i] !== k - 1) return false;
    return true;
  }

  // Pelna mapa odleglosci BFS od zadanego stanu startowego
  function bfsDistances(n, k, startState) {
    var sk = stateKey(startState);
    var dist = {}, q = [startState.slice()];
    dist[sk] = 0;
    while (q.length) {
      var u = q.shift(), uk = stateKey(u), d = dist[uk];
      var moves = validMoves(u, k);
      for (var i = 0; i < moves.length; i++) {
        var vk = stateKey(moves[i].next);
        if (dist[vk] === undefined) {
          dist[vk] = d + 1;
          q.push(moves[i].next);
        }
      }
    }
    return dist;
  }

  // Rozwiazanie BFS (kolejka FIFO -> gwarantuje najkrotsza sciezke ze startState do goalState)
  function runBFS(n, k, startState, goalState) {
    var start = startState ? startState.slice() : [], i;
    if (!start.length) for (i = 0; i < n; i++) start.push(0);
    var goal = goalState ? goalState.slice() : null;
    var sk = stateKey(start);
    var q = [start], seen = {}, par = {}, order = [];
    seen[sk] = true;
    par[sk] = null;
    var goalKey = null;

    while (q.length) {
      var u = q.shift(), uk = stateKey(u);
      order.push(u);
      if (isGoal(u, k, goal)) { goalKey = uk; break; }
      var moves = validMoves(u, k);
      for (i = 0; i < moves.length; i++) {
        var mv = moves[i], vk = stateKey(mv.next);
        if (!seen[vk]) {
          seen[vk] = true;
          par[vk] = { prevKey: uk, prevState: u, move: mv, state: mv.next };
          q.push(mv.next);
        }
      }
    }
    var pathMoves = [], pathStates = [];
    if (goalKey !== null) {
      var cur = goalKey;
      while (par[cur] !== null) {
        pathMoves.unshift(par[cur].move);
        pathStates.unshift(par[cur].state);
        cur = par[cur].prevKey;
      }
      pathStates.unshift(start);
    }
    return {
      solved: goalKey !== null,
      visitedCount: order.length,
      visitedStates: order,
      moves: pathMoves,
      states: pathStates
    };
  }

  // Rozwiazanie DFS (stos LIFO -> sasiad w porzadku leksykograficznym par slupkow)
  function runDFS(n, k, startState, goalState) {
    var start = startState ? startState.slice() : [], i;
    if (!start.length) for (i = 0; i < n; i++) start.push(0);
    var goal = goalState ? goalState.slice() : null;
    var seen = {}, par = {}, order = [];
    var stack = [{ state: start, prevKey: null, move: null }];
    var goalKey = null;

    while (stack.length) {
      var item = stack.pop(), u = item.state, uk = stateKey(u);
      if (seen[uk]) continue;
      seen[uk] = true;
      order.push(u);
      par[uk] = item.prevKey === null ? null : item;
      if (isGoal(u, k, goal)) { goalKey = uk; break; }
      var moves = validMoves(u, k);
      for (i = moves.length - 1; i >= 0; i--) {
        var mv = moves[i], vk = stateKey(mv.next);
        if (!seen[vk]) {
          stack.push({ state: mv.next, prevKey: uk, move: mv });
        }
      }
    }
    var pathMoves = [], pathStates = [];
    if (goalKey !== null) {
      var cur = goalKey;
      while (par[cur] !== null) {
        pathMoves.unshift(par[cur].move);
        pathStates.unshift(par[cur].state);
        cur = par[cur].prevKey;
      }
      pathStates.unshift(start);
    }
    return {
      solved: goalKey !== null,
      visitedCount: order.length,
      visitedStates: order,
      moves: pathMoves,
      states: pathStates
    };
  }

  // Generowanie zadania (losowej lub klasycznej pary START -> CEL wraz z podcelem / mostem)
  function generateTask(n, k, seed, preset) {
    var i, start = [], goal = [], subgoal = [], bridgeOut = [];
    var u = 0, v = k - 1, w = 1;

    if (preset === "classic" || k !== 3) {
      for (i = 0; i < n; i++) { start.push(0); goal.push(k - 1); }
      u = 0; v = k - 1; w = k >= 3 ? 1 : 0;
      for (i = 0; i < n - 1; i++) { subgoal.push(w); bridgeOut.push(w); }
      subgoal.push(u);
      bridgeOut.push(v);
    } else {
      var r = rng(seed);
      var subStates = [];
      (function genSub(prefix) {
        if (prefix.length === n - 1) { subStates.push(prefix); return; }
        for (var p = 0; p < 3; p++) genSub(prefix.concat([p]));
      })([]);

      u = Math.floor(r() * 3);
      v = (u + 1 + Math.floor(r() * 2)) % 3;
      w = 3 - u - v;

      for (i = 0; i < n - 1; i++) { subgoal.push(w); bridgeOut.push(w); }
      subgoal.push(u);
      bridgeOut.push(v);

      var distFromSub = bfsDistances(n, 3, subgoal);
      var distFromOut = bfsDistances(n, 3, bridgeOut);

      // Odleglosc przez drogi okrezne (przez trzeci podtrojkat w):
      // wejscie do w wymaga stanu (v,...,v,u) -> (v,...,v,w), przejscie przez w (2^(n-1)-1), i (u,...,u,w) -> (u,...,u,v)
      var altIn = [], altOut = [];
      for (i = 0; i < n - 1; i++) { altIn.push(v); altOut.push(u); }
      altIn.push(u); altOut.push(v);
      var distFromAltIn = bfsDistances(n, 3, altIn);
      var distFromAltOut = bfsDistances(n, 3, altOut);
      var wTraverse = Math.pow(2, n - 1) - 1;

      var minTotal = n === 2 ? 2 : (n === 3 ? 4 : 6);
      var maxTotal = n === 2 ? 3 : (n === 3 ? 7 : 10);
      var minSide = 1;
      var maxSide = n === 2 ? 1 : (n === 3 ? 3 : 5);

      var candidates = [];
      for (var si = 0; si < subStates.length; si++) {
        var sCand = subStates[si].concat([u]);
        var sk = stateKey(sCand);
        var d1 = distFromSub[sk];
        if (d1 < minSide || d1 > maxSide) continue;
        for (var gi = 0; gi < subStates.length; gi++) {
          var gCand = subStates[gi].concat([v]);
          var gk = stateKey(gCand);
          var d2 = distFromOut[gk];
          if (d2 < minSide || d2 > maxSide) continue;
          var directLen = d1 + 1 + d2;
          if (directLen < minTotal || directLen > maxTotal) continue;
          var indirectLen = distFromAltIn[sk] + 1 + wTraverse + 1 + distFromAltOut[gk];
          if (directLen >= indirectLen) continue; // gwarantuje jednoznacznosc mostu na najkrotszej sciezce
          // Pomin klasyczna pare, w ktorej wszystkie krazki leza na jednym slupku w obu stanach
          var allSameS = sCand.every(function (p) { return p === u; });
          var allSameG = gCand.every(function (p) { return p === v; });
          if (n >= 3 && allSameS && allSameG) continue;
          candidates.push({ start: sCand, goal: gCand, len: directLen });
        }
      }
      if (!candidates.length) {
        // Fallback awaryjny
        start = [w, u];
        while (start.length < n) start.unshift(u);
        goal = [w, v];
        while (goal.length < n) goal.unshift(v);
      } else {
        var pick = candidates[Math.floor(r() * candidates.length)];
        start = pick.start;
        goal = pick.goal;
      }
    }

    var bfs = runBFS(n, k, start, goal);
    var dfs = runDFS(n, k, start, goal);
    var validStart = validMoves(start, k).map(function (m) { return m.label; });

    return {
      n: n,
      k: k,
      seed: seed,
      preset: preset || "random",
      start: start,
      goal: goal,
      subgoal: subgoal,
      bridgeOut: bridgeOut,
      fromPeg: u,
      toPeg: v,
      auxPeg: w,
      bridgeMoveLabel: PEGS[u] + "→" + PEGS[v],
      validStartActions: validStart,
      bfs: bfs,
      dfs: dfs
    };
  }

  // ---- Funkcje sprawdzajace 3 zadania ----
  function checkTask1(task, givenTuple, selectedActions) {
    var n = task.n, k = task.k, start = task.start;
    if (!givenTuple || givenTuple.length !== n || givenTuple.some(function (p) { return p === null || p === undefined || p < 0; })) {
      return { ok: false, msg: "Część 1a: wybierz słupek (A, B lub C) dla każdego krążka p₁ … p" + n + " w stanie START." };
    }
    var tupleOk = stateEq(givenTuple, start);
    var tupleMsg = "";
    if (!tupleOk) {
      var rev = start.slice().reverse();
      if (stateEq(givenTuple, rev) && !stateEq(start, rev)) {
        tupleMsg = "Uwaga na kolejność w krotce! Zapisałeś(aś) krążki od największego do najmniejszego, a p₁ to krążek #1 (najmniejszy), …, p" + n + " to krążek #" + n + " (największy).";
      } else {
        for (var i = 0; i < n; i++) {
          if (givenTuple[i] !== start[i]) {
            tupleMsg = "W stanie START krążek #" + (i + 1) + " (p" + (i + 1) + ") nie leży na słupku " + PEGS[givenTuple[i]] + " – spójrz na rysunek stanu START.";
            break;
          }
        }
      }
    }

    var acts = selectedActions || [];
    if (!acts.length) {
      return {
        ok: false,
        msg: (tupleOk ? "Krotka stanu START " + stateLabel(start) + " poprawna! " : "Błąd w krotce (" + tupleMsg + ") oraz ") +
             "Część 1b: zaznacz przynajmniej jedną dozwoloną akcję (X→Y) w stanie START."
      };
    }

    var expected = task.validStartActions;
    var top = topDisks(start, k);
    var wrongActMsg = "";
    for (var a = 0; a < acts.length; a++) {
      var lbl = acts[a];
      if (expected.indexOf(lbl) < 0) {
        var u = PEGS.indexOf(lbl.charAt(0));
        var v = PEGS.indexOf(lbl.charAt(lbl.length - 1));
        if (top[u] < 0) {
          wrongActMsg = "Akcja <b>" + lbl + "</b> jest niedozwolona w stanie START, bo słupek <b>" + PEGS[u] + "</b> jest pusty.";
        } else {
          wrongActMsg = "Akcja <b>" + lbl + "</b> jest niedozwolona w stanie START: na słupku <b>" + PEGS[u] +
            "</b> na wierzchu leży krążek <b>#" + (top[u] + 1) + "</b>, którego nie wolno położyć na mniejszy krążek <b>#" +
            (top[v] + 1) + "</b> na słupku <b>" + PEGS[v] + "</b>.";
        }
        break;
      }
    }
    if (!wrongActMsg && acts.length < expected.length) {
      wrongActMsg = "Zaznaczone akcje są poprawne, ale to nie wszystkie! W stanie START dozwolone są <b>" +
        expected.length + "</b> akcje (zaznaczono " + acts.length + ").";
    }

    if (tupleOk && !wrongActMsg) {
      return {
        ok: true,
        msg: "Świetnie! Stan START to <b>" + stateLabel(start) + "</b>, a dozwolone w nim akcje to: <b>" + expected.join(", ") + "</b>."
      };
    }
    if (tupleOk && wrongActMsg) {
      return { ok: false, msg: "Krotka stanu START <b>" + stateLabel(start) + "</b> poprawna, ale w części 1b: " + wrongActMsg };
    }
    if (!tupleOk && !wrongActMsg) {
      return { ok: false, msg: "Dozwolone akcje zaznaczone bezbłędnie! Błąd w części 1a (krotka stanu START): " + tupleMsg };
    }
    return { ok: false, msg: "Część 1a: " + tupleMsg + " Część 1b: " + wrongActMsg };
  }

  function checkTask2(task, givenTuple) {
    var n = task.n, u = task.fromPeg, v = task.toPeg, w = task.auxPeg;
    if (!givenTuple || givenTuple.length !== n || givenTuple.some(function (p) { return p === null || p === undefined || p < 0; })) {
      return { ok: false, msg: "Wybierz słupek dla każdego krążka p₁ … p" + n + ", aby określić stan-podcel przed ruchem największego krążka #" + n + "." };
    }
    if (stateEq(givenTuple, task.subgoal)) {
      return {
        ok: true,
        msg: "Doskonale! Stan <b>" + stateLabel(task.subgoal) + "</b> to wejście na most w trójkącie Sierpińskiego: " +
             "mniejsze krążki (1.." + (n - 1) + ") są zebrane na słupku pomocniczym <b>" + PEGS[w] +
             "</b>, dzięki czemu największy krążek <b>#" + n + "</b> może wykonać ruch <b>" + task.bridgeMoveLabel + "</b>!"
      };
    }
    if (stateEq(givenTuple, task.bridgeOut)) {
      return {
        ok: false,
        msg: "Bardzo blisko! Stan <b>" + stateLabel(task.bridgeOut) + "</b> to stan tuż <b>PO</b> przełożeniu krążka #" + n +
             " na słupek " + PEGS[v] + " (wyjście z mostu). Wskaż stan-podcel tuż <b>PRZED</b> tym ruchem (gdy krążek #" + n +
             " leży jeszcze na słupku <b>" + PEGS[u] + "</b>)."
      };
    }
    if (givenTuple[n - 1] !== u) {
      return {
        ok: false,
        msg: "W stanie-podcelu (tuż przed przejściem przez most) największy krążek <b>#" + n + "</b> (p" + n +
             ") musi jeszcze leżeć na swoim początkowym słupku <b>" + PEGS[u] + "</b> (wybrano " + PEGS[givenTuple[n - 1]] + ")."
      };
    }
    for (var d = 0; d < n - 1; d++) {
      if (givenTuple[d] === u) {
        return {
          ok: false,
          msg: "W stanie <b>" + stateLabel(givenTuple) + "</b> mniejszy krążek <b>#" + (d + 1) + "</b> nadal leży na słupku <b>" +
               PEGS[u] + "</b> (na krążku #" + n + ") i blokuje jego podniesienie! Wszystkie mniejsze krążki muszą znaleźć się na słupku pomocniczym <b>" + PEGS[w] + "</b>."
        };
      }
      if (givenTuple[d] === v) {
        return {
          ok: false,
          msg: "W stanie <b>" + stateLabel(givenTuple) + "</b> mniejszy krążek <b>#" + (d + 1) + "</b> leży na słupku docelowym <b>" +
               PEGS[v] + "</b> i nie pozwala odłożyć tam większego krążka <b>#" + n + "</b>! Mniejsze krążki muszą być na słupku pomocniczym <b>" + PEGS[w] + "</b>."
        };
      }
    }
    return {
      ok: false,
      msg: "Stan <b>" + stateLabel(givenTuple) + "</b> nie pozwala jeszcze wykonać ruchu <b>" + task.bridgeMoveLabel + "</b> krążkiem #" + n + "."
    };
  }

  function checkTask3(task, finalState, historyMoves) {
    var movesCount = historyMoves ? historyMoves.length : 0;
    var opt = task.bfs.moves.length;
    if (!stateEq(finalState, task.goal)) {
      if (movesCount === 0) {
        return { ok: false, msg: "Nie wykonano jeszcze żadnego ruchu na słupkach – przełóż krążki ze stanu START do stanu CEL." };
      }
      return {
        ok: false,
        msg: "Wieża na słupkach nie jest jeszcze w stanie docelowym CEL (wykonano " + nMoves(movesCount) + "). Dokończ układanie wieży!"
      };
    }
    if (movesCount > opt) {
      return {
        ok: false,
        msg: "Ułożono stan CEL, ale w <b>" + nMoves(movesCount) + "</b> – a najkrótsza ścieżka (BFS) liczy <b>" +
             nMoves(opt) + "</b> (o " + nMoves(movesCount - opt) + " mniej)."
      };
    }
    return {
      ok: true,
      msg: "Najkrótsza ścieżka – <b>" + nMoves(opt) + "</b>! Rozwiązałeś(aś) zadanie optymalnie, dokładnie tak jak algorytm <b>BFS</b> " +
           "(podczas gdy DFS potrzebował aż <b>" + nMoves(task.dfs.moves.length) + "</b>)."
    };
  }

  // Generowanie wspolrzednych grafu przestrzeni stanow (dla wizualizacji SVG do 256 stanow)
  function buildGraphLayout(n, k) {
    var total = Math.pow(k, n);
    if (total > 256) return null;
    var states = [];
    (function gen(prefix) {
      if (prefix.length === n) { states.push(prefix); return; }
      for (var p = 0; p < k; p++) gen(prefix.concat([p]));
    })([]);

    var rawPos = {}, d;
    if (k === 3) {
      var V = [[-1, 0], [0, -Math.sqrt(3)], [1, 0]];
      states.forEach(function (s) {
        var perm = [0, 1, 2], x = 0, y = 0;
        for (d = n - 1; d >= 0; d--) {
          var p = s[d], c = perm[p], sc = Math.pow(2, d);
          x += sc * V[c][0];
          y += sc * V[c][1];
          var oth = [0, 1, 2].filter(function (q) { return q !== p; });
          var tmp = perm[oth[0]]; perm[oth[0]] = perm[oth[1]]; perm[oth[1]] = tmp;
        }
        rawPos[stateKey(s)] = [x, y];
      });
    } else if (k === 2) {
      states.forEach(function (s) {
        var col = s[0], row = 0;
        for (d = 1; d < n; d++) row += s[d] * Math.pow(2, d - 1);
        rawPos[stateKey(s)] = [col * 2.2, row * 1.2];
      });
    } else {
      states.forEach(function (s) {
        var x = 0, y = 0;
        for (d = n - 1; d >= 0; d--) {
          var ang = -Math.PI / 2 + 2 * Math.PI * s[d] / k;
          var sc = Math.pow(2.25, d);
          x += sc * Math.cos(ang);
          y += sc * Math.sin(ang);
        }
        rawPos[stateKey(s)] = [x, y];
      });
    }

    var xs = [], ys = [];
    for (var key in rawPos) { xs.push(rawPos[key][0]); ys.push(rawPos[key][1]); }
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var W = 380, H = 300, pad = 24;
    var spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
    var scale = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanY);
    var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;

    var pos = {};
    for (var k2 in rawPos) {
      pos[k2] = [W / 2 + (rawPos[k2][0] - cx) * scale, H / 2 + (rawPos[k2][1] - cy) * scale];
    }

    var edges = [], edgeSeen = {};
    states.forEach(function (s) {
      var uk = stateKey(s);
      validMoves(s, k).forEach(function (mv) {
        var vk = stateKey(mv.next);
        var ek = uk < vk ? uk + "|" + vk : vk + "|" + uk;
        if (!edgeSeen[ek]) { edgeSeen[ek] = true; edges.push([uk, vk]); }
      });
    });

    return { W: W, H: H, states: states, pos: pos, edges: edges };
  }

  function plural(k, one, few, many) {
    if (k === 1) return one;
    return (k % 10 >= 2 && k % 10 <= 4 && (k % 100 < 12 || k % 100 > 14)) ? few : many;
  }
  function nMoves(k) { return k + " " + plural(k, "ruch", "ruchy", "ruchów"); }
  function nStates(k) { return k + " " + plural(k, "stan", "stany", "stanów"); }

  return {
    PEGS: PEGS, COLORS: COLORS, rng: rng, stateKey: stateKey, stateLabel: stateLabel, stateEq: stateEq,
    topDisks: topDisks, allActionLabels: allActionLabels, validMoves: validMoves, isGoal: isGoal,
    bfsDistances: bfsDistances, runBFS: runBFS, runDFS: runDFS, generateTask: generateTask,
    checkTask1: checkTask1, checkTask2: checkTask2, checkTask3: checkTask3,
    buildGraphLayout: buildGraphLayout, nMoves: nMoves, nStates: nStates, plural: plural
  };
})();
if (typeof module === "object" && module && module.exports && typeof window === "undefined") module.exports = HN;
"""

_HANOI_UI_JS = r"""
(function () {
  var root = document.getElementById("__ROOT__");
  var CFG = __CFG__;
  var NS = "http://www.w3.org/2000/svg";
  function $(sel) { return root.querySelector(sel); }
  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  var RULES = {
    exam: "📏 <b>Zasady (egzamin):</b> Dla wylosowanych stanów <b>START</b> i <b>CEL</b> rozwiąż 3 zadania: " +
          "<b>(1)</b> zapisz stan START jako krotkę i wskaż wszystkie dozwolone w nim akcje, " +
          "<b>(2)</b> wyznacz <b>stan-podcel</b> (wejście na „most” w trójkącie Sierpińskiego), który pozwala ruszyć największy krążek, " +
          "<b>(3)</b> przełóż krążki ze START do CEL na słupkach w <b>minimalnej liczbie ruchów</b>.<br>" +
          "🔒 Odpowiedzi <b>zatwierdzasz raz</b> – dopiero wtedy zobaczysz wynik (0–3 pkt), poprawne odpowiedzi oraz porównanie z <b>BFS</b> i <b>DFS</b>. Kolejna próba = <b>🎲 Nowe zadanie</b>.",
    playground: "🔓 <b>Plac zabaw:</b> Możesz sprawdzać odpowiedzi (<b>✔ Sprawdź</b>) wielokrotnie, podejrzeć rozwiązanie (<b>👁 Pokaż rozwiązanie</b>), " +
          "klikać węzły bezpośrednio na grafie Sierpińskiego, przełączyć układ na <b>klasyczny (A → C)</b> oraz w każdej chwili otworzyć <b>📊 Porównanie z BFS i DFS</b>."
  };

  var S = {
    mode: CFG.mode,
    n: CFG.n_disks,
    k: 3,
    seed: CFG.seed,
    preset: "random",
    task: null,
    layout: null,
    // Odpowiedzi do 3 zadan:
    ans1Tuple: [],
    ans1Actions: [],
    ans2Tuple: [],
    // Stan gry na slupkach (Zadanie 3):
    state: [],
    animState: null,
    selectedPeg: null,
    history: [],
    trailStates: [],
    visitedSet: {},
    // Wyniki sprawdzania i stan UI:
    res: {},
    reveal: false,
    locked: false,
    confirm: false,
    showCompare: false,
    overlay: "user",
    msg: "",
    msgKind: "info",
    note: "",
    animTimer: null
  };

  function exam() { return S.mode === "exam"; }
  function randomSeed() { return Math.floor(Math.random() * 100000); }

  function stopAnim() {
    if (S.animTimer) { clearInterval(S.animTimer); S.animTimer = null; }
    S.animState = null;
  }

  function newTask(seed) {
    stopAnim();
    S.n = parseInt($(".hn-disks").value, 10);
    S.preset = exam() ? "random" : $(".hn-preset").value;
    S.seed = seed;
    $(".hn-seed").value = seed;

    S.task = HN.generateTask(S.n, S.k, S.seed, S.preset);
    S.layout = HN.buildGraphLayout(S.n, S.k);

    S.ans1Tuple = [];
    S.ans2Tuple = [];
    for (var i = 0; i < S.n; i++) { S.ans1Tuple.push(null); S.ans2Tuple.push(null); }
    S.ans1Actions = [];

    resetTowersToStart(true);

    S.res = {};
    S.reveal = false;
    S.locked = false;
    S.confirm = false;
    S.showCompare = false;
    S.overlay = "user";
    S.note = "";
    $(".hn-show").textContent = "👁 Pokaż rozwiązanie";
    render();
  }

  function resetTowersToStart(silent) {
    stopAnim();
    S.state = S.task.start.slice();
    S.selectedPeg = null;
    S.history = [];
    S.trailStates = [S.state.slice()];
    S.visitedSet = {};
    S.visitedSet[HN.stateKey(S.state)] = true;
    delete S.res.t3;
    S.confirm = false;
    if (!silent) {
      S.msg = "Przywrócono wieżę do stanu początkowego START.";
      S.msgKind = "info";
      render();
    } else {
      S.msg = "Kliknij słupek na planszy poniżej, aby podnieść górny krążek, a następnie słupek docelowy (doprowadź wieżę ze START do CEL).";
      S.msgKind = "info";
    }
  }

  function setMode(m) {
    if (S.mode === m) return;
    stopAnim();
    S.mode = m;
    S.res = {};
    S.note = "";
    S.locked = false;
    S.confirm = false;
    S.reveal = false;
    S.showCompare = false;
    if (m === "exam") {
      $(".hn-preset").value = "random";
      newTask(randomSeed());
      return;
    }
    render();
  }

  function clickPeg(p) {
    if (S.locked && !S.showCompare) {
      S.note = "🔒 Odpowiedzi są już zatwierdzone. Kliknij 🎲 Nowe zadanie, aby spróbować ponownie.";
      render();
      return;
    }
    stopAnim();
    S.confirm = false;
    var top = HN.topDisks(S.state, S.k);
    if (S.selectedPeg === null) {
      if (top[p] < 0) {
        S.msg = "⚠️ Słupek " + HN.PEGS[p] + " jest pusty – wybierz słupek z krążkami.";
        S.msgKind = "warn";
        render();
        return;
      }
      S.selectedPeg = p;
      S.msg = "Podniesiono krążek <b>#" + (top[p] + 1) + "</b> ze słupka <b>" + HN.PEGS[p] + "</b>. Kliknij słupek, na który chcesz go odłożyć.";
      S.msgKind = "info";
      render();
      return;
    }

    if (S.selectedPeg === p) {
      S.selectedPeg = null;
      S.msg = "Odłożono krążek z powrotem na słupek " + HN.PEGS[p] + ".";
      S.msgKind = "info";
      render();
      return;
    }

    var u = S.selectedPeg;
    var du = top[u];
    if (top[p] >= 0 && du > top[p]) {
      S.msg = "⛔ <b>Ruch niedozwolony!</b> Nie wolno kłaść większego krążka (<b>#" + (du + 1) + "</b>) na mniejszy krążek (<b>#" + (top[p] + 1) + "</b> na słupku " + HN.PEGS[p] + ").";
      S.msgKind = "bad";
      render();
      return;
    }

    var next = S.state.slice();
    next[du] = p;
    var mv = { from: u, to: p, disk: du + 1, next: next, label: HN.PEGS[u] + "→" + HN.PEGS[p] };
    S.state = next;
    S.selectedPeg = null;
    S.history.push(mv);
    S.trailStates.push(next.slice());
    S.visitedSet[HN.stateKey(next)] = true;
    delete S.res.t3;

    if (HN.stateEq(S.state, S.task.goal)) {
      if (!exam()) {
        var opt = S.task.bfs.moves.length;
        var diff = S.history.length - opt;
        S.msg = diff === 0
          ? "🏆 <b>Ułożono stan CEL w minimalnej liczbie ruchów (" + HN.nMoves(opt) + ")!</b> Kliknij <b>✔ Sprawdź</b>, aby ocenić wszystkie zadania."
          : "🟡 <b>Ułożono stan CEL w " + HN.nMoves(S.history.length) + ".</b> (Najkrótsza ścieżka BFS ma " + HN.nMoves(opt) + ").";
      } else {
        S.msg = "🟡 <b>Ułożono stan docelowy CEL</b> (liczba wykonanych ruchów: <b>" + S.history.length + "</b>). Gdy ukończysz wszystkie 3 zadania, kliknij <b>🔒 Zatwierdź odpowiedzi</b>.";
      }
      S.msgKind = "ok";
    } else if (HN.stateEq(S.state, S.task.subgoal)) {
      S.msg = "🎯 Wykonano ruch: krążek <b>#" + (du + 1) + "</b> (<b>" + mv.label + "</b>). Dotarłeś(aś) do stanu-podcelu – największy krążek <b>#" + S.n + "</b> ma teraz wolną drogę!";
      S.msgKind = "info";
    } else {
      S.msg = "Wykonano ruch: krążek <b>#" + (du + 1) + "</b> (<b>" + mv.label + "</b>). Łącznie ruchów: <b>" + S.history.length + "</b>.";
      S.msgKind = "info";
    }
    render();
  }

  function undoMove() {
    if (S.locked && !S.showCompare) return;
    stopAnim();
    if (!S.history.length) return;
    S.history.pop();
    S.trailStates.pop();
    S.state = S.trailStates[S.trailStates.length - 1].slice();
    S.selectedPeg = null;
    delete S.res.t3;
    S.confirm = false;
    S.msg = "Cofnięto ostatni ruch (pozostało ruchów: " + S.history.length + ").";
    S.msgKind = "info";
    render();
  }

  function playSequence(states, moves, label, overlayMode) {
    stopAnim();
    if (!states || !states.length) return;
    S.overlay = overlayMode;
    S.showCompare = true;
    S.selectedPeg = null;
    var idx = 0;
    S.animState = states[0].slice();
    S.msg = "▶️ Odtwarzanie (" + label + "): start <code>" + HN.stateLabel(S.animState) + "</code> (0 / " + moves.length + ")";
    S.msgKind = "info";
    render();

    S.animTimer = setInterval(function () {
      if (!document.body.contains(root)) { stopAnim(); return; }
      idx++;
      if (idx >= states.length) {
        if (S.animTimer) { clearInterval(S.animTimer); S.animTimer = null; }
        S.animState = null;
        S.msg = "✅ Zakończono odtwarzanie: <b>" + label + "</b> (" + HN.nMoves(moves.length) + "). Przywrócono widok Twojej wieży.";
        S.msgKind = "ok";
        render();
        return;
      }
      S.animState = states[idx].slice();
      var mv = moves[idx - 1];
      S.msg = "▶️ Odtwarzanie (" + label + ") – krok " + idx + " / " + moves.length + ": krążek <b>#" + mv.disk + "</b> (" + mv.label + ") → <code>" + HN.stateLabel(S.animState) + "</code>";
      render();
    }, moves.length > 25 ? 260 : 540);
  }

  // Rysowanie miniaturowego podgladu stanu (dla START i CEL)
  function drawMiniState(svg, state, accentColor) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = 230, H = 112, baseY = 88, pegTopY = 20;
    var n = S.n, k = S.k, spacing = (W - 24) / k;

    el("rect", { x: 10, y: baseY, width: W - 20, height: 7, rx: 3, fill: "#dee2e6", stroke: "#495057", "stroke-width": 1.2 }, svg);
    var pegs = [];
    for (var p = 0; p < k; p++) pegs.push([]);
    for (var d = n - 1; d >= 0; d--) pegs[state[d]].push(d);

    var maxW = Math.min(58, spacing * 0.84);
    var minW = Math.max(20, maxW * 0.38);
    var diskH = Math.min(14, Math.floor((baseY - pegTopY - 8) / n));

    for (var p2 = 0; p2 < k; p2++) {
      var px = 12 + spacing * (p2 + 0.5);
      el("rect", { x: px - 2.5, y: pegTopY, width: 5, height: baseY - pegTopY, rx: 2, fill: "#868e96", stroke: "#343a40", "stroke-width": 0.8 }, svg);
      el("text", { x: px, y: baseY + 18, "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: accentColor || "#212529" }, svg, HN.PEGS[p2]);
      var stack = pegs[p2];
      for (var idx = 0; idx < stack.length; idx++) {
        var diskIdx = stack[idx];
        var w = n === 1 ? maxW : minW + (maxW - minW) * (diskIdx / (n - 1));
        var y = baseY - (idx + 1) * diskH - 1;
        var col = HN.COLORS[diskIdx % HN.COLORS.length];
        el("rect", { x: px - w / 2, y: y, width: w, height: diskH - 1.5, rx: 3.5, fill: col, stroke: "#212529", "stroke-width": 0.9 }, svg);
        el("text", { x: px, y: y + (diskH - 1.5) / 2 + 3.5, "text-anchor": "middle", "font-size": 9.5, "font-weight": 700, fill: "#fff" }, svg, String(diskIdx + 1));
      }
    }
  }

  function drawTowers() {
    var svg = $(".hn-towers-svg");
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = 520, H = 240;
    var baseY = 196, pegTopY = 44;
    var n = S.n, k = S.k;
    var spacing = (W - 50) / k;
    var curState = S.animState || S.state;
    var top = HN.topDisks(curState, k);
    var showHints = !exam() || S.locked;

    el("rect", { x: 18, y: baseY, width: W - 36, height: 12, rx: 5, fill: "#dee2e6", stroke: "#495057", "stroke-width": 1.6 }, svg);

    var pegs = [], goalPegs = [];
    for (var p = 0; p < k; p++) { pegs.push([]); goalPegs.push([]); }
    for (var d = n - 1; d >= 0; d--) {
      pegs[curState[d]].push(d);
      goalPegs[S.task.goal[d]].push(d);
    }

    var maxDiskW = Math.min(124, spacing * 0.84);
    var minDiskW = Math.max(34, maxDiskW * 0.35);
    var diskH = Math.min(22, Math.floor((baseY - pegTopY - 18) / n));

    for (var p2 = 0; p2 < k; p2++) (function (p) {
      var px = 25 + spacing * (p + 0.5);
      var isSel = S.selectedPeg === p && !S.animState;
      var canDrop = false;
      if (S.selectedPeg !== null && S.selectedPeg !== p && !S.animState) {
        var du = top[S.selectedPeg];
        canDrop = (top[p] < 0 || du < top[p]);
      }

      var grp = el("g", { "class": "hn-peg-grp" }, svg);
      grp.addEventListener("click", function () { clickPeg(p); });

      var bgFill = isSel ? "#fff9db" : (showHints && canDrop ? "#ebfbee" : "transparent");
      var bgStroke = isSel ? "#f08c00" : (showHints && canDrop ? "#2f9e44" : "none");
      el("rect", {
        x: px - spacing * 0.46, y: 12, width: spacing * 0.92, height: baseY + 26,
        rx: 8, fill: bgFill, stroke: bgStroke, "stroke-width": 2, "stroke-dasharray": (showHints && canDrop) ? "5 4" : "none"
      }, grp);

      el("rect", {
        x: px - 5, y: pegTopY, width: 10, height: baseY - pegTopY, rx: 4,
        fill: isSel ? "#f08c00" : "#868e96", stroke: "#343a40", "stroke-width": 1.2
      }, grp);

      el("text", {
        x: px, y: baseY + 26, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "#212529"
      }, grp, "Słupek " + HN.PEGS[p]);

      if (showHints && canDrop) {
        el("text", { x: px, y: 28, "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: "#2f9e44" }, grp, "⬇ połóż tutaj");
      }

      // Delikatne kontury docelowego ukladu CEL (duchy krazkow)
      var gStack = goalPegs[p];
      for (var gIdx = 0; gIdx < gStack.length; gIdx++) {
        var gd = gStack[gIdx];
        var gw = n === 1 ? maxDiskW : minDiskW + (maxDiskW - minDiskW) * (gd / (n - 1));
        var gy = baseY - (gIdx + 1) * diskH - 2;
        el("rect", {
          x: px - gw / 2, y: gy, width: gw, height: diskH - 2, rx: 6,
          fill: "none", stroke: HN.COLORS[gd % HN.COLORS.length], "stroke-width": 1.2,
          "stroke-dasharray": "3 3", "stroke-opacity": 0.45
        }, grp);
      }

      // Aktualne krazki na slupku
      var stack = pegs[p];
      for (var idx = 0; idx < stack.length; idx++) {
        var diskIdx = stack[idx];
        var w = n === 1 ? maxDiskW : minDiskW + (maxDiskW - minDiskW) * (diskIdx / (n - 1));
        var isLifted = isSel && idx === stack.length - 1;
        var y = isLifted ? (pegTopY - diskH - 6) : (baseY - (idx + 1) * diskH - 2);
        var col = HN.COLORS[diskIdx % HN.COLORS.length];

        el("rect", {
          x: px - w / 2, y: y, width: w, height: diskH - 2, rx: 6,
          fill: col, stroke: isLifted ? "#212529" : "#343a40", "stroke-width": isLifted ? 2.5 : 1.2
        }, grp);
        el("text", {
          x: px, y: y + (diskH - 2) / 2 + 4, "text-anchor": "middle",
          "font-size": 11.5, "font-weight": 700, fill: "#ffffff"
        }, grp, "#" + (diskIdx + 1));
      }
    })(p2);
  }

  function drawStateGraph() {
    var box = $(".hn-graph-box");
    var svg = $(".hn-graph-svg");
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    if (!S.layout) {
      box.style.display = "none";
      return;
    }
    box.style.display = "";
    var L = S.layout;
    svg.setAttribute("viewBox", "0 0 " + L.W + " " + L.H);

    function eKey(a, b) { return a < b ? a + "|" + b : b + "|" + a; }
    function buildEdgeMap(states) {
      var m = {};
      if (!states) return m;
      for (var i = 0; i + 1 < states.length; i++) {
        m[eKey(HN.stateKey(states[i]), HN.stateKey(states[i + 1]))] = true;
      }
      return m;
    }

    var canSeeBfsDfs = !exam() || S.locked;
    var userEdges = buildEdgeMap(S.trailStates);
    var bfsEdges = buildEdgeMap(S.task.bfs.states);
    var dfsEdges = buildEdgeMap(S.task.dfs.states);
    var bridgeKey = eKey(HN.stateKey(S.task.subgoal), HN.stateKey(S.task.bridgeOut));
    var showBridgeSolution = (exam() ? S.locked : (S.reveal || (S.res.t2 && S.res.t2.ok)));

    L.edges.forEach(function (e) {
      var p1 = L.pos[e[0]], p2 = L.pos[e[1]], ek = eKey(e[0], e[1]);
      el("line", { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], stroke: "#dee2e6", "stroke-width": 1.6 }, svg);

      if (canSeeBfsDfs && (S.overlay === "dfs" || S.overlay === "all") && dfsEdges[ek]) {
        el("line", { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], stroke: "#9c36b5", "stroke-width": 3.0, "stroke-opacity": 0.65 }, svg);
      }
      if (canSeeBfsDfs && (S.overlay === "bfs" || S.overlay === "all") && bfsEdges[ek]) {
        el("line", { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], stroke: "#1971c2", "stroke-width": 3.8, "stroke-opacity": 0.9 }, svg);
      }
      if ((S.overlay === "user" || S.overlay === "all") && userEdges[ek]) {
        el("line", {
          x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
          stroke: "#f08c00", "stroke-width": 2.8, "stroke-dasharray": S.overlay === "all" ? "4 3" : "none"
        }, svg);
      }
      if (showBridgeSolution && ek === bridgeKey) {
        el("line", {
          x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
          stroke: "#e8590c", "stroke-width": 4.5, "stroke-opacity": 0.95
        }, svg);
      }
    });

    var curState = S.animState || S.state;
    var curKey = HN.stateKey(curState);
    var startKey = HN.stateKey(S.task.start);
    var goalKey = HN.stateKey(S.task.goal);
    var userSubgoalComplete = S.ans2Tuple.every(function (p) { return p !== null && p >= 0; });
    var userSubKey = userSubgoalComplete ? HN.stateKey(S.ans2Tuple) : null;
    var trueSubKey = HN.stateKey(S.task.subgoal);
    var r = L.states.length <= 27 ? 6.5 : 4.6;
    var showNodeTuples = !exam() || S.locked || (S.res.t1 && S.res.t1.ok);

    L.states.forEach(function (st) {
      var sk = HN.stateKey(st), p = L.pos[sk];
      var isCur = sk === curKey, isStart = sk === startKey, isGoal = sk === goalKey;
      var isUserSub = sk === userSubKey;
      var isTrueSub = showBridgeSolution && sk === trueSubKey;

      var grp = el("g", { "class": "hn-gnode" }, svg);
      if (!exam()) {
        grp.style.cursor = "pointer";
        grp.addEventListener("click", function () {
          if (S.locked) return;
          S.ans2Tuple = st.slice();
          delete S.res.t2;
          S.note = "🎯 Wybrano węzeł " + HN.stateLabel(st) + " z grafu jako stan-podcel w Zadaniu 2.";
          render();
        });
      }

      if (isUserSub || isTrueSub) {
        el("circle", {
          cx: p[0], cy: p[1], r: r + 5.2,
          fill: "none", stroke: isTrueSub ? "#2b8a3e" : "#e8590c",
          "stroke-width": 2.4, "stroke-dasharray": isTrueSub ? "none" : "3 2"
        }, grp);
      }

      var fill = isCur ? "#e03131" : (isGoal ? "#ffe066" : (isStart ? "#2f9e44" : (S.visitedSet[sk] ? "#ffd8a8" : "#ffffff")));
      var stroke = isCur ? "#c92a2a" : (isStart ? "#2f9e44" : (isGoal ? "#e67700" : "#868e96"));
      var c = el("circle", {
        cx: p[0], cy: p[1], r: isCur ? r + 2.0 : r,
        fill: fill, stroke: stroke, "stroke-width": isCur || isStart || isGoal ? 2.2 : 1.1
      }, grp);

      var tip = isStart ? "START" : (isGoal ? "CEL" : "Stan");
      if (showNodeTuples) tip += ": " + HN.stateLabel(st);
      el("title", {}, c, tip);
    });
  }

  function buildTupleControls(tupleArr, prefixClass, onSelect) {
    var html = "<div class='hn-tuple-box'><span class='hn-paren'>(</span>";
    for (var i = 0; i < S.n; i++) {
      var col = HN.COLORS[i % HN.COLORS.length];
      var val = tupleArr[i];
      html += "<div class='hn-titem'>" +
        "<span class='hn-tdisk' style='background:" + col + "'>p<sub>" + (i + 1) + "</sub> (#" + (i + 1) + ")</span>" +
        "<div class='hn-tbtns'>";
      for (var p = 0; p < S.k; p++) {
        var active = val === p ? " hn-tbtn-on" : "";
        html += "<button type='button' class='hn-tbtn " + prefixClass + active + "' data-disk='" + i + "' data-peg='" + p + "'" +
          (S.locked ? " disabled" : "") + ">" + HN.PEGS[p] + "</button>";
      }
      html += "</div></div>";
      if (i + 1 < S.n) html += "<span class='hn-comma'>,</span>";
    }
    html += "<span class='hn-paren'>)</span></div>";
    return html;
  }

  function renderTasks() {
    var box = $(".hn-tasks");
    box.innerHTML = "";
    var T = S.task;
    var showSol = function (id) {
      return exam() ? (S.locked && S.res[id] && !S.res[id].ok) : S.reveal;
    };

    // ---- ZADANIE 1: Formalizacja stanu START i dozwolone akcje ----
    var r1 = S.res.t1;
    var card1 = document.createElement("div");
    card1.className = "hn-tcard" + (r1 ? (r1.ok ? " hn-tok" : " hn-tbad") : "");
    card1.style.setProperty("--c", "#1971c2");

    var allActs = HN.allActionLabels(S.k);
    var actButtonsHtml = allActs.map(function (a) {
      var on = S.ans1Actions.indexOf(a.label) >= 0 ? " hn-act-on" : "";
      return "<button type='button' class='hn-act-btn" + on + "' data-act='" + a.label + "'" +
        (S.locked ? " disabled" : "") + ">" + a.label + "</button>";
    }).join("");

    var sol1 = showSol("t1")
      ? "<div class='hn-sol'>Poprawna odpowiedź: <b>s<sub>0</sub> = " + HN.stateLabel(T.start) +
        "</b> &nbsp;·&nbsp; Dozwolone akcje: <b>" + T.validStartActions.join(", ") + "</b></div>"
      : "";

    card1.innerHTML =
      "<div class='hn-thead'><b>1. Formalizacja stanu początkowego START (Lab 2)</b>" +
        (S.locked ? "<span class='hn-tactions'>🔒</span>" : "<span class='hn-tactions'><button type='button' class='hn-clr1' title='Wyczyść zadanie 1'>✕</button></span>") +
      "</div>" +
      "<div class='hn-tdesc'><b>1a.</b> Zapisz stan <b>START</b> (zielona ramka po lewej) jako krotkę <code>(p₁, …, p" + S.n + ")</code>, " +
      "gdzie <code>p₁</code> to słupek najmniejszego krążka <b>#1</b>, a <code>p" + S.n + "</code> – największego <b>#" + S.n + "</b>:</div>" +
      buildTupleControls(S.ans1Tuple, "hn-t1-btn") +
      "<div class='hn-tdesc' style='margin-top:8px'><b>1b.</b> Zaznacz <b>wszystkie</b> akcje (reguły <code>X→Y</code>), które są <b>dozwolone (stosowalne)</b> w stanie <b>START</b>:</div>" +
      "<div class='hn-act-row'>" + actButtonsHtml + "</div>" +
      (r1 ? "<div class='hn-tres'><span class='hn-face'>" + (r1.ok ? "😀" : "😢") + "</span><span>" + r1.msg + "</span></div>" : "") +
      sol1;

    card1.querySelectorAll(".hn-t1-btn").forEach(function (b) {
      b.addEventListener("click", function () {
        if (S.locked) return;
        var d = parseInt(b.getAttribute("data-disk"), 10);
        var p = parseInt(b.getAttribute("data-peg"), 10);
        S.ans1Tuple[d] = p;
        delete S.res.t1;
        S.confirm = false;
        render();
      });
    });
    card1.querySelectorAll(".hn-act-btn").forEach(function (b) {
      b.addEventListener("click", function () {
        if (S.locked) return;
        var act = b.getAttribute("data-act");
        var idx = S.ans1Actions.indexOf(act);
        if (idx >= 0) S.ans1Actions.splice(idx, 1);
        else S.ans1Actions.push(act);
        delete S.res.t1;
        S.confirm = false;
        render();
      });
    });
    var clr1 = card1.querySelector(".hn-clr1");
    if (clr1) clr1.addEventListener("click", function () {
      for (var i = 0; i < S.n; i++) S.ans1Tuple[i] = null;
      S.ans1Actions = [];
      delete S.res.t1;
      S.confirm = false;
      render();
    });
    box.appendChild(card1);

    // ---- ZADANIE 2: Podcel / Most w trojkacie Sierpinskiego (Newell & Simon) ----
    var r2 = S.res.t2;
    var card2 = document.createElement("div");
    card2.className = "hn-tcard" + (r2 ? (r2.ok ? " hn-tok" : " hn-tbad") : "");
    card2.style.setProperty("--c", "#e8590c");

    var sol2 = showSol("t2")
      ? "<div class='hn-sol'>Poprawny stan-podcel (wejście na most): <b>" + HN.stateLabel(T.subgoal) +
        "</b> (mniejsze krążki 1.." + (S.n - 1) + " na słupku pomocniczym <b>" + HN.PEGS[T.auxPeg] +
        "</b>, największy krążek #" + S.n + " na <b>" + HN.PEGS[T.fromPeg] + "</b> gotowy do ruchu <b>" + T.bridgeMoveLabel + "</b>).</div>"
      : "";

    card2.innerHTML =
      "<div class='hn-thead'><b>2. Analiza środków i celów: podcel / most w grafie (Newell i Simon)</b>" +
        (S.locked ? "<span class='hn-tactions'>🔒</span>" : "<span class='hn-tactions'><button type='button' class='hn-clr2' title='Wyczyść zadanie 2'>✕</button></span>") +
      "</div>" +
      "<div class='hn-tdesc'>Aby największy krążek <b>#" + S.n + "</b> mógł przejść ze swojego słupka ze stanu START na docelowy słupek ze stanu CEL, " +
      "musisz najpierw osiągnąć <b>stan-podcel</b> (wejście na „most” między podtrójkątami Sierpińskiego), z którego ten ruch krążkiem <b>#" + S.n + "</b> będzie dozwolony.<br>" +
      "Wskaż ten <b>stan-podcel</b> tuż <i>przed</i> ruchem krążka <b>#" + S.n + "</b>" +
      (!exam() ? " (wybierz słupki poniżej lub kliknij węzeł na grafie):" : ":") + "</div>" +
      buildTupleControls(S.ans2Tuple, "hn-t2-btn") +
      (r2 ? "<div class='hn-tres'><span class='hn-face'>" + (r2.ok ? "😀" : "😢") + "</span><span>" + r2.msg + "</span></div>" : "") +
      sol2;

    card2.querySelectorAll(".hn-t2-btn").forEach(function (b) {
      b.addEventListener("click", function () {
        if (S.locked) return;
        var d = parseInt(b.getAttribute("data-disk"), 10);
        var p = parseInt(b.getAttribute("data-peg"), 10);
        S.ans2Tuple[d] = p;
        delete S.res.t2;
        S.confirm = false;
        render();
      });
    });
    var clr2 = card2.querySelector(".hn-clr2");
    if (clr2) clr2.addEventListener("click", function () {
      for (var i = 0; i < S.n; i++) S.ans2Tuple[i] = null;
      delete S.res.t2;
      S.confirm = false;
      render();
    });
    box.appendChild(card2);

    // ---- ZADANIE 3: Najkrotsza sciezka START -> CEL na slupkach ----
    var r3 = S.res.t3;
    var card3 = document.createElement("div");
    card3.className = "hn-tcard" + (r3 ? (r3.ok ? " hn-tok" : " hn-tbad") : "");
    card3.style.setProperty("--c", "#2f9e44");

    var atGoal = HN.stateEq(S.state, T.goal);
    var statusBadge = atGoal
      ? "<span class='hn-badge hn-badge-ok'>🟡 Ułożono CEL (" + HN.nMoves(S.history.length) + ")</span>"
      : "<span class='hn-badge hn-badge-prog'>W trakcie (" + HN.nMoves(S.history.length) + ")</span>";

    var movesChips = S.history.length
      ? S.history.map(function (m, idx) {
          var col = HN.COLORS[(m.disk - 1) % HN.COLORS.length];
          return "<span class='hn-chip' style='border-color:" + col + "'>" + (idx + 1) + ". " + m.label + " <small>(#" + m.disk + ")</small></span>";
        }).join(" ")
      : "<span class='hn-ph'>klikaj słupki na interaktywnej planszy po lewej, aby przekładać krążki…</span>";

    var sol3 = showSol("t3")
      ? "<div class='hn-sol'>Najkrótsza ścieżka (BFS – <b>" + HN.nMoves(T.bfs.moves.length) + "</b>): <b>" +
        T.bfs.moves.map(function (m) { return m.label + " (#" + m.disk + ")"; }).join(" → ") + "</b></div>"
      : "";

    card3.innerHTML =
      "<div class='hn-thead'><b>3. Najkrótsza ścieżka START → CEL (Lab 3 & 4)</b>" + statusBadge +
        (S.locked ? "<span class='hn-tactions'>🔒</span>" :
         "<span class='hn-tactions'><button type='button' class='hn-undo3' title='Cofnij ostatni ruch'>⌫</button><button type='button' class='hn-clr3' title='Resetuj wieżę do START'>✕</button></span>") +
      "</div>" +
      "<div class='hn-tdesc'>Przełóż krążki na interaktywnej planszy po lewej ze stanu <b>START</b> do stanu <b>CEL</b> w <b>minimalnej możliwej liczbie ruchów</b> (takiej, jaką znajduje algorytm <b>BFS</b>)." +
      (!exam() ? " Najkrótsza droga liczy <b>" + HN.nMoves(T.bfs.moves.length) + "</b>." : "") + "</div>" +
      "<div class='hn-seq-box'>" + movesChips + "</div>" +
      (r3 ? "<div class='hn-tres'><span class='hn-face'>" + (r3.ok ? "😀" : "😢") + "</span><span>" + r3.msg + "</span></div>" : "") +
      sol3;

    var u3 = card3.querySelector(".hn-undo3");
    if (u3) u3.addEventListener("click", undoMove);
    var c3 = card3.querySelector(".hn-clr3");
    if (c3) c3.addEventListener("click", function () { resetTowersToStart(false); });
    box.appendChild(card3);
  }

  function renderComparison() {
    var box = $(".hn-compare");
    var canShow = S.showCompare || S.reveal || (exam() && S.locked);
    if (!canShow) {
      box.style.display = "none";
      return;
    }
    box.style.display = "";
    var T = S.task;
    var totalStates = Math.pow(S.k, S.n);
    var solvedByUser = HN.stateEq(S.state, T.goal);
    var userMoves = S.history.length;
    var userUnique = Object.keys(S.visitedSet).length;
    var bfsMoves = T.bfs.moves.length;
    var dfsMoves = T.dfs.moves.length;

    var badge = "";
    if (solvedByUser && userMoves === bfsMoves) {
      badge = "<div class='hn-verdict hn-v-ok'>🏆 <b>Optymalna ścieżka!</b> Twoja ścieżka ma <b>" + HN.nMoves(userMoves) +
        "</b> – dokładnie tyle samo co najkrótsza ścieżka <b>BFS</b> (podczas gdy <b>DFS</b> błądził po grafie przez <b>" +
        HN.nMoves(dfsMoves) + "</b>).</div>";
    } else if (solvedByUser) {
      badge = "<div class='hn-verdict hn-v-info'>📊 Ułożyłeś(aś) stan CEL w <b>" + HN.nMoves(userMoves) +
        "</b>. Najkrótsza ścieżka (<b>BFS</b>) ma <b>" + HN.nMoves(bfsMoves) + "</b>, natomiast <b>DFS</b> potrzebował aż <b>" +
        HN.nMoves(dfsMoves) + "</b>.</div>";
    } else {
      badge = "<div class='hn-verdict hn-v-info'>📊 <b>Porównanie algorytmów BFS i DFS dla zadania START " +
        HN.stateLabel(T.start) + " → CEL " + HN.stateLabel(T.goal) + ":</b> Najkrótsza droga (BFS) przechodzi przez most <b>" +
        HN.stateLabel(T.subgoal) + " → " + HN.stateLabel(T.bridgeOut) + "</b> i liczy <b>" + HN.nMoves(bfsMoves) + "</b>.</div>";
    }

    function seqHtml(moves, cap) {
      if (!moves || !moves.length) return "<span style='color:#868e96'>—</span>";
      var shown = moves.slice(0, cap || 35).map(function (m) {
        return "<span class='hn-chip' style='border-color:" + HN.COLORS[(m.disk - 1) % HN.COLORS.length] + "'>" +
          m.label + " <small>(#" + m.disk + ")</small></span>";
      }).join(" → ");
      if (moves.length > (cap || 35)) shown += " … <i>(+" + (moves.length - (cap || 35)) + ")</i>";
      return shown;
    }

    box.innerHTML =
      "<h5>📊 Porównanie: Twoja ścieżka vs BFS (najkrótsza ścieżka) vs DFS</h5>" +
      badge +
      "<table class='hn-tbl'>" +
      "<tr><th>Podejście</th><th>Długość ścieżki (ruchy)</th><th>Odwiedzone stany (z " + totalStates + ")</th><th>Różnica względem optimum (BFS)</th><th>Animacja na słupkach</th></tr>" +
      "<tr class='hn-row-user'>" +
        "<td><b>🧑 Twoja ścieżka</b></td>" +
        "<td><b>" + (solvedByUser ? HN.nMoves(userMoves) : HN.nMoves(userMoves) + " (nieukończone)") + "</b></td>" +
        "<td>" + HN.nStates(userUnique) + "</td>" +
        "<td>" + (solvedByUser ? (userMoves === bfsMoves ? "<b>0 (Optymalnie! 🏆)</b>" : "+" + HN.nMoves(userMoves - bfsMoves)) : "—") + "</td>" +
        "<td><button type='button' class='hn-play-user'" + (S.history.length ? "" : " disabled") + ">▶ Odtwórz moje</button></td>" +
      "</tr>" +
      "<tr class='hn-row-bfs'>" +
        "<td><b>🌊 BFS (wszerz – Kolejka FIFO)</b></td>" +
        "<td><b>" + HN.nMoves(bfsMoves) + "</b></td>" +
        "<td>" + HN.nStates(T.bfs.visitedCount) + "</td>" +
        "<td><b>0 (Najkrótsza ścieżka)</b></td>" +
        "<td><button type='button' class='hn-play-bfs'>▶ Pokaż BFS</button></td>" +
      "</tr>" +
      "<tr class='hn-row-dfs'>" +
        "<td><b>🕳️ DFS (w głąb – Stos LIFO)</b></td>" +
        "<td><b>" + HN.nMoves(dfsMoves) + "</b></td>" +
        "<td>" + HN.nStates(T.dfs.visitedCount) + "</td>" +
        "<td>+" + HN.nMoves(dfsMoves - bfsMoves) + " (" + (dfsMoves / Math.max(1, bfsMoves)).toFixed(1) + "× dłużej)</td>" +
        "<td><button type='button' class='hn-play-dfs'>▶ Pokaż DFS</button></td>" +
      "</tr>" +
      "</table>" +
      "<div class='hn-paths-detail'>" +
        "<div><b>🌊 Najkrótsza ścieżka (BFS – " + HN.nMoves(bfsMoves) + "):</b><div class='hn-seq'>" + seqHtml(T.bfs.moves, 32) + "</div></div>" +
        "<div style='margin-top:6px'><b>🕳️ Ścieżka znaleziona przez DFS (" + HN.nMoves(dfsMoves) + "):</b><div class='hn-seq'>" + seqHtml(T.dfs.moves, 32) + "</div></div>" +
      "</div>";

    var btnU = box.querySelector(".hn-play-user");
    if (btnU) btnU.addEventListener("click", function () { playSequence(S.trailStates, S.history, "Twoja ścieżka", "user"); });
    var btnB = box.querySelector(".hn-play-bfs");
    if (btnB) btnB.addEventListener("click", function () { playSequence(T.bfs.states, T.bfs.moves, "Najkrótsza ścieżka – BFS", "bfs"); });
    var btnD = box.querySelector(".hn-play-dfs");
    if (btnD) btnD.addEventListener("click", function () { playSequence(T.dfs.states, T.dfs.moves, "Ścieżka DFS", "dfs"); });
  }

  function checkAll() {
    var T = S.task;
    if (exam()) {
      if (S.locked) return;
      var missing = [];
      if (S.ans1Tuple.some(function (p) { return p === null || p < 0; }) || !S.ans1Actions.length) missing.push("Zadanie 1 (START)");
      if (S.ans2Tuple.some(function (p) { return p === null || p < 0; })) missing.push("Zadanie 2 (Podcel)");
      if (!HN.stateEq(S.state, T.goal)) missing.push("Zadanie 3 (Wieża nie jest w stanie CEL)");
      if (missing.length && !S.confirm) {
        S.confirm = true;
        S.note = "⚠️ Niekompletne odpowiedzi: " + missing.join(", ") + ". Po zatwierdzeniu w trybie egzaminu nie będzie można ich poprawić – kliknij jeszcze raz, aby zatwierdzić mimo to.";
        render();
        return;
      }
      S.locked = true;
      S.confirm = false;
      S.showCompare = true;
      S.overlay = "all";
    }
    S.res = {
      t1: HN.checkTask1(T, S.ans1Tuple, S.ans1Actions),
      t2: HN.checkTask2(T, S.ans2Tuple),
      t3: HN.checkTask3(T, S.state, S.history)
    };
    S.note = "";
    render();
  }

  function render() {
    var T = S.task;
    root.querySelectorAll(".hn-mode button").forEach(function (b) {
      b.className = b.getAttribute("data-mode") === S.mode ? "hn-on" : "";
    });
    $(".hn-preset-wrap").style.display = exam() ? "none" : "";
    $(".hn-rule").innerHTML = RULES[S.mode];

    var showStartTuple = !exam() || S.locked || (S.res.t1 && S.res.t1.ok) || S.reveal;
    $(".hn-start-lbl").innerHTML = "🟢 Stan początkowy (START)" + (showStartTuple ? ": <code>" + HN.stateLabel(T.start) + "</code>" : "");
    $(".hn-goal-lbl").innerHTML = "🟡 Stan docelowy (CEL)" + (showStartTuple ? ": <code>" + HN.stateLabel(T.goal) + "</code>" : "");

    drawMiniState($(".hn-start-svg"), T.start, "#2f9e44");
    drawMiniState($(".hn-goal-svg"), T.goal, "#b08900");

    var totalStates = Math.pow(S.k, S.n);
    var curState = S.animState || S.state;
    $(".hn-stats").innerHTML =
      "Przestrzeń stanów: <b>|S| = " + S.k + "<sup>" + S.n + "</sup> = " + totalStates + "</b> &nbsp;·&nbsp; " +
      (showStartTuple ? "Bieżący stan wieży: <code>" + HN.stateLabel(curState) + "</code> &nbsp;·&nbsp; " : "") +
      "Twoje ruchy: <b>" + S.history.length + "</b>" +
      (!exam() || S.locked ? " &nbsp;·&nbsp; Najkrótsza ścieżka (BFS): <b>" + HN.nMoves(T.bfs.moves.length) + "</b>" : "");

    var msgBox = $(".hn-msg");
    msgBox.className = "hn-msg hn-msg-" + S.msgKind;
    msgBox.innerHTML = S.msg;

    $(".hn-undo").disabled = S.history.length === 0 || (S.locked && !S.showCompare);
    $(".hn-reset").disabled = S.locked && !S.showCompare;

    var canSeeBfsDfs = !exam() || S.locked;
    root.querySelectorAll(".hn-ov-btn").forEach(function (b) {
      var ov = b.getAttribute("data-ov");
      b.disabled = !canSeeBfsDfs && ov !== "user";
      b.className = "hn-ov-btn" + (ov === S.overlay ? " hn-ov-on" : "");
    });

    drawTowers();
    drawStateGraph();
    renderTasks();
    renderComparison();

    $(".hn-note").textContent = S.note;

    var ids = ["t1", "t2", "t3"];
    var nOk = ids.filter(function (id) { return S.res[id] && S.res[id].ok; }).length;
    var anyChecked = ids.some(function (id) { return S.res[id]; });
    var allOk = nOk === 3;
    var praise = "🎉 <b>Wszystkie 3 zadania rozwiązane bezbłędnie!</b> Zauważ, jak formalna reprezentacja krotki (Lab 2), " +
      "wyznaczenie podcelu/mostu na trójkącie Sierpińskiego (Newell i Simon) oraz przeszukiwanie BFS (Lab 3) łączą się w jedną całość.";

    $(".hn-summary").innerHTML = exam()
      ? (S.locked ? "📊 Wynik egzaminu: <b>" + nOk + " / 3</b>. " + (allOk ? praise : "") +
                    "<br>Chcesz spróbować na nowym układzie? Kliknij <b>🎲 Nowe zadanie</b>." : "")
      : (allOk ? praise : (anyChecked ? "Popraw zadania oznaczone na czerwono i kliknij <b>✔ Sprawdź</b> ponownie." : ""));

    var cb = $(".hn-check");
    cb.textContent = !exam() ? "✔ Sprawdź" : S.locked ? "🔒 Zatwierdzone"
                   : S.confirm ? "🔒 Tak, zatwierdź mimo to" : "🔒 Zatwierdź odpowiedzi";
    cb.disabled = exam() && S.locked;

    $(".hn-show").style.display = exam() ? "none" : "";
    $(".hn-toggle-cmp").style.display = exam() ? "none" : "";
    $(".hn-toggle-cmp").textContent = S.showCompare ? "🙈 Ukryj porównanie BFS / DFS" : "📊 Porównaj z BFS i DFS";
  }

  $(".hn-disks").value = String(CFG.n_disks);
  $(".hn-disks").addEventListener("change", function () { newTask(S.seed); });
  $(".hn-preset").addEventListener("change", function () { newTask(S.seed); });
  $(".hn-seed").addEventListener("change", function () { newTask(parseInt(this.value, 10) || 0); });
  $(".hn-new").addEventListener("click", function () { newTask(randomSeed()); });
  $(".hn-undo").addEventListener("click", undoMove);
  $(".hn-reset").addEventListener("click", function () { resetTowersToStart(false); });
  $(".hn-check").addEventListener("click", checkAll);
  $(".hn-show").addEventListener("click", function () {
    S.reveal = !S.reveal;
    if (S.reveal) S. overlay = "all";
    this.textContent = S.reveal ? "🙈 Ukryj rozwiązanie" : "👁 Pokaż rozwiązanie";
    render();
  });
  $(".hn-toggle-cmp").addEventListener("click", function () {
    S.showCompare = !S.showCompare;
    if (S.showCompare && S.overlay === "user") S.overlay = "all";
    render();
  });
  root.querySelectorAll(".hn-mode button").forEach(function (b) {
    b.addEventListener("click", function () { setMode(b.getAttribute("data-mode")); });
  });
  root.querySelectorAll(".hn-ov-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      if (b.disabled) return;
      S.overlay = b.getAttribute("data-ov");
      render();
    });
  });

  newTask(CFG.seed);
})();
"""

_HANOI_CSS = r"""
<style>
.hn-card{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#212529;background:#ffffff;
  border:1px solid #ced4da;border-radius:10px;padding:14px 16px;max-width:1120px;font-size:13.5px;line-height:1.45}
.hn-card *{box-sizing:border-box}
.hn-card h4{margin:0;font-size:17px;color:#212529}
.hn-card h5{margin:0 0 8px 0;font-size:15px;color:#212529}
.hn-top{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between;margin-bottom:8px}
.hn-mode{display:inline-flex;border:1px solid #adb5bd;border-radius:8px;overflow:hidden}
.hn-card .hn-mode button{border:none;border-radius:0;background:#f8f9fa;padding:5px 12px;font-size:13.5px}
.hn-card .hn-mode button + button{border-left:1px solid #adb5bd}
.hn-card .hn-mode button.hn-on{background:#343a40;color:#fff;font-weight:700}
.hn-controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:6px}
.hn-controls label{display:flex;gap:6px;align-items:center;font-weight:600;color:#212529}
.hn-card select,.hn-card input,.hn-card button{font-size:13.5px;padding:4px 10px;border:1px solid #adb5bd;border-radius:6px;
  background:#f8f9fa;color:#212529;cursor:pointer}
.hn-card select,.hn-card input{background:#fff}
.hn-card input.hn-seed{width:90px}
.hn-card button:hover:not(:disabled){background:#e9ecef}
.hn-card button:disabled{opacity:0.5;cursor:default}
.hn-rule{background:#f1f3f5;border-radius:6px;padding:7px 11px;margin:6px 0;color:#343a40;font-size:13px}
.hn-stats{background:#f8f9fa;border:1px solid #e9ecef;border-radius:6px;padding:5px 10px;margin:6px 0;font-size:13px;color:#212529}
.hn-msg{border-radius:6px;padding:6px 10px;margin:6px 0;font-size:13px}
.hn-msg-info{background:#e7f5ff;border-left:4px solid #1971c2;color:#212529}
.hn-msg-ok{background:#d3f9d8;border-left:4px solid #2b8a3e;color:#212529}
.hn-msg-warn{background:#fff3bf;border-left:4px solid #f08c00;color:#212529}
.hn-msg-bad{background:#ffe3e3;border-left:4px solid #e03131;color:#212529}
.hn-body{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;margin-top:8px}
.hn-left-col{flex:1.15 1 460px;min-width:310px;display:flex;flex-direction:column;gap:8px}
.hn-previews{display:flex;gap:8px;flex-wrap:wrap}
.hn-prev-card{flex:1 1 200px;border:1px solid #dee2e6;border-radius:8px;padding:5px 8px;background:#fff}
.hn-prev-start{border-top:3px solid #2f9e44;background:#f8fff9}
.hn-prev-goal{border-top:3px solid #f08c00;background:#fffdf5}
.hn-prev-title{font-size:12px;font-weight:700;margin-bottom:2px}
.hn-prev-svg{width:100%;height:auto;display:block}
.hn-towers-box{border:1px solid #dee2e6;border-radius:8px;background:#ffffff;padding:6px 8px}
.hn-towers-head{display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px;font-size:12.5px}
.hn-towers-svg{width:100%;height:auto;display:block;user-select:none}
.hn-peg-grp{cursor:pointer}
.hn-graph-box{border:1px solid #dee2e6;border-radius:8px;background:#f8f9fa;padding:8px}
.hn-graph-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:4px;font-size:12px}
.hn-ov-bar{display:inline-flex;border:1px solid #adb5bd;border-radius:6px;overflow:hidden}
.hn-card .hn-ov-bar button{padding:2px 7px;font-size:11.5px;border:none;border-radius:0;background:#fff}
.hn-card .hn-ov-bar button + button{border-left:1px solid #adb5bd}
.hn-card .hn-ov-bar button.hn-ov-on{background:#343a40;color:#fff;font-weight:700}
.hn-graph-svg{width:100%;height:auto;display:block;max-height:240px}
.hn-legend{display:flex;flex-wrap:wrap;gap:8px;font-size:11.5px;color:#495057;justify-content:center;margin-top:4px}
.hn-tasks{flex:1 1 430px;min-width:300px;display:flex;flex-direction:column;gap:10px}
.hn-tcard{border:2px solid #dee2e6;border-left:6px solid var(--c);border-radius:8px;padding:8px 10px;background:#fff}
.hn-tcard.hn-tok{border-color:#2f9e44;background:#ebfbee}
.hn-tcard.hn-tbad{border-color:#e03131;background:#fff5f5}
.hn-thead{display:flex;align-items:center;gap:6px;flex-wrap:wrap;color:#212529;margin-bottom:4px}
.hn-tactions{margin-left:auto;display:flex;gap:4px}
.hn-card .hn-tactions button{padding:1px 7px;font-size:12.5px}
.hn-tdesc{font-size:12.5px;color:#343a40;margin-bottom:6px}
.hn-tuple-box{display:flex;flex-wrap:wrap;align-items:center;gap:6px;background:#f8f9fa;padding:6px 8px;border-radius:6px;border:1px solid #e9ecef}
.hn-paren{font-size:18px;font-weight:700;color:#495057}
.hn-comma{font-size:16px;font-weight:700;color:#495057}
.hn-titem{display:flex;align-items:center;gap:4px}
.hn-tdisk{color:#fff;font-size:11.5px;font-weight:700;padding:2px 6px;border-radius:5px}
.hn-tbtns{display:inline-flex;border:1px solid #adb5bd;border-radius:5px;overflow:hidden}
.hn-card .hn-tbtns button{border:none;border-radius:0;padding:2px 8px;font-size:12.5px;background:#fff;font-weight:600}
.hn-card .hn-tbtns button + button{border-left:1px solid #adb5bd}
.hn-card .hn-tbtns button.hn-tbtn-on{background:#212529;color:#fff}
.hn-act-row{display:flex;flex-wrap:wrap;gap:6px}
.hn-card button.hn-act-btn{padding:3px 10px;font-size:12.5px;font-weight:600;border-radius:14px;background:#fff}
.hn-card button.hn-act-btn.hn-act-on{background:#1971c2;border-color:#1864ab;color:#fff}
.hn-seq-box{background:#f8f9fa;border:1px solid #e9ecef;border-radius:6px;padding:6px 8px;min-height:32px;display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.hn-ph{color:#868e96;font-style:italic;font-size:12.5px}
.hn-badge{font-size:11.5px;font-weight:700;padding:1px 8px;border-radius:10px}
.hn-badge-ok{background:#fff3bf;color:#b08900;border:1px solid #fcc419}
.hn-badge-prog{background:#e9ecef;color:#495057}
.hn-tres{margin-top:6px;display:flex;align-items:center;gap:8px;color:#212529;font-size:13px}
.hn-face{font-size:24px;line-height:1;flex:none}
.hn-sol{margin-top:5px;padding-top:4px;border-top:1px dashed #ced4da;color:#495057;font-size:12.5px}
.hn-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}
.hn-card button.hn-check{background:#2f9e44;border-color:#2b8a3e;color:#fff;font-weight:700;padding:6px 14px}
.hn-card button.hn-check:hover:not(:disabled){background:#2b8a3e}
.hn-card button.hn-toggle-cmp{background:#1971c2;border-color:#1864ab;color:#fff;font-weight:600}
.hn-card button.hn-toggle-cmp:hover:not(:disabled){background:#1864ab}
.hn-note{color:#e8590c;min-height:18px;font-size:13px;margin-top:6px}
.hn-summary{margin-top:6px;color:#212529;min-height:20px;font-size:13.5px}
.hn-chip{display:inline-block;padding:1px 7px;border-radius:10px;border:2px solid #adb5bd;background:#fff;font-size:12px;font-weight:600}
.hn-compare{margin-top:12px;padding:12px;border:2px solid #1971c2;border-radius:8px;background:#f8f9fa}
.hn-verdict{padding:8px 10px;border-radius:6px;margin-bottom:8px;font-size:13px}
.hn-v-ok{background:#d3f9d8;border-left:4px solid #2b8a3e}
.hn-v-info{background:#e7f5ff;border-left:4px solid #1971c2}
.hn-tbl{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff;margin-bottom:8px}
.hn-tbl th,.hn-tbl td{border:1px solid #ced4da;padding:5px 8px;text-align:left;color:#212529}
.hn-tbl th{background:#e9ecef;font-weight:700}
.hn-row-user td{background:#fff9db}
.hn-row-bfs td{background:#e7f5ff}
.hn-row-dfs td{background:#f3d9fa}
.hn-paths-detail{font-size:12px;color:#212529;background:#fff;border:1px solid #dee2e6;border-radius:6px;padding:8px}
.hn-seq{margin-top:4px;display:flex;flex-wrap:wrap;gap:4px;align-items:center;line-height:1.5}
</style>
"""

_HANOI_HTML = r"""
<div class="hn-card" id="__ROOT__">
  <div class="hn-top">
    <h4>🗼 Ćwiczenie: Wieże Hanoi – przestrzeń stanów, podcele i najkrótsza ścieżka</h4>
    <div class="hn-mode">
      <button type="button" data-mode="exam">📝 Egzamin</button><button type="button" data-mode="playground">🔓 Plac zabaw</button>
    </div>
  </div>
  <div class="hn-controls">
    <label>Krążków (n): <select class="hn-disks">
      <option value="2">2 krążki</option>
      <option value="3">3 krążki</option>
      <option value="4">4 krążki</option>
    </select></label>
    <label class="hn-preset-wrap">Układ: <select class="hn-preset">
      <option value="random">🎲 Losowy (START → CEL)</option>
      <option value="classic">🗼 Klasyczny (A → C)</option>
    </select></label>
    <label>Numer zadania: <input class="hn-seed" type="number" min="0" step="1"></label>
    <button type="button" class="hn-new">🎲 Nowe zadanie</button>
  </div>
  <div class="hn-rule"></div>
  <div class="hn-stats"></div>
  <div class="hn-body">
    <div class="hn-left-col">
      <div class="hn-previews">
        <div class="hn-prev-card hn-prev-start">
          <div class="hn-prev-title hn-start-lbl">🟢 Stan początkowy (START)</div>
          <svg class="hn-prev-svg hn-start-svg" viewBox="0 0 230 112" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
        <div class="hn-prev-card hn-prev-goal">
          <div class="hn-prev-title hn-goal-lbl">🟡 Stan docelowy (CEL)</div>
          <svg class="hn-prev-svg hn-goal-svg" viewBox="0 0 230 112" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
      </div>
      <div class="hn-towers-box">
        <div class="hn-towers-head">
          <b>Interaktywne słupki (przekładaj krążki do Zadania 3):</b>
          <div>
            <button type="button" class="hn-undo">↶ Cofnij ruch</button>
            <button type="button" class="hn-reset">⟲ Wróć do START</button>
          </div>
        </div>
        <div class="hn-msg"></div>
        <svg class="hn-towers-svg" viewBox="0 0 520 240" xmlns="http://www.w3.org/2000/svg"></svg>
      </div>
      <div class="hn-graph-box">
        <div class="hn-graph-head">
          <b>Mapa przestrzeni stanów (trójkąt Sierpińskiego):</b>
          <div class="hn-ov-bar">
            <button type="button" class="hn-ov-btn" data-ov="user">Twoja</button>
            <button type="button" class="hn-ov-btn" data-ov="bfs">BFS</button>
            <button type="button" class="hn-ov-btn" data-ov="dfs">DFS</button>
            <button type="button" class="hn-ov-btn" data-ov="all">Wszystkie</button>
          </div>
        </div>
        <svg class="hn-graph-svg" viewBox="0 0 380 300" xmlns="http://www.w3.org/2000/svg"></svg>
        <div class="hn-legend">
          <span>🟢 Start</span>
          <span>🟡 Cel</span>
          <span>🔴 Bieżący</span>
          <span>🎯 Podcel (Zad. 2)</span>
          <span style="color:#f08c00;font-weight:700">▬ Twoja</span>
          <span style="color:#1971c2;font-weight:700">▬ BFS</span>
          <span style="color:#9c36b5;font-weight:700">▬ DFS</span>
        </div>
      </div>
    </div>
    <div class="hn-tasks"></div>
  </div>
  <div class="hn-note"></div>
  <div class="hn-actions">
    <button type="button" class="hn-check">🔒 Zatwierdź odpowiedzi</button>
    <button type="button" class="hn-show">👁 Pokaż rozwiązanie</button>
    <button type="button" class="hn-toggle-cmp">📊 Porównaj z BFS i DFS</button>
  </div>
  <div class="hn-summary"></div>
  <div class="hn-compare" style="display:none"></div>
</div>
"""


def build_hanoi_html(
    n_disks: int = 3,
    n_pegs: int = 3,
    seed: int | None = None,
    mode: str = "exam",
) -> str:
    """Buduje samodzielny 3-czesciowy widget HTML/SVG/JS z cwiczeniem Wiez Hanoi."""
    import random

    if not 2 <= n_disks <= 4:
        raise ValueError("n_disks w cwiczeniu musi byc w przedziale 2..4")
    if not 2 <= n_pegs <= 5:
        raise ValueError("n_pegs musi byc w przedziale 2..5")
    if mode not in HANOI_MODES:
        raise ValueError(f"mode musi byc jednym z {HANOI_MODES}")
    if seed is None:
        seed = random.randrange(100000)
    root = "hn-" + uuid.uuid4().hex[:10]
    cfg = json.dumps({"n_disks": int(n_disks), "n_pegs": int(n_pegs), "seed": int(seed), "mode": mode})
    ui = _HANOI_UI_JS.replace("__ROOT__", root).replace("__CFG__", cfg)
    return (
        _HANOI_CSS
        + _HANOI_HTML.replace("__ROOT__", root)
        + "<script>(function(){\n"
        + _HANOI_LOGIC_JS
        + "\n"
        + ui
        + "\n})();</script>"
    )


def hanoi_exercise(
    n_disks: int = 3,
    n_pegs: int = 3,
    seed: int | None = None,
    mode: str = "exam",
):
    """Wyswietla 3-czesciowe interaktywne cwiczenie Wieze Hanoi (tryb 'exam' lub 'playground').
    Podaj seed (np. 2026), zeby cala grupa dostala to samo zadanie START -> CEL."""
    from IPython.display import HTML, display
    display(HTML(build_hanoi_html(n_disks=n_disks, n_pegs=n_pegs, seed=seed, mode=mode)))
