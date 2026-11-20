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
# Interaktywne cwiczenie w przegladarce (HTML + SVG + JS, bez ipywidgets)
# Student sam rozwiazuje Wieze Hanoi (min. 2 krazki, min. 2 slupki)
# i porownuje swoja liczbe ruchow z BFS (najkrotsza sciezka) oraz DFS!
# --------------------------------------------------------------------------
_HANOI_LOGIC_JS = r"""
var HN = (function () {
  var PEGS = ["A", "B", "C", "D", "E"];
  var COLORS = ["#e03131", "#f08c00", "#2f9e44", "#1971c2", "#9c36b5", "#0ca678"];

  function stateKey(s) { return s.join(","); }
  function stateLabel(s) { return "(" + s.map(function (p) { return PEGS[p]; }).join(", ") + ")"; }

  function topDisks(s, k) {
    var top = [];
    for (var p = 0; p < k; p++) top.push(-1);
    for (var d = s.length - 1; d >= 0; d--) top[s[d]] = d;
    return top;
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

  function isGoal(s, k) {
    for (var i = 0; i < s.length; i++) if (s[i] !== k - 1) return false;
    return true;
  }

  // Rozwiazanie BFS (kolejka FIFO -> gwarantuje najkrotsza sciezke)
  function runBFS(n, k) {
    var start = [], i;
    for (i = 0; i < n; i++) start.push(0);
    var sk = stateKey(start);
    var q = [start], seen = {}, par = {}, order = [];
    seen[sk] = true;
    par[sk] = null;
    var goalKey = null;

    while (q.length) {
      var u = q.shift(), uk = stateKey(u);
      order.push(u);
      if (isGoal(u, k)) { goalKey = uk; break; }
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

  // Rozwiazanie DFS (stos LIFO / rekurencja -> sasiad w porzadku leksykograficznym par slupkow)
  function runDFS(n, k) {
    var start = [], i;
    for (i = 0; i < n; i++) start.push(0);
    var seen = {}, par = {}, order = [];
    var stack = [{ state: start, prevKey: null, move: null }];
    var goalKey = null;

    while (stack.length) {
      var item = stack.pop(), u = item.state, uk = stateKey(u);
      if (seen[uk]) continue;
      seen[uk] = true;
      order.push(u);
      par[uk] = item.prevKey === null ? null : item;
      if (isGoal(u, k)) { goalKey = uk; break; }
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

  // Generowanie wspolrzednych grafu przestrzeni stanow (dla wizualizacji SVG do 256 stanow)
  function buildGraphLayout(n, k) {
    var total = Math.pow(k, n);
    if (total > 256) return null;
    var states = [];
    (function gen(prefix) {
      if (prefix.length === n) { states.push(prefix); return; }
      for (var p = 0; p < k; p++) gen(prefix.concat([p]));
    })([]);

    var rawPos = {}, i, d;
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
    var W = 380, H = 310, pad = 26;
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
    PEGS: PEGS, COLORS: COLORS, stateKey: stateKey, stateLabel: stateLabel,
    topDisks: topDisks, validMoves: validMoves, isGoal: isGoal,
    runBFS: runBFS, runDFS: runDFS, buildGraphLayout: buildGraphLayout,
    nMoves: nMoves, nStates: nStates, plural: plural
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

  var S = {
    n: CFG.n_disks,
    k: CFG.n_pegs,
    state: [],
    selectedPeg: null,
    history: [],        // [{from, to, disk, next, label}]
    trailStates: [],    // [state0, state1, ...]
    visitedSet: {},
    bfs: null,
    dfs: null,
    layout: null,
    showCompare: false,
    overlay: "user",    // "user" | "bfs" | "dfs" | "all"
    msg: "",
    msgKind: "info",
    animTimer: null
  };

  function stopAnim() {
    if (S.animTimer) { clearInterval(S.animTimer); S.animTimer = null; }
  }

  function initGame() {
    stopAnim();
    S.n = parseInt($(".hn-disks").value, 10);
    S.k = parseInt($(".hn-pegs").value, 10);
    S.state = [];
    for (var i = 0; i < S.n; i++) S.state.push(0);
    S.selectedPeg = null;
    S.history = [];
    S.trailStates = [S.state.slice()];
    S.visitedSet = {};
    S.visitedSet[HN.stateKey(S.state)] = true;
    S.bfs = HN.runBFS(S.n, S.k);
    S.dfs = HN.runDFS(S.n, S.k);
    S.layout = HN.buildGraphLayout(S.n, S.k);
    S.showCompare = false;
    S.overlay = "user";
    S.msg = "Kliknij słupek, z którego chcesz podnieść górny krążek, a następnie słupek docelowy.";
    S.msgKind = "info";
    render();
  }

  function clickPeg(p) {
    stopAnim();
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

    // Wykonaj ruch u -> p
    var next = S.state.slice();
    next[du] = p;
    var mv = { from: u, to: p, disk: du + 1, next: next, label: HN.PEGS[u] + "→" + HN.PEGS[p] };
    S.state = next;
    S.selectedPeg = null;
    S.history.push(mv);
    S.trailStates.push(next.slice());
    S.visitedSet[HN.stateKey(next)] = true;

    if (HN.isGoal(S.state, S.k)) {
      S.showCompare = true;
      S.overlay = "all";
      var opt = S.bfs.moves.length;
      var diff = S.history.length - opt;
      if (diff === 0) {
        S.msg = "🏆 <b>Brawo! Rozwiązano w minimalnej możliwej liczbie ruchów (" + HN.nMoves(opt) + ")!</b> Zobacz poniżej porównanie z BFS i DFS.";
      } else {
        S.msg = "🎉 <b>Cel osiągnięty w " + HN.nMoves(S.history.length) + "!</b> Najkrótsza ścieżka (BFS) liczy <b>" + HN.nMoves(opt) + "</b> (o " + HN.nMoves(diff) + " mniej). Zobacz porównanie poniżej!";
      }
      S.msgKind = "ok";
    } else if (S.k === 2) {
      S.showCompare = true;
      S.msg = "🤔 Przełożono krążek #1 na słupek B, ale większego krążka #2 nie da się ruszyć! Przy <b>2 słupkach</b> stan docelowy jest <b>nieosiągalny</b> (zobacz analizę BFS/DFS poniżej).";
      S.msgKind = "warn";
    } else {
      S.msg = "Wykonano ruch: krążek <b>#" + (du + 1) + "</b>: <b>" + mv.label + "</b> (stan: <code>" + HN.stateLabel(S.state) + "</code>).";
      S.msgKind = "info";
    }
    render();
  }

  function undoMove() {
    stopAnim();
    if (!S.history.length) return;
    S.history.pop();
    S.trailStates.pop();
    S.state = S.trailStates[S.trailStates.length - 1].slice();
    S.selectedPeg = null;
    S.msg = "Cofnięto ostatni ruch. Bieżący stan: <code>" + HN.stateLabel(S.state) + "</code>.";
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
    S.state = states[0].slice();
    S.msg = "▶️ Odtwarzanie (" + label + "): start <code>" + HN.stateLabel(S.state) + "</code> (0 / " + moves.length + ")";
    S.msgKind = "info";
    render();

    S.animTimer = setInterval(function () {
      if (!document.body.contains(root)) { stopAnim(); return; }
      idx++;
      if (idx >= states.length) {
        stopAnim();
        S.msg = "✅ Zakończono odtwarzanie: <b>" + label + "</b> (" + HN.nMoves(moves.length) + ").";
        S.msgKind = "ok";
        render();
        return;
      }
      S.state = states[idx].slice();
      var mv = moves[idx - 1];
      S.msg = "▶️ Odtwarzanie (" + label + ") – krok " + idx + " / " + moves.length + ": krążek <b>#" + mv.disk + "</b> (" + mv.label + ")";
      render();
    }, moves.length > 25 ? 240 : 520);
  }

  function drawTowers() {
    var svg = $(".hn-towers-svg");
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = 540, H = 290;
    var baseY = 238, pegTopY = 58;
    var n = S.n, k = S.k;
    var spacing = (W - 60) / k;
    var top = HN.topDisks(S.state, k);

    // Podstawa
    el("rect", { x: 20, y: baseY, width: W - 40, height: 14, rx: 5, fill: "#dee2e6", stroke: "#495057", "stroke-width": 1.8 }, svg);

    // Zbierz krazki na kazdym slupku (od najwiekszego na dole do najmniejszego na gorze)
    var pegs = [];
    for (var p = 0; p < k; p++) pegs.push([]);
    for (var d = n - 1; d >= 0; d--) pegs[S.state[d]].push(d);

    var maxDiskW = Math.min(130, spacing * 0.84);
    var minDiskW = Math.max(34, maxDiskW * 0.34);
    var diskH = Math.min(24, Math.floor((baseY - pegTopY - 22) / n));

    for (var p2 = 0; p2 < k; p2++) (function (p) {
      var px = 30 + spacing * (p + 0.5);
      var isSel = S.selectedPeg === p;
      var canDrop = false;
      if (S.selectedPeg !== null && S.selectedPeg !== p) {
        var du = top[S.selectedPeg];
        canDrop = (top[p] < 0 || du < top[p]);
      }

      var grp = el("g", { "class": "hn-peg-grp" }, svg);
      grp.addEventListener("click", function () { clickPeg(p); });

      // Tlo klikalnego obszaru slupka
      var bgFill = isSel ? "#fff9db" : (canDrop ? "#ebfbee" : "transparent");
      var bgStroke = isSel ? "#f08c00" : (canDrop ? "#2f9e44" : "none");
      el("rect", {
        x: px - spacing * 0.46, y: 18, width: spacing * 0.92, height: baseY + 34,
        rx: 8, fill: bgFill, stroke: bgStroke, "stroke-width": 2, "stroke-dasharray": canDrop ? "5 4" : "none"
      }, grp);

      // Slupek
      el("rect", {
        x: px - 5, y: pegTopY, width: 10, height: baseY - pegTopY, rx: 4,
        fill: isSel ? "#f08c00" : "#868e96", stroke: "#343a40", "stroke-width": 1.2
      }, grp);

      // Etykieta slupka (A, B, C...)
      var lblTxt = HN.PEGS[p] + (p === 0 ? " (START)" : (p === k - 1 ? " (CEL)" : ""));
      el("text", {
        x: px, y: baseY + 32, "text-anchor": "middle", "font-size": 13, "font-weight": 700,
        fill: p === 0 ? "#2f9e44" : (p === k - 1 ? "#b08900" : "#212529")
      }, grp, lblTxt);

      if (canDrop) {
        el("text", { x: px, y: 36, "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: "#2f9e44" }, grp, "⬇ połóż tutaj");
      }

      // Krazki na slupku
      var stack = pegs[p];
      for (var idx = 0; idx < stack.length; idx++) {
        var diskIdx = stack[idx];
        var w = n === 1 ? maxDiskW : minDiskW + (maxDiskW - minDiskW) * (diskIdx / (n - 1));
        var isLifted = isSel && idx === stack.length - 1;
        var y = isLifted ? (pegTopY - diskH - 8) : (baseY - (idx + 1) * diskH - 2);
        var col = HN.COLORS[diskIdx % HN.COLORS.length];

        el("rect", {
          x: px - w / 2, y: y, width: w, height: diskH - 2, rx: 6,
          fill: col, stroke: isLifted ? "#212529" : "#343a40", "stroke-width": isLifted ? 2.5 : 1.2
        }, grp);
        el("text", {
          x: px, y: y + (diskH - 2) / 2 + 4, "text-anchor": "middle",
          "font-size": 12, "font-weight": 700, fill: "#ffffff"
        }, grp, String(diskIdx + 1));
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

    var userEdges = buildEdgeMap(S.trailStates);
    var bfsEdges = buildEdgeMap(S.bfs.states);
    var dfsEdges = buildEdgeMap(S.dfs.states);

    // Krawedzie tla
    L.edges.forEach(function (e) {
      var p1 = L.pos[e[0]], p2 = L.pos[e[1]], ek = eKey(e[0], e[1]);
      el("line", {
        x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
        stroke: "#dee2e6", "stroke-width": 1.6
      }, svg);

      if ((S.overlay === "dfs" || S.overlay === "all") && dfsEdges[ek]) {
        el("line", {
          x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
          stroke: "#9c36b5", "stroke-width": 3.0, "stroke-opacity": 0.65
        }, svg);
      }
      if ((S.overlay === "bfs" || S.overlay === "all") && bfsEdges[ek]) {
        el("line", {
          x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
          stroke: "#1971c2", "stroke-width": 3.8, "stroke-opacity": 0.9
        }, svg);
      }
      if ((S.overlay === "user" || S.overlay === "all") && userEdges[ek]) {
        el("line", {
          x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
          stroke: "#f08c00", "stroke-width": 2.8, "stroke-dasharray": S.overlay === "all" ? "4 3" : "none"
        }, svg);
      }
    });

    // Wezly
    var curKey = HN.stateKey(S.state);
    var startKey = HN.stateKey(S.trailStates[0]);
    var goalArr = [];
    for (var i = 0; i < S.n; i++) goalArr.push(S.k - 1);
    var goalKey = HN.stateKey(goalArr);
    var r = L.states.length <= 27 ? 6.5 : (L.states.length <= 81 ? 4.5 : 3.2);

    L.states.forEach(function (st) {
      var sk = HN.stateKey(st), p = L.pos[sk];
      var isCur = sk === curKey, isStart = sk === startKey, isGoal = sk === goalKey;
      var fill = isCur ? "#e03131" : (isGoal ? "#ffe066" : (isStart ? "#2f9e44" : (S.visitedSet[sk] ? "#ffd8a8" : "#ffffff")));
      var stroke = isCur ? "#c92a2a" : (isStart ? "#2f9e44" : (isGoal ? "#e67700" : "#868e96"));
      var c = el("circle", {
        cx: p[0], cy: p[1], r: isCur ? r + 2.2 : r,
        fill: fill, stroke: stroke, "stroke-width": isCur || isStart || isGoal ? 2.2 : 1.1
      }, svg);
      el("title", {}, c, HN.stateLabel(st));
    });
  }

  function renderComparison() {
    var box = $(".hn-compare");
    if (!S.showCompare) {
      box.style.display = "none";
      return;
    }
    box.style.display = "";
    var totalStates = Math.pow(S.k, S.n);
    var solvedByUser = HN.isGoal(S.state, S.k);
    var userMoves = S.history.length;
    var userUnique = Object.keys(S.visitedSet).length;

    var bfsMoves = S.bfs.solved ? S.bfs.moves.length : null;
    var dfsMoves = S.dfs.solved ? S.dfs.moves.length : null;

    var badge = "";
    if (!S.bfs.solved) {
      badge = "<div class='hn-verdict hn-v-warn'>⚠️ <b>Analiza przestrzeni stanów (k = 2 słupki):</b> Przestrzeń liczy <b>" +
        totalStates + " stanów</b>, ale ze stanu początkowego osiągalne są tylko <b>2 stany</b>! " +
        "Zarówno <b>BFS</b>, jak i <b>DFS</b> po odwiedzeniu 2 stanów opróżniają kolejkę/stos i udowadniają, że rozwiązanie nie istnieje.</div>";
    } else if (solvedByUser) {
      var extra = userMoves - bfsMoves;
      if (extra === 0) {
        badge = "<div class='hn-verdict hn-v-ok'>🏆 <b>Perfekcyjny wynik!</b> Rozwiązałeś(aś) łamigłówkę w <b>" +
          HN.nMoves(userMoves) + "</b> – to dokładnie <b>najkrótsza możliwa ścieżka</b> (taka sama jak w BFS), " +
          "podczas gdy DFS potrzebował aż <b>" + HN.nMoves(dfsMoves) + "</b>!</div>";
      } else {
        badge = "<div class='hn-verdict hn-v-info'>🎉 <b>Zagadka rozwiązana w " + HN.nMoves(userMoves) + "!</b> " +
          "Najkrótsza ścieżka znaleziona przez <b>BFS</b> ma <b>" + HN.nMoves(bfsMoves) + "</b> (Twoja ścieżka miała o " +
          HN.nMoves(extra) + " więcej). Zauważ jednak, że <b>DFS</b> błądził znacznie dłużej i potrzebował aż <b>" +
          HN.nMoves(dfsMoves) + "</b>!</div>";
      }
    } else {
      badge = "<div class='hn-verdict hn-v-info'>📊 <b>Podgląd porównania (przed ukończeniem):</b> " +
        "Dotychczas wykonałeś(aś) <b>" + HN.nMoves(userMoves) + "</b>. Najkrótsza droga (BFS) wymaga <b>" +
        HN.nMoves(bfsMoves) + "</b>, a ścieżka DFS liczy <b>" + HN.nMoves(dfsMoves) + "</b>.</div>";
    }

    function seqHtml(moves, cap) {
      if (!moves || !moves.length) return "<span style='color:#868e96'>— (brak ścieżki)</span>";
      var shown = moves.slice(0, cap || 35).map(function (m) {
        return "<span class='hn-chip' style='border-color:" + HN.COLORS[(m.disk - 1) % HN.COLORS.length] + "'>" +
          m.label + " <small>(kr." + m.disk + ")</small></span>";
      }).join(" → ");
      if (moves.length > (cap || 35)) shown += " … <i>(+ " + (moves.length - (cap || 35)) + " kolejnych)</i>";
      return shown;
    }

    box.innerHTML =
      "<h5>📊 Porównanie: Twoje rozwiązanie vs BFS (najkrótsza ścieżka) vs DFS</h5>" +
      badge +
      "<table class='hn-tbl'>" +
      "<tr><th>Podejście</th><th>Liczba ruchów (długość ścieżki)</th><th>Odwiedzone stany w grafie (z " + totalStates + ")</th><th>Różnica względem najkrótszej ścieżki</th><th>Akcja</th></tr>" +
      "<tr class='hn-row-user'>" +
        "<td><b>🧑 Twoje rozwiązanie</b></td>" +
        "<td><b>" + (solvedByUser ? HN.nMoves(userMoves) : HN.nMoves(userMoves) + " (w trakcie)") + "</b></td>" +
        "<td>" + HN.nStates(userUnique) + "</td>" +
        "<td>" + (!S.bfs.solved ? "cel nieosiągalny" : (solvedByUser ? (userMoves === bfsMoves ? "<b>0 (Optymalne! 🏆)</b>" : "+" + HN.nMoves(userMoves - bfsMoves)) : "—")) + "</td>" +
        "<td><button class='hn-play-user'>▶ Odtwórz moje</button></td>" +
      "</tr>" +
      "<tr class='hn-row-bfs'>" +
        "<td><b>🌊 BFS (wszerz – Kolejka FIFO)</b><br><small>Gwarantuje najkrótszą ścieżkę</small></td>" +
        "<td><b>" + (S.bfs.solved ? HN.nMoves(bfsMoves) : "brak") + "</b></td>" +
        "<td>" + HN.nStates(S.bfs.visitedCount) + " <small>(przeszukano wszerz)</small></td>" +
        "<td><b>0 (Najkrótsza ścieżka)</b></td>" +
        "<td>" + (S.bfs.solved ? "<button class='hn-play-bfs'>▶ Pokaż BFS</button>" : "—") + "</td>" +
      "</tr>" +
      "<tr class='hn-row-dfs'>" +
        "<td><b>🕳️ DFS (w głąb – Stos LIFO)</b><br><small>Idzie w głąb pierwszą wolną gałęzią</small></td>" +
        "<td><b>" + (S.dfs.solved ? HN.nMoves(dfsMoves) : "brak") + "</b></td>" +
        "<td>" + HN.nStates(S.dfs.visitedCount) + " <small>(przeszukano w głąb)</small></td>" +
        "<td>" + (S.dfs.solved ? "+" + HN.nMoves(dfsMoves - bfsMoves) + " (" + (dfsMoves / bfsMoves).toFixed(1) + "× dłużej!)" : "—") + "</td>" +
        "<td>" + (S.dfs.solved ? "<button class='hn-play-dfs'>▶ Pokaż DFS</button>" : "—") + "</td>" +
      "</tr>" +
      "</table>" +
      "<div class='hn-paths-detail'>" +
        "<div><b>🌊 Najkrótsza ścieżka (BFS – " + (S.bfs.solved ? HN.nMoves(bfsMoves) : "brak") + "):</b><div class='hn-seq'>" + seqHtml(S.bfs.moves, 32) + "</div></div>" +
        "<div style='margin-top:6px'><b>🕳️ Ścieżka znaleziona przez DFS (" + (S.dfs.solved ? HN.nMoves(dfsMoves) : "brak") + "):</b><div class='hn-seq'>" + seqHtml(S.dfs.moves, 32) + "</div></div>" +
      "</div>";

    var btnU = box.querySelector(".hn-play-user");
    if (btnU) btnU.addEventListener("click", function () { playSequence(S.trailStates, S.history, "Twoja ścieżka", "user"); });
    var btnB = box.querySelector(".hn-play-bfs");
    if (btnB) btnB.addEventListener("click", function () { playSequence(S.bfs.states, S.bfs.moves, "Najkrótsza ścieżka – BFS", "bfs"); });
    var btnD = box.querySelector(".hn-play-dfs");
    if (btnD) btnD.addEventListener("click", function () { playSequence(S.dfs.states, S.dfs.moves, "Ścieżka DFS", "dfs"); });
  }

  function render() {
    var totalStates = Math.pow(S.k, S.n);
    var optText = S.bfs.solved ? HN.nMoves(S.bfs.moves.length) : "brak (nieosiągalny)";
    $(".hn-stats").innerHTML =
      "Rozmiar przestrzeni stanów: <b>|S| = " + S.k + "<sup>" + S.n + "</sup> = " + totalStates + "</b> &nbsp;·&nbsp; " +
      "Bieżący stan: <code>" + HN.stateLabel(S.state) + "</code> &nbsp;·&nbsp; " +
      "Twoje ruchy: <b>" + S.history.length + "</b> &nbsp;·&nbsp; " +
      "Najkrótsza ścieżka (BFS): <b>" + optText + "</b>";

    var msgBox = $(".hn-msg");
    msgBox.className = "hn-msg hn-msg-" + S.msgKind;
    msgBox.innerHTML = S.msg;

    $(".hn-undo").disabled = S.history.length === 0;
    $(".hn-toggle-cmp").textContent = S.showCompare ? "🙈 Ukryj porównanie BFS / DFS" : "📊 Porównaj z BFS i DFS (najkrótsza ścieżka)";

    var histHtml = S.history.length
      ? S.history.map(function (m, idx) {
          return "<span class='hn-chip' style='border-color:" + HN.COLORS[(m.disk - 1) % HN.COLORS.length] + "'>" +
            (idx + 1) + ". " + m.label + "</span>";
        }).join(" ")
      : "<span style='color:#868e96;font-style:italic'>(jeszcze nie wykonano żadnego ruchu)</span>";
    $(".hn-history").innerHTML = "<b>Twoje ruchy (" + S.history.length + "):</b> " + histHtml;

    root.querySelectorAll(".hn-ov-btn").forEach(function (b) {
      b.className = "hn-ov-btn" + (b.getAttribute("data-ov") === S.overlay ? " hn-ov-on" : "");
    });

    drawTowers();
    drawStateGraph();
    renderComparison();
  }

  $(".hn-disks").value = String(CFG.n_disks);
  $(".hn-pegs").value = String(CFG.n_pegs);
  $(".hn-disks").addEventListener("change", initGame);
  $(".hn-pegs").addEventListener("change", initGame);
  $(".hn-undo").addEventListener("click", undoMove);
  $(".hn-reset").addEventListener("click", initGame);
  $(".hn-toggle-cmp").addEventListener("click", function () {
    S.showCompare = !S.showCompare;
    if (S.showCompare && S.overlay === "user") S.overlay = "all";
    render();
  });
  root.querySelectorAll(".hn-ov-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      S.overlay = b.getAttribute("data-ov");
      render();
    });
  });

  initGame();
})();
"""

_HANOI_CSS = r"""
<style>
.hn-card{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#212529;background:#ffffff;
  border:1px solid #ced4da;border-radius:10px;padding:14px 16px;max-width:1040px;font-size:14px;line-height:1.45}
.hn-card *{box-sizing:border-box}
.hn-card h4{margin:0 0 8px 0;font-size:17px;color:#212529}
.hn-card h5{margin:0 0 8px 0;font-size:15px;color:#212529}
.hn-controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:8px}
.hn-controls label{display:flex;gap:6px;align-items:center;font-weight:600;color:#212529}
.hn-card select,.hn-card button{font-size:13.5px;padding:5px 11px;border:1px solid #adb5bd;border-radius:6px;
  background:#f8f9fa;color:#212529;cursor:pointer}
.hn-card button:hover:not(:disabled){background:#e9ecef}
.hn-card button:disabled{opacity:0.5;cursor:default}
.hn-card button.hn-toggle-cmp{background:#1971c2;border-color:#1864ab;color:#fff;font-weight:600}
.hn-card button.hn-toggle-cmp:hover{background:#1864ab}
.hn-stats{background:#f1f3f5;border-radius:6px;padding:6px 10px;margin:6px 0;font-size:13px;color:#212529}
.hn-msg{border-radius:6px;padding:7px 11px;margin:6px 0;font-size:13.5px}
.hn-msg-info{background:#e7f5ff;border-left:4px solid #1971c2;color:#212529}
.hn-msg-ok{background:#d3f9d8;border-left:4px solid #2b8a3e;color:#212529}
.hn-msg-warn{background:#fff3bf;border-left:4px solid #f08c00;color:#212529}
.hn-msg-bad{background:#ffe3e3;border-left:4px solid #e03131;color:#212529}
.hn-body{display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;margin-top:8px}
.hn-towers-box{flex:1.35 1 380px;min-width:290px;border:1px solid #dee2e6;border-radius:8px;background:#ffffff;padding:6px}
.hn-towers-svg{width:100%;height:auto;display:block;user-select:none}
.hn-peg-grp{cursor:pointer}
.hn-graph-box{flex:1 1 300px;min-width:260px;border:1px solid #dee2e6;border-radius:8px;background:#f8f9fa;padding:8px;
  display:flex;flex-direction:column;justify-content:space-between}
.hn-graph-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:4px;font-size:12px}
.hn-ov-bar{display:inline-flex;border:1px solid #adb5bd;border-radius:6px;overflow:hidden}
.hn-card .hn-ov-bar button{padding:2px 7px;font-size:11.5px;border:none;border-radius:0;background:#fff}
.hn-card .hn-ov-bar button + button{border-left:1px solid #adb5bd}
.hn-card .hn-ov-bar button.hn-ov-on{background:#343a40;color:#fff;font-weight:700}
.hn-graph-svg{width:100%;height:auto;display:block}
.hn-legend{display:flex;flex-wrap:wrap;gap:8px;font-size:11.5px;color:#495057;justify-content:center}
.hn-history{margin-top:8px;font-size:13px;min-height:24px;display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.hn-chip{display:inline-block;padding:1px 7px;border-radius:10px;border:2px solid #adb5bd;background:#fff;font-size:12px;font-weight:600}
.hn-compare{margin-top:12px;padding:12px;border:2px solid #1971c2;border-radius:8px;background:#f8f9fa}
.hn-verdict{padding:8px 10px;border-radius:6px;margin-bottom:8px;font-size:13.5px}
.hn-v-ok{background:#d3f9d8;border-left:4px solid #2b8a3e}
.hn-v-info{background:#e7f5ff;border-left:4px solid #1971c2}
.hn-v-warn{background:#fff3bf;border-left:4px solid #f08c00}
.hn-tbl{width:100%;border-collapse:collapse;font-size:13px;background:#fff;margin-bottom:8px}
.hn-tbl th,.hn-tbl td{border:1px solid #ced4da;padding:6px 8px;text-align:left;color:#212529}
.hn-tbl th{background:#e9ecef;font-weight:700}
.hn-row-user td{background:#fff9db}
.hn-row-bfs td{background:#e7f5ff}
.hn-row-dfs td{background:#f3d9fa}
.hn-paths-detail{font-size:12.5px;color:#212529;background:#fff;border:1px solid #dee2e6;border-radius:6px;padding:8px}
.hn-seq{margin-top:4px;display:flex;flex-wrap:wrap;gap:4px;align-items:center;line-height:1.6}
</style>
"""

_HANOI_HTML = r"""
<div class="hn-card" id="__ROOT__">
  <h4>🗼 Wyzwanie: Rozwiąż Wieże Hanoi i porównaj swój wynik z BFS oraz DFS</h4>
  <div class="hn-controls">
    <label>Krążków (n): <select class="hn-disks">
      <option value="2">2 krążki</option>
      <option value="3">3 krążki</option>
      <option value="4">4 krążki</option>
      <option value="5">5 krążków</option>
      <option value="6">6 krążków</option>
    </select></label>
    <label>Słupków (k): <select class="hn-pegs">
      <option value="2">2 słupki (A–B)</option>
      <option value="3">3 słupki (A–C)</option>
      <option value="4">4 słupki (A–D)</option>
      <option value="5">5 słupków (A–E)</option>
    </select></label>
    <button class="hn-undo">↶ Cofnij ruch</button>
    <button class="hn-reset">⟲ Od początku</button>
    <button class="hn-toggle-cmp">📊 Porównaj z BFS i DFS (najkrótsza ścieżka)</button>
  </div>
  <div class="hn-stats"></div>
  <div class="hn-msg"></div>
  <div class="hn-body">
    <div class="hn-towers-box">
      <svg class="hn-towers-svg" viewBox="0 0 540 290" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div class="hn-graph-box">
      <div class="hn-graph-head">
        <b>Mapa przestrzeni stanów (graf):</b>
        <div class="hn-ov-bar">
          <button class="hn-ov-btn" data-ov="user">Twoja</button>
          <button class="hn-ov-btn" data-ov="bfs">BFS</button>
          <button class="hn-ov-btn" data-ov="dfs">DFS</button>
          <button class="hn-ov-btn" data-ov="all">Wszystkie</button>
        </div>
      </div>
      <svg class="hn-graph-svg" viewBox="0 0 380 310" xmlns="http://www.w3.org/2000/svg"></svg>
      <div class="hn-legend">
        <span>🟢 Start</span>
        <span>🟡 Cel</span>
        <span>🔴 Bieżący</span>
        <span style="color:#f08c00;font-weight:700">▬ Twoja ścieżka</span>
        <span style="color:#1971c2;font-weight:700">▬ Najkrótsza (BFS)</span>
        <span style="color:#9c36b5;font-weight:700">▬ Ścieżka DFS</span>
      </div>
    </div>
  </div>
  <div class="hn-history"></div>
  <div class="hn-compare" style="display:none"></div>
</div>
"""


def build_hanoi_html(n_disks: int = 3, n_pegs: int = 3) -> str:
    """Buduje samodzielny widget HTML/SVG/JS do rozwiazywania Wiez Hanoi i porownania z BFS/DFS."""
    if not 2 <= n_disks <= 6:
        raise ValueError("n_disks musi byc w przedziale 2..6")
    if not 2 <= n_pegs <= 5:
        raise ValueError("n_pegs musi byc w przedziale 2..5")
    root = "hn-" + uuid.uuid4().hex[:10]
    cfg = json.dumps({"n_disks": int(n_disks), "n_pegs": int(n_pegs)})
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


def hanoi_exercise(n_disks: int = 3, n_pegs: int = 3):
    """Wyswietla interaktywna gre Wieze Hanoi z wyborem liczby krazkow (min. 2) i slupkow (min. 2)
    oraz porownaniem ruchow gracza z BFS (najkrotsza sciezka) i DFS."""
    from IPython.display import HTML, display
    display(HTML(build_hanoi_html(n_disks=n_disks, n_pegs=n_pegs)))
