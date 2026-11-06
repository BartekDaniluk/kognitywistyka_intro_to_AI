"""
lab_states.py - interaktywne przestrzenie stanow do zajec "Wprowadzenie do AI".

Dwa problemy:
  * VacuumWorld  - swiat agenta-odkurzacza (n pol w linii),
  * WaterJugs    - problem dwoch pojemnikow (dzbanow) z woda.

Kazdy problem jest zdefiniowany "podrecznikowo" (AIMA, rozdz. 3):
  stan poczatkowy, akcje(s), model przejscia result(s, a),
  test celu is_goal(s), koszt akcji action_cost(s, a, s').

Z definicji automatycznie budowana jest przestrzen stanow (graf), tabela
przejsc, rozwiazanie optymalne (przeszukiwanie z jednolitym kosztem) oraz
interaktywna symulacja w Jupyter / Google Colab (ipywidgets + matplotlib).

Szybki start w notebooku:
    import lab_states as ls
    ls.vacuum_app()        # odkurzacz
    ls.water_jugs_app()    # pojemniki
"""
from __future__ import annotations

import heapq
import html
import itertools
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

# --------------------------------------------------------------------------
# Jezyk interfejsu
# --------------------------------------------------------------------------
LANG = "pl"

_T = {
    "pl": {
        # ogolne
        "states": "Stany", "initial": "Stan początkowy", "goal": "Test celu / stany docelowe",
        "actions": "Akcje", "transition": "Model przejścia", "cost": "Koszt akcji",
        "stats": "Rozmiar przestrzeni", "component": "Składnik", "definition": "Definicja",
        "tab_graph": "Graf + symulacja", "tab_def": "Definicja formalna", "tab_table": "Tabela przejść",
        "show_solution": "Pokaż rozwiązanie optymalne", "show_all": "Pokaż też stany nieosiągalne",
        "edge_labels": "Etykiety krawędzi", "self_loops": "Pętle (akcje bez zmiany stanu)",
        "undo": "↶ Cofnij", "reset": "⟲ Od początku",
        "current": "Stan bieżący", "path": "Wykonane akcje", "path_empty": "(jeszcze nic)",
        "total_cost": "koszt łączny", "goal_reached": "🎉 Cel osiągnięty!",
        "not_applicable": "niedozwolona w tym stanie",
        "no_solution": "Brak rozwiązania – żaden stan docelowy nie jest osiągalny ze stanu początkowego.",
        "solution": "Rozwiązanie optymalne", "steps": "kroków",
        "n_states": "wszystkich stanów: <b>{all}</b>, osiągalnych ze stanu początkowego: <b>{reach}</b>, "
                    "przejść (krawędzi) między osiągalnymi: <b>{edges}</b>",
        "lg_initial": "stan początkowy", "lg_goal": "stan docelowy", "lg_current": "stan bieżący",
        "lg_unreach": "nieosiągalny", "lg_solution": "rozwiązanie", "lg_trail": "twoja ścieżka",
        "state_col": "Stan", "no_goal_states": "(brak stanów docelowych w przestrzeni)",
        # odkurzacz
        "vac_title": "Świat odkurzacza",
        "vac_n": "Liczba pól", "vac_maxdirty": "Maks. brudnych", "vac_pos": "Start: pozycja",
        "vac_dirt": "Start: brud w polach", "vac_move_cost": "Koszt ruchu", "vac_suck_cost": "Koszt odkurzania",
        "vac_noop": "Akcja „nic nie rób”", "vac_dirty": "ś", "vac_clean": "c",
        "vac_L": "W lewo", "vac_R": "W prawo", "vac_S": "Odkurzaj", "vac_N": "Nic nie rób",
        "vac_Ls": "L", "vac_Rs": "P", "vac_Ss": "O", "vac_Ns": "N",
        "vac_states_desc": "[położenie robota; czystość każdego pola], np. <code>[A; A(ś), B(c)]</code> "
                           "(ś = śmieci, c = czysto). Pól: {n}, maks. brudnych pól: {k}.",
        "vac_goal_desc": "wszystkie pola czyste",
        "vac_tr_L": "przesuń robota o jedno pole w lewo (na skrajnym lewym polu – bez zmian)",
        "vac_tr_R": "przesuń robota o jedno pole w prawo (na skrajnym prawym polu – bez zmian)",
        "vac_tr_S": "pole, na którym stoi robot, staje się czyste",
        "vac_tr_N": "stan się nie zmienia",
        "vac_warn_dirt": "⚠ Stan początkowy ma więcej brudnych pól niż limit – limit zwiększono.",
        # pojemniki
        "jug_title": "Problem pojemników z wodą",
        "jug_capA": "Pojemność A", "jug_capB": "Pojemność B", "jug_initA": "Start: woda w A",
        "jug_initB": "Start: woda w B", "jug_goalA": "Cel: woda w A", "jug_goalB": "Cel: woda w B",
        "jug_cost": "Koszt akcji", "jug_cost_unit": "1 za każdą akcję", "jug_cost_water": "litry przelanej wody",
        "jug_any": "dowolnie",
        "jug_states_desc": "(X, Y), gdzie X – ilość wody w pojemniku A, Y – w pojemniku B, "
                           "0 ≤ X ≤ {a}, 0 ≤ Y ≤ {b} (liczby całkowite)",
        "jug_r1": "napełnij pojemnik A z kranu", "jug_r2": "napełnij pojemnik B z kranu",
        "jug_r3": "opróżnij pojemnik A do zlewu", "jug_r4": "opróżnij pojemnik B do zlewu",
        "jug_r5": "napełnij pojemnik A wodą z pojemnika B", "jug_r6": "napełnij pojemnik B wodą z pojemnika A",
        "jug_r7": "przelej całą wodę z pojemnika A do pojemnika B",
        "jug_r8": "przelej całą wodę z pojemnika B do pojemnika A",
        "jug_rule": "Reguła", "jug_cond": "Warunek", "jug_effect": "Wynik", "jug_desc": "Opis",
        "jug_note": "Akcje typu „wlej/wylej/przelej <i>pewną</i> ilość wody” pomijamy – dawałyby "
                    "nieskończenie wiele stanów (brak podziałki na pojemnikach).",
        "jug_cost_unit_desc": "każda akcja kosztuje 1 (minimalizujemy liczbę czynności)",
        "jug_cost_water_desc": "koszt = liczba litrów wody nalanej, wylanej lub przelanej w danej akcji",
    },
    "en": {
        "states": "States", "initial": "Initial state", "goal": "Goal test / goal states",
        "actions": "Actions", "transition": "Transition model", "cost": "Action cost",
        "stats": "State-space size", "component": "Component", "definition": "Definition",
        "tab_graph": "Graph + simulation", "tab_def": "Formal definition", "tab_table": "Transition table",
        "show_solution": "Show optimal solution", "show_all": "Show unreachable states too",
        "edge_labels": "Edge labels", "self_loops": "Self-loops (actions with no effect)",
        "undo": "↶ Undo", "reset": "⟲ Restart",
        "current": "Current state", "path": "Actions taken", "path_empty": "(nothing yet)",
        "total_cost": "total cost", "goal_reached": "🎉 Goal reached!",
        "not_applicable": "not applicable in this state",
        "no_solution": "No solution – no goal state is reachable from the initial state.",
        "solution": "Optimal solution", "steps": "steps",
        "n_states": "all states: <b>{all}</b>, reachable from the initial state: <b>{reach}</b>, "
                    "transitions (edges) between reachable states: <b>{edges}</b>",
        "lg_initial": "initial state", "lg_goal": "goal state", "lg_current": "current state",
        "lg_unreach": "unreachable", "lg_solution": "solution", "lg_trail": "your path",
        "state_col": "State", "no_goal_states": "(no goal states in the space)",
        "vac_title": "Vacuum world",
        "vac_n": "Squares", "vac_maxdirty": "Max dirty", "vac_pos": "Start: position",
        "vac_dirt": "Start: dirt in squares", "vac_move_cost": "Move cost", "vac_suck_cost": "Suck cost",
        "vac_noop": "NoOp action", "vac_dirty": "D", "vac_clean": "C",
        "vac_L": "Left", "vac_R": "Right", "vac_S": "Suck", "vac_N": "NoOp",
        "vac_Ls": "L", "vac_Rs": "R", "vac_Ss": "S", "vac_Ns": "N",
        "vac_states_desc": "[robot position; status of every square], e.g. <code>[A; A(D), B(C)]</code> "
                           "(D = dirty, C = clean). Squares: {n}, max dirty squares: {k}.",
        "vac_goal_desc": "all squares clean",
        "vac_tr_L": "move the robot one square left (no effect in the leftmost square)",
        "vac_tr_R": "move the robot one square right (no effect in the rightmost square)",
        "vac_tr_S": "the square under the robot becomes clean",
        "vac_tr_N": "the state does not change",
        "vac_warn_dirt": "⚠ The initial state has more dirty squares than the limit – limit raised.",
        "jug_title": "Water jug problem",
        "jug_capA": "Capacity A", "jug_capB": "Capacity B", "jug_initA": "Start: water in A",
        "jug_initB": "Start: water in B", "jug_goalA": "Goal: water in A", "jug_goalB": "Goal: water in B",
        "jug_cost": "Action cost", "jug_cost_unit": "1 per action", "jug_cost_water": "litres of water moved",
        "jug_any": "any",
        "jug_states_desc": "(X, Y), where X – water in jug A, Y – water in jug B, "
                           "0 ≤ X ≤ {a}, 0 ≤ Y ≤ {b} (integers)",
        "jug_r1": "fill jug A from the tap", "jug_r2": "fill jug B from the tap",
        "jug_r3": "empty jug A into the sink", "jug_r4": "empty jug B into the sink",
        "jug_r5": "fill jug A with water from jug B", "jug_r6": "fill jug B with water from jug A",
        "jug_r7": "pour all water from jug A into jug B", "jug_r8": "pour all water from jug B into jug A",
        "jug_rule": "Rule", "jug_cond": "Condition", "jug_effect": "Result", "jug_desc": "Description",
        "jug_note": "Actions like “pour <i>some</i> amount of water” are omitted – they would give "
                    "infinitely many states (the jugs have no markings).",
        "jug_cost_unit_desc": "every action costs 1 (we minimise the number of actions)",
        "jug_cost_water_desc": "cost = litres of water filled, emptied or poured in that action",
    },
}


def set_language(lang: str) -> None:
    """'pl' albo 'en'."""
    global LANG
    if lang not in _T:
        raise ValueError(f"unknown language {lang!r}, use one of {list(_T)}")
    LANG = lang


def t(key: str, **kw) -> str:
    s = _T[LANG][key]
    return s.format(**kw) if kw else s


def n_steps(n: int) -> str:
    """'1 krok', '2 kroki', '5 kroków' (polska odmiana) / '1 step', '2 steps'."""
    if LANG == "pl":
        if n == 1:
            w = "krok"
        elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            w = "kroki"
        else:
            w = "kroków"
    else:
        w = "step" if n == 1 else "steps"
    return f"{n} {w}"


# Kolory
C_INITIAL = "#2f9e44"
C_GOAL_FILL = "#ffe066"
C_CURRENT = "#e03131"
C_SOLUTION = "#1c7ed6"
C_TRAIL = "#f76707"
C_NODE = "#ffffff"
C_NODE_EDGE = "#495057"
C_UNREACH = "#dee2e6"


# --------------------------------------------------------------------------
# Ogolny problem przeszukiwania
# --------------------------------------------------------------------------
class Problem:
    """Bazowa klasa problemu (sformulowanie jak w AIMA)."""

    def __init__(self, initial):
        self.initial = initial

    # --- definicja problemu (do nadpisania) ---
    def all_states(self):
        """Wszystkie stany przestrzeni (takze nieosiagalne)."""
        raise NotImplementedError

    def action_list(self):
        """Wszystkie akcje (identyfikatory), w stalej kolejnosci."""
        raise NotImplementedError

    def applicable(self, s, a) -> bool:
        return True

    def actions(self, s):
        return [a for a in self.action_list() if self.applicable(s, a)]

    def result(self, s, a):
        raise NotImplementedError

    def is_goal(self, s) -> bool:
        raise NotImplementedError

    def action_cost(self, s, a, s2):
        return 1

    # --- prezentacja (do nadpisania) ---
    def sort_key(self, s):
        return s

    def state_label(self, s) -> str:
        return str(s)

    def action_name(self, a) -> str:
        return str(a)

    def action_short(self, a) -> str:
        return str(a)

    def action_color(self, a) -> str:
        return "#868e96"

    def legend_actions(self):
        """[(etykieta, kolor)] do legendy krawedzi."""
        return []

    def layout(self, states):
        raise NotImplementedError

    node_w, node_h = 0.9, 0.5

    def draw_node_content(self, ax, s, x, y, faded: bool):
        ax.text(x, y, self.state_label(s), ha="center", va="center", fontsize=8,
                color="#adb5bd" if faded else "black", zorder=6)

    def draw_state(self, ax, s):
        ax.text(0.5, 0.5, self.state_label(s), ha="center", va="center", fontsize=16)
        ax.axis("off")

    def definition_rows(self):
        """[(nazwa skladnika, html)]"""
        return []

    def solution_text(self, steps) -> str:
        return " – ".join(self.action_short(a) for _, a, _ in steps)


class StateSpace:
    """Graf przestrzeni stanow wygenerowany z definicji problemu."""

    def __init__(self, problem: Problem, include_unreachable: bool = False):
        self.problem = problem
        # stany osiagalne - BFS od stanu poczatkowego
        reach = {problem.initial}
        q = deque([problem.initial])
        while q:
            s = q.popleft()
            for a in problem.actions(s):
                s2 = problem.result(s, a)
                if s2 not in reach:
                    reach.add(s2)
                    q.append(s2)
        self.reachable = reach
        everything = set(problem.all_states()) | reach
        self.all_states = sorted(everything, key=problem.sort_key)
        self.number = {s: i + 1 for i, s in enumerate(self.all_states)}
        shown = everything if include_unreachable else reach
        self.states = sorted(shown, key=problem.sort_key)
        shown = set(self.states)
        # krawedzie: (s, s2) -> [akcje]
        self.edges: dict = {}
        for s in self.states:
            for a in problem.actions(s):
                s2 = problem.result(s, a)
                if s2 in shown:
                    self.edges.setdefault((s, s2), []).append(a)
        self.goals = [s for s in self.states if problem.is_goal(s)]

    def n_edges_reachable(self, include_loops=False):
        return sum(1 for (s, s2) in self.edges
                   if s in self.reachable and s2 in self.reachable and (include_loops or s != s2))

    def label(self, s) -> str:
        return f"#{self.number[s]} {self.problem.state_label(s)}"


def solve(problem: Problem):
    """Przeszukiwanie z jednolitym kosztem (Dijkstra). Zwraca (kroki, koszt) albo None.
    kroki = [(stan, akcja, nastepny_stan), ...]"""
    start = problem.initial
    counter = itertools.count()
    frontier = [(0, next(counter), start)]
    g = {start: 0}
    parent = {start: None}
    while frontier:
        c, _, s = heapq.heappop(frontier)
        if c > g[s]:
            continue
        if problem.is_goal(s):
            steps = []
            while parent[s] is not None:
                prev, a = parent[s]
                steps.append((prev, a, s))
                s = prev
            return steps[::-1], c
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            nc = c + problem.action_cost(s, a, s2)
            if s2 not in g or nc < g[s2]:
                g[s2] = nc
                parent[s2] = (s, a)
                heapq.heappush(frontier, (nc, next(counter), s2))
    return None


# --------------------------------------------------------------------------
# Rysowanie grafu
# --------------------------------------------------------------------------
def _bezier_mid(p1, p2, rad):
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    return (x1 + x2) / 2 + 0.5 * rad * dy, (y1 + y2) / 2 - 0.5 * rad * dx


def draw_space(space: StateSpace, ax=None, current=None, trail=(), solution=None,
               edge_labels=True, self_loops=True, scale=0.9, max_size=(15, 13)):
    """Rysuje graf przestrzeni stanow. Zwraca figure."""
    p = space.problem
    pos = p.layout(space.states)
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    w, h = p.node_w, p.node_h
    x0, x1 = min(xs) - w, max(xs) + w
    y0, y1 = min(ys) - h - 0.4, max(ys) + h + 0.7
    if ax is None:
        fw, fh = (x1 - x0) * scale, (y1 - y0) * scale
        k = min(1.0, max_size[0] / fw, max_size[1] / fh)
        fig, ax = plt.subplots(figsize=(max(fw * k, 4), max(fh * k, 3)))
    else:
        fig = ax.figure
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")

    sol_edges = {(s, s2) for s, _, s2 in (solution or [])}
    trail_edges = {(s, s2) for s, _, s2 in trail}
    sol_states = {s for st in (solution or []) for s in (st[0], st[2])}
    trail_states = {s for st in trail for s in (st[0], st[2])}

    # wezly (najpierw, zeby strzalki mogly sie do nich przycinac)
    patches = {}
    for s in space.states:
        x, y = pos[s]
        faded = s not in space.reachable
        fc = C_UNREACH if faded else (C_GOAL_FILL if p.is_goal(s) else C_NODE)
        ec, lw = C_NODE_EDGE, 1.0
        if s in sol_states:
            ec, lw = C_SOLUTION, 2.0
        if s in trail_states:
            ec, lw = C_TRAIL, 2.0
        if s == p.initial:
            ec, lw = C_INITIAL, 3.0
        if s == current:
            ec, lw = C_CURRENT, 3.5
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             fc=fc, ec=ec, lw=lw, zorder=4, alpha=0.6 if faded else 1.0)
        ax.add_patch(box)
        patches[s] = box
        p.draw_node_content(ax, s, x, y, faded)
        ax.text(x - w / 2, y + h / 2 + 0.03, str(space.number[s]), fontsize=6.5,
                color="#868e96", ha="left", va="bottom", zorder=6)

    # krawedzie
    for (s, s2), acts in space.edges.items():
        lbl = ",".join(p.action_short(a) for a in acts)
        col = p.action_color(acts[0])
        lw, z, alpha = 1.0, 2, 0.75
        if (s, s2) in sol_edges:
            col, lw, z, alpha = C_SOLUTION, 3.0, 3, 1.0
        if (s, s2) in trail_edges:
            col, lw, z, alpha = C_TRAIL, 3.0, 3, 1.0
        if s not in space.reachable:
            alpha = 0.3
        x, y = pos[s]
        if s == s2:
            if not self_loops:
                continue
            arc = Arc((x, y + h / 2), w * 0.45, 0.55, theta1=-10, theta2=190,
                      color=col, lw=lw, zorder=z, alpha=alpha)
            ax.add_patch(arc)
            ax.annotate("", xy=(x + w * 0.225, y + h / 2 + 0.02),
                        xytext=(x + w * 0.226, y + h / 2 + 0.08),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=8),
                        zorder=z)
            if edge_labels:
                ax.text(x, y + h / 2 + 0.3, lbl, fontsize=6.5, ha="center", va="bottom",
                        color=col, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
            continue
        rad = 0.18
        arr = FancyArrowPatch(pos[s], pos[s2], patchA=patches[s], patchB=patches[s2],
                              arrowstyle="-|>", mutation_scale=10, lw=lw, color=col,
                              connectionstyle=f"arc3,rad={rad}", zorder=z, alpha=alpha)
        ax.add_patch(arr)
        if edge_labels:
            mx, my = _bezier_mid(pos[s], pos[s2], rad)
            ax.text(mx, my, lbl, fontsize=7, ha="center", va="center", color=col, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # legenda
    handles = [
        Patch(fc=C_NODE, ec=C_INITIAL, lw=2.5, label=t("lg_initial")),
        Patch(fc=C_GOAL_FILL, ec=C_NODE_EDGE, label=t("lg_goal")),
    ]
    if current is not None:
        handles.append(Patch(fc=C_NODE, ec=C_CURRENT, lw=2.5, label=t("lg_current")))
    if any(s not in space.reachable for s in space.states):
        handles.append(Patch(fc=C_UNREACH, ec=C_NODE_EDGE, label=t("lg_unreach")))
    if solution:
        handles.append(Line2D([], [], color=C_SOLUTION, lw=3, label=t("lg_solution")))
    if trail:
        handles.append(Line2D([], [], color=C_TRAIL, lw=3, label=t("lg_trail")))
    handles += [Line2D([], [], color=c, lw=1.5, label=l) for l, c in p.legend_actions()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8,
              frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Swiat odkurzacza
# --------------------------------------------------------------------------
class VacuumWorld(Problem):
    """Stan: (pozycja robota, krotka brudu), np. (0, (1, 0)) = [A; A(s), B(c)]."""

    ACTIONS = ("L", "R", "S")

    def __init__(self, n=2, initial=None, max_dirty=None, move_cost=1, suck_cost=1, noop=False):
        self.n = n
        if initial is None:
            initial = (0, (1,) * n)
        pos, dirt = initial
        dirt = tuple(int(bool(d)) for d in dirt)
        assert len(dirt) == n and 0 <= pos < n
        self.max_dirty = n if max_dirty is None else max(max_dirty, sum(dirt))
        self.move_cost, self.suck_cost, self.noop = move_cost, suck_cost, noop
        super().__init__((pos, dirt))
        cw = 0.42
        self.node_w, self.node_h = n * cw + 0.16, 0.52
        self._cw = cw

    @staticmethod
    def cell_name(i):
        return "ABCDEFGH"[i]

    def all_states(self):
        for dirt in itertools.product((1, 0), repeat=self.n):
            if sum(dirt) <= self.max_dirty:
                for p in range(self.n):
                    yield (p, dirt)

    def action_list(self):
        return self.ACTIONS + (("N",) if self.noop else ())

    def result(self, s, a):
        p, dirt = s
        if a == "L":
            return (max(p - 1, 0), dirt)
        if a == "R":
            return (min(p + 1, self.n - 1), dirt)
        if a == "S":
            d = list(dirt)
            d[p] = 0
            return (p, tuple(d))
        return s

    def is_goal(self, s):
        return not any(s[1])

    def action_cost(self, s, a, s2):
        return {"L": self.move_cost, "R": self.move_cost, "S": self.suck_cost}.get(a, 0 if a == "N" else 1)

    def sort_key(self, s):
        p, dirt = s
        return (p, tuple(1 - d for d in dirt))  # [A;ss], [A;sc], [A;cs], ... jak na wykladzie

    def state_label(self, s):
        p, dirt = s
        cells = ", ".join(f"{self.cell_name(i)}({t('vac_dirty') if d else t('vac_clean')})"
                          for i, d in enumerate(dirt))
        return f"[{self.cell_name(p)}; {cells}]"

    def action_name(self, a):
        return t(f"vac_{a}")

    def action_short(self, a):
        return t(f"vac_{a}s")

    def action_color(self, a):
        return {"L": "#5c7cfa", "R": "#5c7cfa", "S": "#a0522d", "N": "#adb5bd"}[a]

    def legend_actions(self):
        out = [(f"{t('vac_Ls')}/{t('vac_Rs')} = {t('vac_L')}/{t('vac_R')}", "#5c7cfa"),
               (f"{t('vac_Ss')} = {t('vac_S')}", "#a0522d")]
        if self.noop:
            out.append((f"{t('vac_Ns')} = {t('vac_N')}", "#adb5bd"))
        return out

    def layout(self, states):
        n = self.n
        by_level: dict = {}
        for p, dirt in states:
            by_level.setdefault(sum(dirt), set()).add(dirt)
        levels = sorted(by_level, reverse=True)  # od najbrudniejszych do czystych
        xgap = self.node_w + 1.6
        row_h = 1.3
        block_h = n * row_h + 0.8
        pos = {}
        for li, L in enumerate(levels):
            masks = sorted(by_level[L], reverse=True)
            total = (len(masks) - 1) * block_h + (n - 1) * row_h
            for bi, m in enumerate(masks):
                top = total / 2 - bi * block_h
                for p in range(n):
                    if (p, m) in states:
                        pos[(p, m)] = (li * xgap, top - p * row_h)
        return pos

    def draw_node_content(self, ax, s, x, y, faded):
        p, dirt = s
        cw, n = self._cw, self.n
        left = x - n * cw / 2
        a = 0.35 if faded else 1.0
        for i, d in enumerate(dirt):
            cx = left + i * cw
            ax.add_patch(Rectangle((cx + 0.02, y - 0.19), cw - 0.04, 0.38, fc="#f1f3f5" if not d else "#f3e3c3",
                                   ec="#868e96", lw=0.6, zorder=5, alpha=a))
            if d:
                for dx, dy in ((0.1, -0.1), (0.2, -0.05), (0.3, -0.12), (0.15, 0.0)):
                    ax.add_patch(Circle((cx + dx, y + dy), 0.025, fc="#7f4f24", ec="none", zorder=6, alpha=a))
        ax.add_patch(Circle((left + p * cw + cw / 2, y + 0.06), 0.1, fc="#364fc7", ec="white",
                            lw=0.8, zorder=7, alpha=a))

    def draw_state(self, ax, s):
        p, dirt = s
        n = self.n
        for i, d in enumerate(dirt):
            ax.add_patch(Rectangle((i, 0), 0.96, 0.96, fc="#f3e3c3" if d else "#f8f9fa", ec="#495057", lw=1.5))
            ax.text(i + 0.48, -0.12, self.cell_name(i), ha="center", va="top", fontsize=13, weight="bold")
            if d:
                for dx, dy, r in ((0.2, 0.2, 0.05), (0.35, 0.3, 0.035), (0.7, 0.18, 0.06),
                                  (0.6, 0.35, 0.03), (0.8, 0.4, 0.04), (0.25, 0.42, 0.03)):
                    ax.add_patch(Circle((i + dx, dy), r, fc="#7f4f24", ec="none"))
        # robot
        ax.add_patch(Circle((p + 0.48, 0.62), 0.22, fc="#364fc7", ec="#1c2a6b", lw=2))
        ax.add_patch(Circle((p + 0.48, 0.62), 0.07, fc="#91a7ff", ec="none"))
        ax.set_xlim(-0.1, n + 0.05)
        ax.set_ylim(-0.4, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")

    def definition_rows(self):
        names = ", ".join(f"<b>{self.action_name(a)}</b> ({self.action_short(a)})" for a in self.action_list())
        tr = "<br>".join(f"<b>{self.action_name(a)}</b>: {t('vac_tr_' + a)}" for a in self.action_list())
        cost = (f"{t('vac_L')}, {t('vac_R')}: <b>{self.move_cost}</b>; "
                f"{t('vac_S')}: <b>{self.suck_cost}</b>")
        if self.noop:
            cost += f"; {t('vac_N')}: <b>0</b>"
        return [
            (t("states"), t("vac_states_desc", n=self.n, k=self.max_dirty)),
            (t("initial"), html.escape(self.state_label(self.initial))),
            (t("goal"), t("vac_goal_desc")),
            (t("actions"), names),
            (t("transition"), tr),
            (t("cost"), cost),
        ]


# --------------------------------------------------------------------------
# Pojemniki z woda
# --------------------------------------------------------------------------
class WaterJugs(Problem):
    """Stan: (X, Y). Reguly 1-8 numerowane jak w materialach do zajec."""

    COLORS = {1: "#1c7ed6", 2: "#1c7ed6", 3: "#868e96", 4: "#868e96",
              5: "#9c36b5", 6: "#9c36b5", 7: "#0ca678", 8: "#0ca678"}

    def __init__(self, cap_a=4, cap_b=3, initial=(0, 0), goal=(2, 0), cost="unit"):
        """goal = (ga, gb); None oznacza 'dowolna ilosc'."""
        self.A, self.B = cap_a, cap_b
        x, y = initial
        assert 0 <= x <= cap_a and 0 <= y <= cap_b
        self.goal = goal
        self.cost_model = cost
        super().__init__((x, y))
        self.node_w, self.node_h = 0.95, 0.5

    def all_states(self):
        return [(x, y) for x in range(self.A + 1) for y in range(self.B + 1)]

    def action_list(self):
        return tuple(range(1, 9))

    def applicable(self, s, a):
        X, Y, A, B = *s, self.A, self.B
        return {
            1: X < A, 2: Y < B, 3: X > 0, 4: Y > 0,
            5: X + Y >= A and X < A and Y > 0,
            6: X + Y >= B and X > 0 and Y < B,
            7: X > 0 and Y < B and X + Y <= B,
            8: X < A and Y > 0 and X + Y <= A,
        }[a]

    def result(self, s, a):
        X, Y, A, B = *s, self.A, self.B
        return {
            1: (A, Y), 2: (X, B), 3: (0, Y), 4: (X, 0),
            5: (A, X + Y - A), 6: (X + Y - B, B), 7: (0, X + Y), 8: (X + Y, 0),
        }[a]

    def is_goal(self, s):
        ga, gb = self.goal
        return (ga is None or s[0] == ga) and (gb is None or s[1] == gb)

    def action_cost(self, s, a, s2):
        if self.cost_model == "unit":
            return 1
        (x1, y1), (x2, y2) = s, s2
        return max(abs(x2 - x1), abs(y2 - y1))  # litry nalane / wylane / przelane

    def state_label(self, s):
        return f"({s[0]}, {s[1]})"

    def action_name(self, a):
        return f"{a}. {t(f'jug_r{a}')}"

    def action_short(self, a):
        return str(a)

    def action_color(self, a):
        return self.COLORS[a]

    def legend_actions(self):
        return [("1, 2", self.COLORS[1]), ("3, 4", self.COLORS[3]),
                ("5, 6", self.COLORS[5]), ("7, 8", self.COLORS[7])]

    def rule_text(self, a):
        A, B = self.A, self.B
        cond = {
            1: f"X &lt; {A}", 2: f"Y &lt; {B}", 3: "X &gt; 0", 4: "Y &gt; 0",
            5: f"X+Y ≥ {A}; X &lt; {A}; Y &gt; 0", 6: f"X+Y ≥ {B}; X &gt; 0; Y &lt; {B}",
            7: f"X &gt; 0; Y &lt; {B}; X+Y ≤ {B}", 8: f"X &lt; {A}; Y &gt; 0; X+Y ≤ {A}",
        }[a]
        eff = {
            1: f"({A}, Y)", 2: f"(X, {B})", 3: "(0, Y)", 4: "(X, 0)",
            5: f"({A}, X+Y−{A})", 6: f"(X+Y−{B}, {B})", 7: "(0, X+Y)", 8: "(X+Y, 0)",
        }[a]
        return cond, eff

    def layout(self, states):
        return {(x, y): (x * 2.4, y * 2.0) for x, y in states}

    def draw_state(self, ax, s):
        X, Y = s
        top = max(self.A, self.B)
        for i, (name, cap, val) in enumerate((("A", self.A, X), ("B", self.B, Y))):
            x0 = i * 1.6
            hcap = cap / top
            ax.add_patch(Rectangle((x0, 0), 1.0, hcap * val / cap if cap else 0, fc="#74c0fc", ec="none"))
            ax.add_patch(Rectangle((x0, 0), 1.0, hcap, fc="none", ec="#343a40", lw=2.5))
            ax.text(x0 + 0.5, -0.06, f"{name}: {val} / {cap} l", ha="center", va="top", fontsize=12,
                    weight="bold")
        ax.set_xlim(-0.2, 2.8)
        ax.set_ylim(-0.3, 1.08)
        ax.set_aspect("equal")
        ax.axis("off")

    def definition_rows(self):
        rows = "".join(
            f"<tr><td style='text-align:center'><b>{a}</b></td><td>(X, Y | {self.rule_text(a)[0]})</td>"
            f"<td>→ {self.rule_text(a)[1]}</td><td>{t(f'jug_r{a}')}</td></tr>"
            for a in self.action_list())
        table = (f"<table class='ls'><tr><th>{t('jug_rule')}</th><th>{t('jug_cond')}</th>"
                 f"<th>{t('jug_effect')}</th><th>{t('jug_desc')}</th></tr>{rows}</table>"
                 f"<div style='color:#868e96;font-size:90%;margin-top:4px'>{t('jug_note')}</div>")
        ga, gb = self.goal
        goal = f"X = {'*' if ga is None else ga}, Y = {'*' if gb is None else gb}"
        return [
            (t("states"), t("jug_states_desc", a=self.A, b=self.B)),
            (t("initial"), self.state_label(self.initial)),
            (t("goal"), goal + (f" &nbsp;(* = {t('jug_any')})" if None in self.goal else "")),
            (t("actions") + " / " + t("transition"), table),
            (t("cost"), t("jug_cost_unit_desc" if self.cost_model == "unit" else "jug_cost_water_desc")),
        ]


# --------------------------------------------------------------------------
# Winda - przyklad wprowadzajacy
# --------------------------------------------------------------------------
class Elevator(Problem):
    """Stan: (P, D) - pietro P (0 = parter) i drzwi D: 'o' otwarte / 'z' zamkniete.
    Reguly: 1 jedz w gore, 2 jedz w dol (tylko przy zamknietych drzwiach),
    3 otworz drzwi, 4 zamknij drzwi."""

    _S = {
        "pl": {1: "jedź w górę", 2: "jedź w dół", 3: "otwórz drzwi", 4: "zamknij drzwi",
               "states": "(P, D), gdzie P – piętro (0 = parter, …, {top}), D – drzwi: o = otwarte, z = zamknięte",
               "o": "o", "z": "z"},
        "en": {1: "go up", 2: "go down", 3: "open the door", 4: "close the door",
               "states": "(F, D), where F – floor (0 = ground, …, {top}), D – door: o = open, z = closed",
               "o": "o", "z": "z"},
    }
    COLORS = {1: "#1c7ed6", 2: "#1c7ed6", 3: "#2f9e44", 4: "#2f9e44"}

    def __init__(self, floors=3, initial=(0, "o"), goal=(2, "o")):
        self.floors = floors
        self.goal = goal
        super().__init__(initial)
        self.node_w, self.node_h = 0.95, 0.5

    def _s(self, k):
        return self._S[LANG][k]

    def all_states(self):
        return [(p, d) for p in range(self.floors) for d in ("z", "o")]

    def action_list(self):
        return (1, 2, 3, 4)

    def applicable(self, s, a):
        p, d = s
        return {1: d == "z" and p < self.floors - 1, 2: d == "z" and p > 0,
                3: d == "z", 4: d == "o"}[a]

    def result(self, s, a):
        p, d = s
        return {1: (p + 1, "z"), 2: (p - 1, "z"), 3: (p, "o"), 4: (p, "z")}[a]

    def is_goal(self, s):
        gp, gd = self.goal
        return (gp is None or s[0] == gp) and (gd is None or s[1] == gd)

    def sort_key(self, s):
        return (s[0], s[1] == "o")

    def state_label(self, s):
        return f"({s[0]}, {self._s(s[1])})"

    def action_name(self, a):
        return f"{a}. {self._s(a)}"

    def action_short(self, a):
        return str(a)

    def action_color(self, a):
        return self.COLORS[a]

    def legend_actions(self):
        return [(f"1, 2 = {self._s(1)} / {self._s(2)}", self.COLORS[1]),
                (f"3, 4 = {self._s(3)} / {self._s(4)}", self.COLORS[3])]

    def layout(self, states):
        return {(p, d): (p * 2.4, 1.8 if d == "o" else 0.0) for p, d in states}

    def draw_state(self, ax, s):
        p, d = s
        n = self.floors
        ax.add_patch(Rectangle((0, 0), 1.2, n, fc="#f8f9fa", ec="#495057", lw=2))
        for f in range(n):
            ax.plot([0, 1.2], [f, f], color="#adb5bd", lw=1)
            ax.text(-0.15, f + 0.5, str(f), ha="right", va="center", fontsize=12, weight="bold")
        ax.add_patch(Rectangle((0.15, p + 0.08), 0.9, 0.84, fc="#ced4da", ec="#343a40", lw=2))
        gap = 0.3 if d == "o" else 0.0
        for x0 in (0.15, 0.6 + gap / 2):
            ax.add_patch(Rectangle((x0, p + 0.08), 0.45 - gap / 2, 0.84, fc="#868e96", ec="#343a40", lw=1))
        ax.set_xlim(-0.6, 1.5)
        ax.set_ylim(-0.1, n + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")

    def definition_rows(self):
        top = self.floors - 1
        cond = {1: f"D = z; P &lt; {top}", 2: "D = z; P &gt; 0", 3: "D = z", 4: "D = o"}
        eff = {1: "(P+1, z)", 2: "(P−1, z)", 3: "(P, o)", 4: "(P, z)"}
        rows = "".join(f"<tr><td style='text-align:center'><b>{a}</b></td><td>(P, D | {cond[a]})</td>"
                       f"<td>→ {eff[a]}</td><td>{self._s(a)}</td></tr>" for a in self.action_list())
        table = (f"<table class='ls'><tr><th>{t('jug_rule')}</th><th>{t('jug_cond')}</th>"
                 f"<th>{t('jug_effect')}</th><th>{t('jug_desc')}</th></tr>{rows}</table>")
        gp, gd = self.goal
        return [
            (t("states"), self._s("states").format(top=top)),
            (t("initial"), self.state_label(self.initial)),
            (t("goal"), f"P = {'*' if gp is None else gp}, D = {'*' if gd is None else self._s(gd)}"),
            (t("actions") + " / " + t("transition"), table),
            (t("cost"), t("jug_cost_unit_desc")),
        ]


# --------------------------------------------------------------------------
# HTML: definicja formalna i tabela przejsc
# --------------------------------------------------------------------------
_CSS = """<style>
table.ls{border-collapse:collapse;font-size:13px;margin:4px 0;color:#212529;background:#ffffff}
table.ls td,table.ls th{border:1px solid #ced4da;padding:3px 8px;vertical-align:top;text-align:left;
  color:#212529;background:#ffffff}
table.ls th{background:#f1f3f5;color:#212529}
table.ls code{color:#212529;background:#f1f3f5}
table.ls tr.ini td{background:#ebfbee}
table.ls tr.goal td{background:#fff9db}
table.ls tr.unr td{color:#adb5bd}
</style>"""


def definition_html(space: StateSpace, solution=None) -> str:
    p = space.problem
    rows = list(p.definition_rows())
    goals = [space.label(s) for s in space.all_states if p.is_goal(s)]
    rows[2] = (rows[2][0], rows[2][1] + "<br><span style='color:#495057'>→ "
               + (html.escape(", ".join(goals)) if goals else t("no_goal_states")) + "</span>")
    rows.append((t("stats"), t("n_states", all=len(space.all_states), reach=len(space.reachable),
                               edges=space.n_edges_reachable())))
    if solution is None:
        sol = t("no_solution")
    else:
        steps, cost = solution
        seq = " → ".join([html.escape(p.state_label(p.initial))] +
                         [f"<i>{html.escape(p.action_short(a))}</i> → {html.escape(p.state_label(s2))}"
                          for _, a, s2 in steps])
        sol = (f"<b>{html.escape(p.solution_text(steps)) or '∅'}</b> "
               f"({n_steps(len(steps))}, {t('total_cost')} = {cost})<br>"
               f"<span style='font-size:90%'>{seq}</span>")
    rows.append((t("solution"), sol))
    body = "".join(f"<tr><th style='width:170px'>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return _CSS + f"<table class='ls'>{body}</table>"


def transition_table_html(space: StateSpace) -> str:
    p = space.problem
    acts = p.action_list()
    head = f"<tr><th>{t('state_col')}</th>" + "".join(
        f"<th title='{html.escape(p.action_name(a))}'>{html.escape(p.action_name(a))}</th>" for a in acts) + "</tr>"
    rows = []
    for s in space.states:
        cls = "ini" if s == p.initial else "goal" if p.is_goal(s) else "unr" if s not in space.reachable else ""
        cells = []
        for a in acts:
            if p.applicable(s, a):
                s2 = p.result(s, a)
                cells.append(f"<td>#{space.number[s2]} {html.escape(p.state_label(s2))}</td>")
            else:
                cells.append("<td style='color:#ced4da'>—</td>")
        rows.append(f"<tr class='{cls}'><td><b>{html.escape(space.label(s))}</b></td>{''.join(cells)}</tr>")
    return _CSS + f"<table class='ls'>{head}{''.join(rows)}</table>"


# --------------------------------------------------------------------------
# Funkcje "bez widgetow" (do uzycia w zwyklych komorkach)
# --------------------------------------------------------------------------
def show(problem: Problem, include_unreachable=False, show_solution=True, **kw):
    """Narysuj przestrzen stanow i wypisz definicje (bez interakcji)."""
    from IPython.display import HTML, display
    space = StateSpace(problem, include_unreachable)
    sol = solve(problem)
    display(HTML(definition_html(space, sol)))
    fig = draw_space(space, solution=sol[0] if (sol and show_solution) else None, **kw)
    display(fig)
    plt.close(fig)


def show_table(problem: Problem, include_unreachable=False):
    from IPython.display import HTML, display
    display(HTML(transition_table_html(StateSpace(problem, include_unreachable))))


# --------------------------------------------------------------------------
# Aplikacje interaktywne (ipywidgets)
# --------------------------------------------------------------------------
class _App:
    title_key = ""

    def __init__(self):
        import ipywidgets as W
        self.W = W
        self._busy = False
        self.history = []
        self.out_graph = W.Output()
        self.out_state = W.Output()
        self.out_def = W.Output()
        self.out_table = W.Output()
        self.status = W.HTML()
        self.warn = W.HTML()
        self.path_html = W.HTML()

        self.cb_solution = W.Checkbox(value=False, description=t("show_solution"), indent=False)
        self.cb_all = W.Checkbox(value=False, description=t("show_all"), indent=False)
        self.cb_labels = W.Checkbox(value=True, description=t("edge_labels"), indent=False)
        self.cb_loops = W.Checkbox(value=self._default_loops(), description=t("self_loops"), indent=False)
        for cb in (self.cb_solution, self.cb_labels, self.cb_loops):
            cb.observe(lambda _: self._render_graph(), names="value")
        self.cb_all.observe(lambda _: self._rebuild_space(), names="value")

        self.btn_undo = W.Button(description=t("undo"), layout=W.Layout(width="auto"))
        self.btn_reset = W.Button(description=t("reset"), layout=W.Layout(width="auto"))
        self.btn_undo.on_click(self._undo)
        self.btn_reset.on_click(lambda _: self._reset_play())
        self.action_box = W.HBox(layout=W.Layout(flex_flow="row wrap"))

        self.controls = self._build_controls()
        opts = W.HBox([self.cb_solution, self.cb_all, self.cb_labels, self.cb_loops],
                      layout=W.Layout(flex_flow="row wrap", grid_gap="12px"))
        play = W.VBox([
            W.HTML(f"<b>{t('current')}</b>"), self.out_state, self.action_box,
            W.HBox([self.btn_undo, self.btn_reset]), self.status, self.path_html,
        ], layout=W.Layout(width="420px"))
        tabs = W.Tab(children=[W.VBox([W.HBox([play, W.VBox([opts])]), self.out_graph]),
                               self.out_def, self.out_table])
        for i, k in enumerate(("tab_graph", "tab_def", "tab_table")):
            tabs.set_title(i, t(k))
        self.widget = W.VBox([W.HTML(f"<h3 style='margin:4px 0'>{t(self.title_key)}</h3>"),
                              self.controls, self.warn, tabs])
        self.rebuild()

    # --- do nadpisania ---
    def _default_loops(self):
        return False

    def _build_controls(self):
        raise NotImplementedError

    def make_problem(self) -> Problem:
        raise NotImplementedError

    # --- logika ---
    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def rebuild(self, *_):
        if self._busy:
            return
        self.problem = self.make_problem()
        self.solution = solve(self.problem)
        self._rebuild_space()

    def _rebuild_space(self):
        self.space = StateSpace(self.problem, self.cb_all.value)
        W = self.W
        self.buttons = {}
        for a in self.problem.action_list():
            b = W.Button(description=self.problem.action_name(a), tooltip=self.problem.action_name(a),
                         layout=W.Layout(width="auto", margin="2px"))
            b.on_click(lambda _, a=a: self._do(a))
            self.buttons[a] = b
        self.action_box.children = list(self.buttons.values())
        with self.out_def:
            self.out_def.clear_output(wait=True)
            from IPython.display import HTML, display
            display(HTML(definition_html(self.space, self.solution)))
        with self.out_table:
            self.out_table.clear_output(wait=True)
            from IPython.display import HTML, display
            display(HTML(transition_table_html(self.space)))
        self._reset_play()

    def _current(self):
        return self.history[-1][2] if self.history else self.problem.initial

    def _do(self, a):
        s = self._current()
        if not self.problem.applicable(s, a):
            return
        self.history.append((s, a, self.problem.result(s, a)))
        self._render_play()

    def _undo(self, _=None):
        if self.history:
            self.history.pop()
            self._render_play()

    def _reset_play(self):
        self.history = []
        self._render_play()

    def _render_play(self):
        p, s = self.problem, self._current()
        for a, b in self.buttons.items():
            ok = p.applicable(s, a)
            b.disabled = not ok
            b.tooltip = p.action_name(a) + ("" if ok else f" – {t('not_applicable')}")
        with self.out_state:
            self.out_state.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(4.0, 1.9))
            p.draw_state(ax, s)
            ax.set_title(self.space.label(s), fontsize=10)
            fig.tight_layout()
            from IPython.display import display
            display(fig)
            plt.close(fig)
        cost = sum(p.action_cost(*st) for st in self.history)
        self.status.value = (f"<span style='color:{C_INITIAL};font-size:16px'><b>{t('goal_reached')}</b></span>"
                             if p.is_goal(s) else "")
        seq = " – ".join(p.action_short(a) for _, a, _ in self.history) or t("path_empty")
        self.path_html.value = (f"<b>{t('path')}:</b> {html.escape(seq)}<br>"
                                f"{n_steps(len(self.history))}, {t('total_cost')} = <b>{cost}</b>")
        self._render_graph()

    def _render_graph(self):
        sol = self.solution[0] if (self.solution and self.cb_solution.value) else None
        with self.out_graph:
            self.out_graph.clear_output(wait=True)
            fig = draw_space(self.space, current=self._current(), trail=self.history, solution=sol,
                             edge_labels=self.cb_labels.value, self_loops=self.cb_loops.value)
            from IPython.display import display
            display(fig)
            plt.close(fig)


class VacuumApp(_App):
    title_key = "vac_title"

    def _default_loops(self):
        return True

    def _build_controls(self):
        W = self.W
        st = {"description_width": "initial"}
        self.w_n = W.IntSlider(value=2, min=1, max=4, description=t("vac_n"), style=st)
        self.w_k = W.IntSlider(value=2, min=0, max=2, description=t("vac_maxdirty"), style=st)
        self.w_pos = W.Dropdown(options=self._pos_options(2), value=0, description=t("vac_pos"), style=st,
                                layout=W.Layout(width="180px"))
        self.w_move = W.IntSlider(value=1, min=0, max=5, description=t("vac_move_cost"), style=st)
        self.w_suck = W.IntSlider(value=1, min=0, max=5, description=t("vac_suck_cost"), style=st)
        self.w_noop = W.Checkbox(value=False, description=t("vac_noop"), indent=False)
        self.dirt_box = W.HBox()
        self._make_dirt_boxes(2)
        self.w_n.observe(self._on_n, names="value")
        for w in (self.w_k, self.w_pos, self.w_move, self.w_suck, self.w_noop):
            w.observe(self.rebuild, names="value")
        return W.VBox([
            W.HBox([self.w_n, self.w_k, self.w_pos]),
            W.HBox([W.Label(t("vac_dirt") + ":"), self.dirt_box]),
            W.HBox([self.w_move, self.w_suck, self.w_noop]),
        ])

    @staticmethod
    def _pos_options(n):
        return [(VacuumWorld.cell_name(i), i) for i in range(n)]

    def _make_dirt_boxes(self, n):
        W = self.W
        self.w_dirt = [W.Checkbox(value=True, description=VacuumWorld.cell_name(i), indent=False,
                                  layout=W.Layout(width="60px")) for i in range(n)]
        for cb in self.w_dirt:
            cb.observe(self.rebuild, names="value")
        self.dirt_box.children = self.w_dirt

    def _on_n(self, change):
        n = change["new"]
        self._busy = True
        try:
            self.w_k.max = n
            self.w_k.value = n
            self.w_pos.options = self._pos_options(n)
            self.w_pos.value = 0
            self._make_dirt_boxes(n)
        finally:
            self._busy = False
        self.rebuild()

    def make_problem(self):
        dirt = tuple(int(cb.value) for cb in self.w_dirt)
        k = self.w_k.value
        self.warn.value = (f"<span style='color:#e8590c'>{t('vac_warn_dirt')}</span>"
                           if sum(dirt) > k else "")
        return VacuumWorld(n=self.w_n.value, initial=(self.w_pos.value, dirt), max_dirty=k,
                           move_cost=self.w_move.value, suck_cost=self.w_suck.value, noop=self.w_noop.value)


class WaterJugsApp(_App):
    title_key = "jug_title"

    def _build_controls(self):
        W = self.W
        st = {"description_width": "initial"}
        self.w_ca = W.IntSlider(value=4, min=1, max=9, description=t("jug_capA"), style=st)
        self.w_cb = W.IntSlider(value=3, min=1, max=9, description=t("jug_capB"), style=st)
        self.w_ia = W.IntSlider(value=0, min=0, max=4, description=t("jug_initA"), style=st)
        self.w_ib = W.IntSlider(value=0, min=0, max=3, description=t("jug_initB"), style=st)
        self.w_ga = W.Dropdown(options=self._goal_opts(4), value=2, description=t("jug_goalA"), style=st,
                               layout=W.Layout(width="200px"))
        self.w_gb = W.Dropdown(options=self._goal_opts(3), value=0, description=t("jug_goalB"), style=st,
                               layout=W.Layout(width="200px"))
        self.w_cost = W.Dropdown(options=[(t("jug_cost_unit"), "unit"), (t("jug_cost_water"), "water")],
                                 value="unit", description=t("jug_cost"), style=st)
        self.w_ca.observe(self._on_cap, names="value")
        self.w_cb.observe(self._on_cap, names="value")
        for w in (self.w_ia, self.w_ib, self.w_ga, self.w_gb, self.w_cost):
            w.observe(self.rebuild, names="value")
        return W.VBox([W.HBox([self.w_ca, self.w_cb]), W.HBox([self.w_ia, self.w_ib]),
                       W.HBox([self.w_ga, self.w_gb, self.w_cost])])

    @staticmethod
    def _goal_opts(cap):
        return [(t("jug_any"), None)] + [(str(i), i) for i in range(cap + 1)]

    def _on_cap(self, _):
        self._busy = True
        try:
            for slider, goal, cap in ((self.w_ia, self.w_ga, self.w_ca.value), (self.w_ib, self.w_gb, self.w_cb.value)):
                slider.max = cap
                old = goal.value
                goal.options = self._goal_opts(cap)
                goal.value = old if (old is None or old <= cap) else None
        finally:
            self._busy = False
        self.rebuild()

    def make_problem(self):
        return WaterJugs(self.w_ca.value, self.w_cb.value, initial=(self.w_ia.value, self.w_ib.value),
                         goal=(self.w_ga.value, self.w_gb.value), cost=self.w_cost.value)


def vacuum_app():
    """Interaktywny swiat odkurzacza."""
    return VacuumApp()


def water_jugs_app():
    """Interaktywny problem pojemnikow."""
    return WaterJugsApp()
