"""
lab_search.py - interaktywne przeszukiwanie przestrzeni stanow (BFS, DFS, Dijkstra)
oraz zastosowania w analizie glowic uwagi (Attention Heads) w modelach Transformer.

Moduł do zajęć "Wprowadzenie do AI" (kognitywistyka, UMCS) – laboratorium 3.
Korzysta z definicji problemow (Elevator, VacuumWorld, WaterJugs, Problem, StateSpace)
z modulu lab_states.py (laboratorium 2).

Szybki start w notebooku:
    import lab_search as lsearch
    lsearch.dijkstra_vs_bfs_demo()       # powtorka: Dijkstra (wagi=1) to BFS
    lsearch.search_app()                 # glowny symulator BFS / DFS / Dijkstra
    lsearch.recursion_vs_iteration_app() # porownanie iteracji (stos/kolejka) i rekurencji (call stack)
    lsearch.attention_traversal_app()    # przeszukiwanie drzewa uwagi (Attention Heads) w Transformerze
"""
from __future__ import annotations

import heapq
import html
import itertools
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

# Upewnij sie, ze lab02_stany jest w sys.path (zarowno lokalnie, jak i w Colabie)
_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB02 = os.path.normpath(os.path.join(_HERE, "..", "lab02_stany"))
if os.path.isdir(_LAB02) and _LAB02 not in sys.path:
    sys.path.insert(0, _LAB02)

import lab_states as ls
from lab_states import (
    C_CURRENT,
    C_GOAL_FILL,
    C_INITIAL,
    C_NODE,
    C_NODE_EDGE,
    C_SOLUTION,
    C_TRAIL,
    C_UNREACH,
     Elevator,
    Problem,
    StateSpace,
    VacuumWorld,
    WaterJugs,
    _bezier_mid,
    _CSS,
    n_steps,
)

# Dodatkowe kolory dla wizualizacji przeszukiwania
C_VISITED_FILL = "#d0ebff"      # jasnoniebieski - stany odwiedzone (zamkniete)
C_VISITED_EDGE = "#1971c2"
C_FRONTIER_FILL = "#f3d9fa"     # jasnofioletowy - stany w kolejce / na stosie (otwarte)
C_FRONTIER_EDGE = "#9c36b5"
C_CALLSTACK_FILL = "#fff3bf"    # jasnozolty - aktywne wywolania na stosie rekurencji
C_CALLSTACK_EDGE = "#f08c00"
C_TREE_EDGE = "#1971c2"         # krawedz drzewa przeszukiwania
C_ACTIVE_EDGE = "#e03131"       # wlasnie sprawdzana krawedz


# --------------------------------------------------------------------------
# Struktura kroku przeszukiwania (SearchStep)
# --------------------------------------------------------------------------
@dataclass
class SearchStep:
    step_idx: int
    phase: str                       # 'init', 'pop', 'neighbor', 'skip', 'backtrack', 'goal', 'done'
    current: Any = None              # aktualnie przetwarzany stan
    active_edge: tuple | None = None # (u, a, v) wlasnie analizowane przejscie
    visited: list = field(default_factory=list)          # lista stanow w kolejnosci odwiedzenia
    frontier: list = field(default_factory=list)         # [(stan, rodzic, akcja, glebokosc, koszt), ...]
    call_stack: list = field(default_factory=list)       # [{'func': str, 'state': s, 'depth': int, 'note': str}, ...]
    tree_edges: list = field(default_factory=list)       # [(u, a, v), ...] krawedzie drzewa odkryc
    depth_map: dict = field(default_factory=dict)        # stan -> glebokosc (liczba krokow od startu)
    cost_map: dict = field(default_factory=dict)         # stan -> koszt g(s)
    solution: list | None = None                         # [(u, a, v), ...] jezeli osiagnieto cel
    code_line: int = 0                                   # podswietlana linia pseudokodu/kodu
    title: str = ""
    message: str = ""


def _reconstruct_path(parent: dict, s) -> list:
    steps = []
    while parent.get(s) is not None:
        prev, a = parent[s]
        steps.append((prev, a, s))
        s = prev
    return steps[::-1]


# --------------------------------------------------------------------------
# Algorytmy przeszukiwania generujace pelny slad (trace)
# --------------------------------------------------------------------------
def trace_bfs_iterative(problem: Problem, detailed: bool = False, stop_on_goal: bool = True) -> list[SearchStep]:
    """Iteracyjny BFS z uzyciem kolejki FIFO (collections.deque)."""
    start = problem.initial
    visited: list = []
    visited_set: set = {start}
    parent: dict = {start: None}
    depth_map: dict = {start: 0}
    cost_map: dict = {start: 0}
    tree_edges: list = []
    # Element kolejki: (stan, rodzic, akcja, glebokosc, koszt)
    queue = deque([(start, None, None, 0, 0)])
    steps: list[SearchStep] = []

    def snap(phase, current=None, active_edge=None, solution=None, code_line=0, title="", message=""):
        steps.append(SearchStep(
            step_idx=len(steps),
            phase=phase,
            current=current,
            active_edge=active_edge,
            visited=list(visited),
            frontier=list(queue),
            call_stack=[],
            tree_edges=list(tree_edges),
            depth_map=dict(depth_map),
            cost_map=dict(cost_map),
            solution=list(solution) if solution is not None else None,
            code_line=code_line,
            title=title,
            message=message,
        ))

    snap(
        "init",
        current=None,
        code_line=2,
        title="Inicjalizacja kolejki FIFO",
        message=f"Wstawiamy stan początkowy <b>{html.escape(problem.state_label(start))}</b> do kolejki FIFO (warstwa d=0).",
    )

    while queue:
        s, par, act, d, c = queue.popleft()
        visited.append(s)
        lbl = html.escape(problem.state_label(s))

        if problem.is_goal(s) and stop_on_goal:
            sol = _reconstruct_path(parent, s)
            snap(
                "goal",
                current=s,
                solution=sol,
                code_line=6,
                title=f"🎉 Cel osiągnięty: {problem.state_label(s)}!",
                message=(f"Pobrano z przodu kolejki FIFO (<code>popleft()</code>) stan <b>{lbl}</b> (głębokość d={d}). "
                         f"To stan docelowy! Znaleziono najkrótszą ścieżkę ({n_steps(len(sol))}, koszt={c})."),
            )
            return steps

        if detailed:
            snap(
                "pop",
                current=s,
                code_line=5,
                title=f"Pobranie z początku kolejki: {problem.state_label(s)} (d={d})",
                message=(f"<code>queue.popleft()</code> → pobieramy stan <b>{lbl}</b> z przodu kolejki FIFO "
                         f"(odwiedzony jako #{len(visited)}, warstwa d={d})."),
            )

        added_now = []
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            a_lbl = html.escape(problem.action_short(a))
            s2_lbl = html.escape(problem.state_label(s2))
            if s2 not in visited_set:
                visited_set.add(s2)
                parent[s2] = (s, a)
                nd = d + 1
                nc = c + problem.action_cost(s, a, s2)
                depth_map[s2] = nd
                cost_map[s2] = nc
                tree_edges.append((s, a, s2))
                queue.append((s2, s, a, nd, nc))
                added_now.append(f"{s2_lbl} (przez {a_lbl})")
                if detailed:
                    snap(
                        "neighbor",
                        current=s,
                        active_edge=(s, a, s2),
                        code_line=10,
                        title=f"Dopisanie na koniec kolejki: {problem.state_label(s2)}",
                        message=(f"Z <b>{lbl}</b> akcja <b>{a_lbl}</b> prowadzi do nowego stanu <b>{s2_lbl}</b>. "
                                 f"<code>queue.append({s2_lbl})</code> – dopisujemy na <b>koniec</b> kolejki (d={nd})."),
                    )
            else:
                if detailed and s2 != s:
                    snap(
                        "skip",
                        current=s,
                        active_edge=(s, a, s2),
                        code_line=8,
                        title=f"Pominięcie znanego stanu: {problem.state_label(s2)}",
                        message=(f"Z <b>{lbl}</b> akcja <b>{a_lbl}</b> prowadzi do <b>{s2_lbl}</b>, "
                                 f"który był już wcześniej odkryty – pomijamy (unikamy cyklu!)."),
                    )

        if not detailed:
            desc_added = ", ".join(added_now) if added_now else "brak nowych sąsiadów"
            snap(
                "pop",
                current=s,
                code_line=10,
                title=f"Krok #{len(visited)}: rozwinięcie {problem.state_label(s)} (warstwa d={d})",
                message=(f"Pobrano z przodu kolejki <b>{lbl}</b> (d={d}). "
                         f"Dopisano na koniec kolejki: <b>{desc_added}</b>."),
            )

    snap(
        "done",
        current=None,
        code_line=11,
        title="Kolejka pusta – koniec przeszukiwania",
        message="Kolejka FIFO jest pusta. Wszystkie osiągalne stany zostały odwiedzone.",
    )
    return steps


def trace_dfs_iterative(problem: Problem, detailed: bool = False, stop_on_goal: bool = True) -> list[SearchStep]:
    """Iteracyjny DFS z uzyciem jawnego stosu LIFO (list.append / list.pop)."""
    start = problem.initial
    visited: list = []
    visited_set: set = set()
    parent: dict = {start: None}
    depth_map: dict = {start: 0}
    cost_map: dict = {start: 0}
    tree_edges: list = []
    # Element stosu: (stan, rodzic, akcja, glebokosc, koszt)
    stack = [(start, None, None, 0, 0)]
    steps: list[SearchStep] = []

    def snap(phase, current=None, active_edge=None, solution=None, code_line=0, title="", message=""):
        steps.append(SearchStep(
            step_idx=len(steps),
            phase=phase,
            current=current,
            active_edge=active_edge,
            visited=list(visited),
            frontier=list(stack),
            call_stack=[],
            tree_edges=list(tree_edges),
            depth_map=dict(depth_map),
            cost_map=dict(cost_map),
            solution=list(solution) if solution is not None else None,
            code_line=code_line,
            title=title,
            message=message,
        ))

    snap(
        "init",
        current=None,
        code_line=2,
        title="Inicjalizacja stosu LIFO",
        message=f"Kładziemy stan początkowy <b>{html.escape(problem.state_label(start))}</b> na wierzch stosu LIFO (d=0).",
    )

    while stack:
        s, par, act, d, c = stack.pop()
        lbl = html.escape(problem.state_label(s))
        if s in visited_set:
            if detailed:
                snap(
                    "skip",
                    current=s,
                    code_line=6,
                    title=f"Zdjęto ze stosu już odwiedzony stan: {problem.state_label(s)}",
                    message=f"Stan <b>{lbl}</b> był już wcześniej odwiedzony – pomijamy go.",
                )
            continue

        visited_set.add(s)
        visited.append(s)
        if par is not None:
            parent[s] = (par, act)
            depth_map[s] = d
            cost_map[s] = c
            tree_edges.append((par, act, s))

        if problem.is_goal(s) and stop_on_goal:
            sol = _reconstruct_path(parent, s)
            snap(
                "goal",
                current=s,
                solution=sol,
                code_line=8,
                title=f"🎉 Cel osiągnięty: {problem.state_label(s)}!",
                message=(f"Zdjęto z wierzchu stosu LIFO (<code>stack.pop()</code>) stan <b>{lbl}</b> (głębokość d={d}). "
                         f"To stan docelowy! Znaleziona ścieżka ma {n_steps(len(sol))} (koszt={c})."),
            )
            return steps

        if detailed:
            snap(
                "pop",
                current=s,
                code_line=5,
                title=f"Zdjęcie z wierzchu stosu: {problem.state_label(s)} (d={d})",
                message=(f"<code>stack.pop()</code> → zdejmujemy z <b>wierzchu stosu</b> stan <b>{lbl}</b> "
                         f"(odwiedzony jako #{len(visited)}, głębokość d={d})."),
            )

        # Odwracamy kolejnosc akcji przy odkladaniu na stos, zeby pierwszy sasiad znalazl sie na samym wierzchu!
        actions = list(problem.actions(s))
        pushed_now = []
        for a in reversed(actions):
            s2 = problem.result(s, a)
            a_lbl = html.escape(problem.action_short(a))
            s2_lbl = html.escape(problem.state_label(s2))
            if s2 not in visited_set:
                nd = d + 1
                nc = c + problem.action_cost(s, a, s2)
                if s2 not in depth_map:
                    depth_map[s2] = nd
                    cost_map[s2] = nc
                stack.append((s2, s, a, nd, nc))
                pushed_now.append(f"{s2_lbl} (przez {a_lbl})")
                if detailed:
                    snap(
                        "neighbor",
                        current=s,
                        active_edge=(s, a, s2),
                        code_line=11,
                        title=f"Położenie na wierzch stosu: {problem.state_label(s2)}",
                        message=(f"Z <b>{lbl}</b> akcja <b>{a_lbl}</b> prowadzi do <b>{s2_lbl}</b>. "
                                 f"<code>stack.append({s2_lbl})</code> – odkładamy na <b>wierzch stosu</b> (d={nd})."),
                    )

        if not detailed:
            # W opisie pokazujemy od gory stosu
            desc_pushed = ", ".join(reversed(pushed_now)) if pushed_now else "ślepy zaułek (cofanie / backtrack!)"
            snap(
                "pop",
                current=s,
                code_line=11,
                title=f"Krok #{len(visited)}: wejście w głąb do {problem.state_label(s)} (d={d})",
                message=(f"Zdjęto z wierzchu stosu <b>{lbl}</b> (d={d}). "
                         f"Na wierzch stosu trafiły: <b>{desc_pushed}</b>."),
            )

    snap(
        "done",
        current=None,
        code_line=12,
        title="Stos pusty – koniec przeszukiwania",
        message="Stos LIFO jest pusty. Wszystkie osiągalne stany zostały odwiedzone.",
    )
    return steps


def trace_dfs_recursive(problem: Problem, detailed: bool = True, stop_on_goal: bool = True) -> list[SearchStep]:
    """Rekurencyjny DFS pokazujacy jawnie stos wywolan funkcji (Call Stack)."""
    start = problem.initial
    visited: list = []
    visited_set: set = set()
    parent: dict = {start: None}
    depth_map: dict = {start: 0}
    cost_map: dict = {start: 0}
    tree_edges: list = []
    call_stack: list[dict] = []
    steps: list[SearchStep] = []
    found_goal = False

    def snap(phase, current=None, active_edge=None, solution=None, code_line=0, title="", message=""):
        # Jako 'frontier' przekazujemy stan stosu wywolan, zeby podglady byly spojne
        frontier_from_stack = [
            (frame["state"], frame.get("parent"), frame.get("action"), frame["depth"], cost_map.get(frame["state"], 0))
            for frame in call_stack
        ]
        steps.append(SearchStep(
            step_idx=len(steps),
            phase=phase,
            current=current,
            active_edge=active_edge,
            visited=list(visited),
            frontier=frontier_from_stack,
            call_stack=[dict(f) for f in call_stack],
            tree_edges=list(tree_edges),
            depth_map=dict(depth_map),
            cost_map=dict(cost_map),
            solution=list(solution) if solution is not None else None,
            code_line=code_line,
            title=title,
            message=message,
        ))

    snap(
        "init",
        current=None,
        code_line=1,
        title="Start DFS rekurencyjnego",
        message=(f"Za chwilę wywołamy funkcję <code>dfs({html.escape(problem.state_label(start))})</code>. "
                 f"Kolejne wywołania będą odkładane na <b>stosie wywołań (Call Stack)</b> w pamięci programu."),
    )

    def _dfs(s, par, act, d, c):
        nonlocal found_goal
        if found_goal:
            return
        visited_set.add(s)
        visited.append(s)
        depth_map[s] = d
        cost_map[s] = c
        if par is not None:
            parent[s] = (par, act)
            tree_edges.append((par, act, s))

        lbl = html.escape(problem.state_label(s))
        call_stack.append({
            "func": f"dfs({problem.state_label(s)})",
            "state": s,
            "parent": par,
            "action": act,
            "depth": d,
            "status": "active",
            "note": f"głęb. d={d}",
        })

        if problem.is_goal(s) and stop_on_goal:
            found_goal = True
            sol = _reconstruct_path(parent, s)
            call_stack[-1]["status"] = "goal"
            snap(
                "goal",
                current=s,
                solution=sol,
                code_line=4,
                title=f"🎉 Cel osiągnięty w wywołaniu dfs({problem.state_label(s)})!",
                message=(f"Wywołanie <code>dfs({lbl})</code> (na głębokości d={d}, wysokość stosu wywołań = {len(call_stack)}) "
                         f"trafiło na stan docelowy! Ścieżka na stosie wywołań to nasze rozwiązanie ({n_steps(len(sol))})."),
            )
            return

        snap(
            "pop",
            current=s,
            active_edge=(par, act, s) if par is not None else None,
            code_line=2,
            title=f"▶ Wywołanie dfs({problem.state_label(s)}) [głęb. d={d}]",
            message=(f"Wchodzimy do funkcji <code>dfs({lbl})</code>. Na <b>stos wywołań</b> trafia nowa ramka "
                     f"(wysokość stosu: <b>{len(call_stack)}</b>). Oznaczamy stan jako odwiedzony (#{len(visited)})."),
        )

        for a in problem.actions(s):
            if found_goal:
                return
            s2 = problem.result(s, a)
            a_lbl = html.escape(problem.action_short(a))
            s2_lbl = html.escape(problem.state_label(s2))
            if s2 not in visited_set:
                call_stack[-1]["note"] = f"czeka na dfs({problem.state_label(s2)})"
                nc = c + problem.action_cost(s, a, s2)
                _dfs(s2, s, a, d + 1, nc)
                if found_goal:
                    return
                call_stack[-1]["status"] = "active"
                call_stack[-1]["note"] = f"wznowiono (d={d})"
                if detailed:
                    snap(
                        "backtrack",
                        current=s,
                        active_edge=(s, a, s2),
                        code_line=7,
                        title=f"↩ Powrót (backtrack) do dfs({problem.state_label(s)})",
                        message=(f"Funkcja <code>dfs({s2_lbl})</code> zakończyła działanie (<code>return</code>). "
                                 f"Zdejmujemy jej ramkę ze stosu wywołań i wracamy do <code>dfs({lbl})</code> (d={d})."),
                    )
            else:
                if detailed and s2 != s:
                    snap(
                        "skip",
                        current=s,
                        active_edge=(s, a, s2),
                        code_line=6,
                        title=f"Pominięcie odwiedzonego sąsiada: {problem.state_label(s2)}",
                        message=(f"W <code>dfs({lbl})</code> sprawdzamy sąsiada <b>{s2_lbl}</b> (akcja {a_lbl}) – "
                                 f"był już odwiedzony, więc nie wywołujemy dla niego rekurencji."),
                    )

        call_stack.pop()

    _dfs(start, None, None, 0, 0)
    if not found_goal:
        snap(
            "done",
            current=None,
            code_line=9,
            title="Koniec rekurencji – stos wywołań pusty",
            message="Główne wywołanie <code>dfs()</code> zakończyło się. Stos wywołań jest pusty.",
        )
    return steps


def trace_bfs_recursive(problem: Problem, detailed: bool = False, stop_on_goal: bool = True) -> list[SearchStep]:
    """Rekurencyjny BFS dzialajacy warstwa po warstwie: bfs_level(obecna_warstwa, d)."""
    start = problem.initial
    visited: list = []
    visited_set: set = {start}
    parent: dict = {start: None}
    depth_map: dict = {start: 0}
    cost_map: dict = {start: 0}
    tree_edges: list = []
    call_stack: list[dict] = []
    steps: list[SearchStep] = []
    found_goal = False

    def snap(phase, current=None, active_edge=None, frontier_list=(), solution=None, code_line=0, title="", message=""):
        steps.append(SearchStep(
            step_idx=len(steps),
            phase=phase,
            current=current,
            active_edge=active_edge,
            visited=list(visited),
            frontier=list(frontier_list),
            call_stack=[dict(f) for f in call_stack],
            tree_edges=list(tree_edges),
            depth_map=dict(depth_map),
            cost_map=dict(cost_map),
            solution=list(solution) if solution is not None else None,
            code_line=code_line,
            title=title,
            message=message,
        ))

    snap(
        "init",
        current=None,
        frontier_list=[(start, None, None, 0, 0)],
        code_line=1,
        title="Start BFS rekurencyjnego (warstwowego)",
        message=(f"Zamiast pętli <code>while</code> wywołujemy rekurencyjnie <code>bfs_warstwa([start], d=0)</code>. "
                 f"Każde wywołanie przetwarza całą warstwę na głębokości <code>d</code> i buduje listę stanów dla warstwy <code>d+1</code>."),
    )

    def _bfs_layer(layer: list, d: int):
        nonlocal found_goal
        if not layer or found_goal:
            return
        names = ", ".join(problem.state_label(item[0]) for item in layer)
        call_stack.append({
            "func": f"bfs_warstwa(d={d})",
            "state": layer[0][0],
            "depth": d,
            "status": "active",
            "note": f"stany ({len(layer)}): {names}",
        })

        snap(
            "pop",
            current=layer[0][0],
            frontier_list=layer,
            code_line=2,
            title=f"▶ Wywołanie rekurencyjne bfs_warstwa(d={d}) [{len(layer)} st.]",
            message=(f"Wchodzimy do <code>bfs_warstwa(d={d})</code>. Przetwarzamy wszystkie stany na głębokości d={d}: "
                     f"<b>{html.escape(names)}</b> i zbieramy ich sąsiadów do warstwy d={d+1}."),
        )

        next_layer = []
        for idx, (s, par, act, _, c) in enumerate(layer):
            visited.append(s)
            lbl = html.escape(problem.state_label(s))
            if problem.is_goal(s) and stop_on_goal:
                found_goal = True
                sol = _reconstruct_path(parent, s)
                call_stack[-1]["status"] = "goal"
                snap(
                    "goal",
                    current=s,
                    frontier_list=layer[idx + 1 :] + next_layer,
                    solution=sol,
                    code_line=5,
                    title=f"🎉 Cel osiągnięty w warstwie d={d}: {problem.state_label(s)}!",
                    message=(f"W wywołaniu <code>bfs_warstwa(d={d})</code> natrafiliśmy na stan docelowy <b>{lbl}</b>! "
                             f"Rozwiązanie ma {n_steps(len(sol))} (koszt={c})."),
                )
                return

            for a in problem.actions(s):
                s2 = problem.result(s, a)
                if s2 not in visited_set:
                    visited_set.add(s2)
                    parent[s2] = (s, a)
                    nd = d + 1
                    nc = c + problem.action_cost(s, a, s2)
                    depth_map[s2] = nd
                    cost_map[s2] = nc
                    tree_edges.append((s, a, s2))
                    next_layer.append((s2, s, a, nd, nc))

            if detailed:
                snap(
                    "neighbor",
                    current=s,
                    frontier_list=layer[idx + 1 :] + next_layer,
                    code_line=8,
                    title=f"Przetworzono {problem.state_label(s)} z warstwy d={d}",
                    message=(f"Sprawdzono sąsiadów stanu <b>{lbl}</b>. "
                             f"Nowa warstwa d={d+1} liczy teraz {len(next_layer)} stan(ów)."),
                )

        call_stack[-1]["note"] = f"wywołuje d={d+1}"
        _bfs_layer(next_layer, d + 1)
        if not found_goal:
            call_stack.pop()

    _bfs_layer([(start, None, None, 0, 0)], 0)
    if not found_goal:
        snap(
            "done",
            current=None,
            code_line=10,
            title="Koniec BFS rekurencyjnego",
            message="Kolejna warstwa jest pusta (warunek bazowy rekurencji) – wszystkie osiągalne stany zostały odwiedzone.",
        )
    return steps


def trace_dijkstra(problem: Problem, detailed: bool = False, stop_on_goal: bool = True) -> list[SearchStep]:
    """Przeszukiwanie z jednolitym kosztem (Dijkstra / UCS) z kolejka priorytetowa."""
    start = problem.initial
    visited: list = []
    visited_set: set = set()
    parent: dict = {start: None}
    depth_map: dict = {start: 0}
    cost_map: dict = {start: 0}
    tree_edges: list = []
    counter = itertools.count()
    # kopiec: (koszt_g, licznik, stan, rodzic, akcja, glebokosc)
    heap = [(0, next(counter), start, None, None, 0)]
    steps: list[SearchStep] = []

    def heap_as_frontier():
        seen = set()
        out = []
        for c, _, s, par, act, d in sorted(heap):
            if s not in visited_set and s not in seen:
                seen.add(s)
                out.append((s, par, act, d, c))
        return out

    def snap(phase, current=None, active_edge=None, solution=None, code_line=0, title="", message=""):
        steps.append(SearchStep(
            step_idx=len(steps),
            phase=phase,
            current=current,
            active_edge=active_edge,
            visited=list(visited),
            frontier=heap_as_frontier(),
            call_stack=[],
            tree_edges=list(tree_edges),
            depth_map=dict(depth_map),
            cost_map=dict(cost_map),
            solution=list(solution) if solution is not None else None,
            code_line=code_line,
            title=title,
            message=message,
        ))

    snap(
        "init",
        current=None,
        code_line=2,
        title="Inicjalizacja kolejki priorytetowej (Dijkstra / UCS)",
        message=(f"Wstawiamy stan początkowy <b>{html.escape(problem.state_label(start))}</b> z kosztem <code>g=0</code> "
                 f"do kolejki priorytetowej. Zauważ: gdy każda akcja kosztuje 1, koszt <code>g(s)</code> jest równy głębokości <code>d(s)</code>!"),
    )

    while heap:
        c, _, s, par, act, d = heapq.heappop(heap)
        if s in visited_set:
            continue
        visited_set.add(s)
        visited.append(s)
        if par is not None:
            tree_edges.append((par, act, s))
        lbl = html.escape(problem.state_label(s))

        if problem.is_goal(s) and stop_on_goal:
            sol = _reconstruct_path(parent, s)
            snap(
                "goal",
                current=s,
                solution=sol,
                code_line=6,
                title=f"🎉 Cel osiągnięty: {problem.state_label(s)} (koszt g={c})!",
                message=(f"Wyjęto z kolejki priorytetowej najtańszy stan <b>{lbl}</b> (koszt g={c}, głębokość d={d}). "
                         f"To stan docelowy! Rozwiązanie optymalne ma koszt <b>{c}</b> ({n_steps(len(sol))})."),
            )
            return steps

        added = []
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            step_cost = problem.action_cost(s, a, s2)
            nc = c + step_cost
            nd = d + 1
            if s2 not in cost_map or nc < cost_map[s2]:
                cost_map[s2] = nc
                depth_map[s2] = nd
                parent[s2] = (s, a)
                heapq.heappush(heap, (nc, next(counter), s2, s, a, nd))
                added.append(f"{html.escape(problem.state_label(s2))} (g={nc})")

        desc = ", ".join(added) if added else "brak tańszych sąsiadów"
        snap(
            "pop",
            current=s,
            code_line=9,
            title=f"Krok #{len(visited)}: wybór najtańszego stanu {problem.state_label(s)} (g={c}, d={d})",
            message=(f"<code>heappop()</code> → stan o najmniejszym koszcie <b>{lbl}</b> (koszt g={c}, d={d}). "
                     f"Zaktualizowano w kolejce: <b>{desc}</b>."),
        )

    snap(
        "done",
        current=None,
        code_line=11,
        title="Kolejka priorytetowa pusta",
        message="Wszystkie osiągalne stany zostały odwiedzone.",
    )
    return steps


def run_trace(problem: Problem, algo: str = "bfs_iter", detailed: bool = False, stop_on_goal: bool = True) -> list[SearchStep]:
    """Uruchamia wybrany algorytm i zwraca liste krokow SearchStep."""
    if algo == "bfs_iter":
        return trace_bfs_iterative(problem, detailed=detailed, stop_on_goal=stop_on_goal)
    if algo == "dfs_iter":
        return trace_dfs_iterative(problem, detailed=detailed, stop_on_goal=stop_on_goal)
    if algo == "dfs_rec":
        return trace_dfs_recursive(problem, detailed=detailed, stop_on_goal=stop_on_goal)
    if algo == "bfs_rec":
        return trace_bfs_recursive(problem, detailed=detailed, stop_on_goal=stop_on_goal)
    if algo == "dijkstra":
        return trace_dijkstra(problem, detailed=detailed, stop_on_goal=stop_on_goal)
    raise ValueError(f"Nieznany algorytm: {algo!r}")


# --------------------------------------------------------------------------
# Kody zrodlowe (do podgladu edukacyjnego z podswietlaniem linii)
# --------------------------------------------------------------------------
ALGO_INFO = {
    "bfs_iter": {
        "name": "BFS – iteracyjny (Kolejka FIFO)",
        "short": "BFS (kolejka FIFO)",
        "struct_title": "Kolejka FIFO (First-In, First-Out)",
        "struct_kind": "queue",
        "code": [
            "def bfs_iteracyjny(problem):",
            "    kolejka = deque([problem.initial])       # 1. Kolejka FIFO",
            "    odwiedzone = {problem.initial}",
            "    while kolejka:",
            "        s = kolejka.popleft()                # 2. Pobierz z PRZODU kolejki",
            "        if problem.is_goal(s):",
            "            return odtworz_sciezke(s)        # 🎉 Znaleziono najkrótszą ścieżkę!",
            "        for a in problem.actions(s):",
            "            s2 = problem.result(s, a)",
            "            if s2 not in odwiedzone:",
            "                odwiedzone.add(s2)",
            "                kolejka.append(s2)           # 3. Dopisz na KONIEC kolejki",
        ],
    },
    "dfs_iter": {
        "name": "DFS – iteracyjny (Stos LIFO)",
        "short": "DFS (stos LIFO)",
        "struct_title": "Stos LIFO (Last-In, First-Out)",
        "struct_kind": "stack",
        "code": [
            "def dfs_iteracyjny(problem):",
            "    stos = [problem.initial]                 # 1. Jawny stos LIFO (lista)",
            "    odwiedzone = set()",
            "    while stos:",
            "        s = stos.pop()                       # 2. Zdejmij z WIERZCHU stosu",
            "        if s in odwiedzone:",
            "            continue",
            "        odwiedzone.add(s)",
            "        if problem.is_goal(s):",
            "            return odtworz_sciezke(s)        # 🎉 Cel osiągnięty!",
            "        for a in reversed(problem.actions(s)):",
            "            if problem.result(s, a) not in odwiedzone:",
            "                stos.append(problem.result(s, a))  # 3. Połóż na WIERZCH stosu",
        ],
    },
    "dfs_rec": {
        "name": "DFS – rekurencyjny (Stos wywołań funkcji)",
        "short": "DFS (rekurencja)",
        "struct_title": "Stos wywołań (Call Stack w pamięci)",
        "struct_kind": "call_stack",
        "code": [
            "def dfs_rekurencyjny(s, odwiedzone):",
            "    odwiedzone.add(s)                        # 1. Wejście do funkcji (nowa ramka)",
            "    if problem.is_goal(s):",
            "        return [s]                           # 🎉 Cel osiągnięty!",
            "    for a in problem.actions(s):",
            "        s2 = problem.result(s, a)",
            "        if s2 not in odwiedzone:",
            "             wynik = dfs_rekurencyjny(s2, odwiedzone)  # 2. Rekurencja w głąb!",
            "            if wynik is not None: return [s] + wynik",
            "    return None                              # 3. Powrót (backtrack) o piętro wyżej",
        ],
    },
    "bfs_rec": {
        "name": "BFS – rekurencyjny (Warstwa po warstwie)",
        "short": "BFS (rekurencja warstwowa)",
        "struct_title": "Stos wywołań warstw + Kolejka warstwy",
        "struct_kind": "bfs_rec",
        "code": [
            "def bfs_warstwa(obecna_warstwa, odwiedzone, d=0):",
            "    if not obecna_warstwa: return None       # Warunek bazowy: brak stanów",
            "    nastepna_warstwa = []",
            "    for s in obecna_warstwa:                 # 1. Przejrzyj całą warstwę d",
            "        if problem.is_goal(s):",
            "            return odtworz_sciezke(s)        # 🎉 Cel osiągnięty!",
            "        for a in problem.actions(s):",
            "            s2 = problem.result(s, a)",
            "            if s2 not in odwiedzone:",
            "                odwiedzone.add(s2); nastepna_warstwa.append(s2)",
            "    return bfs_warstwa(nastepna_warstwa, odwiedzone, d + 1)  # 2. Rekurencja dla d+1",
        ],
    },
    "dijkstra": {
        "name": "Dijkstra / UCS (Kolejka priorytetowa wg kosztu g)",
        "short": "Dijkstra / UCS",
        "struct_title": "Kolejka priorytetowa (Min-Heap wg g(s))",
        "struct_kind": "pq",
        "code": [
            "def dijkstra_ucs(problem):",
            "    kopiec = [(0, problem.initial)]          # (koszt_g, stan)",
            "    najtanszy = {problem.initial: 0}",
            "    while kopiec:",
            "        g, s = heapq.heappop(kopiec)         # Wyjmij stan o NAJMNIEJSZYM koszcie g",
            "        if problem.is_goal(s):",
            "            return odtworz_sciezke(s), g     # 🎉 Rozwiązanie optymalne!",
            "        for a in problem.actions(s):",
            "            s2 = problem.result(s, a)",
            "            nowy_g = g + problem.action_cost(s, a, s2)",
            "            if s2 not in najtanszy or nowy_g < najtanszy[s2]:",
            "                najtanszy[s2] = nowy_g; heapq.heappush(kopiec, (nowy_g, s2))",
        ],
    },
}


def format_code_html(algo: str, highlight_line: int = 0) -> str:
    lines = ALGO_INFO[algo]["code"]
    rows = []
    for i, line in enumerate(lines, start=1):
        active = i == highlight_line
        bg = "#fff3bf" if active else "#f8f9fa"
        fw = "bold" if active else "normal"
        marker = "▶" if active else "&nbsp;"
        rows.append(
            f"<div style='background:{bg};font-weight:{fw};padding:1px 6px;"
            f"font-family:monospace;white-space:pre;color:#212529;border-left:3px solid "
            f"{'#f08c00' if active else 'transparent'}'>"
            f"<span style='color:#868e96;user-select:none'>{marker} {i:2d} │ </span>{html.escape(line)}</div>"
        )
    return (
        "<div style='border:1px solid #ced4da;border-radius:6px;overflow:hidden;"
        "background:#f8f9fa;font-size:12px;margin:4px 0'>"
        + "".join(rows)
        + "</div>"
    )


# --------------------------------------------------------------------------
# Rysowanie grafu z wizualizacja kroku przeszukiwania
# --------------------------------------------------------------------------
def draw_search_step(
    space: StateSpace,
    step: SearchStep,
    ax=None,
    edge_labels: bool = True,
    self_loops: bool = False,
    show_depth: bool = True,
    title: str | None = None,
):
    """Rysuje graf przestrzeni stanow w danym kroku przeszukiwania."""
    p = space.problem
    pos = p.layout(space.states)
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    w, h = p.node_w, p.node_h
    x0, x1 = min(xs) - w - 0.25, max(xs) + w + 0.25
    y0, y1 = min(ys) - h - 0.55, max(ys) + h + 0.75

    if ax is None:
        fw = max((x1 - x0) * 0.88, 5.2)
        fh = max((y1 - y0) * 0.88, 3.8)
        fig, ax = plt.subplots(figsize=(min(fw, 11), min(fh, 8.5)))
    else:
        fig = ax.figure

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, weight="bold", pad=8, color="#212529")

    visited_order = {s: idx + 1 for idx, s in enumerate(step.visited)}
    frontier_states = {item[0] for item in step.frontier}
    callstack_states = {f["state"] for f in step.call_stack}
    tree_edge_set = {(u, v) for u, _, v in step.tree_edges}
    sol_edge_set = {(u, v) for u, _, v in (step.solution or [])}
    sol_state_set = {s for st in (step.solution or []) for s in (st[0], st[2])}

    # 1. Rysowanie wezlow
    patches = {}
    for s in space.states:
        x, y = pos[s]
        faded = s not in space.reachable
        fc = C_UNREACH if faded else C_NODE
        ec, lw, ls_style = C_NODE_EDGE, 1.1, "solid"

        if p.is_goal(s) and not faded:
            fc = C_GOAL_FILL
        if s in frontier_states and s not in visited_order:
            fc = C_FRONTIER_FILL
            ec, lw = C_FRONTIER_EDGE, 1.8
        if s in visited_order:
            fc = C_GOAL_FILL if p.is_goal(s) else C_VISITED_FILL
            ec, lw = C_VISITED_EDGE, 1.8
        if s in callstack_states:
            ec, lw = C_CALLSTACK_EDGE, 2.3
        if s in sol_state_set:
            ec, lw = C_SOLUTION, 2.8
        if s == p.initial:
            ec, lw = C_INITIAL, max(lw, 2.6)
        if s == step.current:
            ec, lw = C_CURRENT, 3.6
            # delikatna obwodka podswietlenia wokół biezacego stanu
            glow = FancyBboxPatch(
                (x - w / 2 - 0.06, y - h / 2 - 0.06),
                w + 0.12,
                h + 0.12,
                boxstyle="round,pad=0.02,rounding_size=0.11",
                fc="none",
                ec=C_CURRENT,
                lw=1.5,
                alpha=0.35,
                zorder=3,
            )
            ax.add_patch(glow)

        box = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=fc,
            ec=ec,
            lw=lw,
            linestyle=ls_style,
            zorder=4,
            alpha=0.55 if faded else 1.0,
        )
        ax.add_patch(box)
        patches[s] = box
        p.draw_node_content(ax, s, x, y, faded)

        # Numer stanu w przestrzeni (#1, #2...) w lewym gornym rogu
        ax.text(
            x - w / 2,
            y + h / 2 + 0.04,
            f"#{space.number[s]}",
            fontsize=6.5,
            color="#868e96",
            ha="left",
            va="bottom",
            zorder=6,
        )

        # Znaczek kolejnosci odwiedzenia (1, 2, 3...) w prawym gornym rogu
        if s in visited_order:
            ord_num = visited_order[s]
            ax.text(
                x + w / 2,
                y + h / 2 + 0.04,
                f"krok {ord_num}",
                fontsize=7,
                weight="bold",
                color="white",
                ha="right",
                va="bottom",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.16", fc="#1971c2", ec="none"),
            )
        elif s in frontier_states:
            ax.text(
                x + w / 2,
                y + h / 2 + 0.04,
                "czeka",
                fontsize=6.5,
                weight="bold",
                color="#9c36b5",
                ha="right",
                va="bottom",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.14", fc="#f3d9fa", ec="#9c36b5", lw=0.7),
            )

        # Glebokosc d i koszt g pod wezlem
        if show_depth and s in step.depth_map:
            d_val = step.depth_map[s]
            g_val = step.cost_map.get(s, d_val)
            info_txt = f"d={d_val}" if g_val == d_val else f"d={d_val}, g={g_val}"
            ax.text(
                x,
                y - h / 2 - 0.06,
                info_txt,
                fontsize=6.8,
                color="#495057",
                ha="center",
                va="top",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75),
            )

    # 2. Rysowanie krawedzi
    for (s, s2), acts in space.edges.items():
        lbl = ",".join(p.action_short(a) for a in acts)
        col = "#adb5bd"
        lw, z, alpha = 0.9, 2, 0.55

        if (s, s2) in tree_edge_set:
            col, lw, z, alpha = C_TREE_EDGE, 2.2, 3, 0.95
        if (s, s2) in sol_edge_set:
            col, lw, z, alpha = "#2b8a3e", 3.2, 3.5, 1.0
        if step.active_edge and (step.active_edge[0], step.active_edge[2]) == (s, s2):
            col, lw, z, alpha = C_ACTIVE_EDGE, 3.2, 3.8, 1.0

        if s not in space.reachable:
            alpha = 0.25

        x, y = pos[s]
        if s == s2:
            if not self_loops:
                continue
            arc = Arc(
                (x, y + h / 2),
                w * 0.45,
                0.55,
                theta1=-10,
                theta2=190,
                color=col,
                lw=lw,
                zorder=z,
                alpha=alpha,
            )
            ax.add_patch(arc)
            continue

        rad = 0.18
        arr = FancyArrowPatch(
            pos[s],
            pos[s2],
            patchA=patches[s],
            patchB=patches[s2],
            arrowstyle="-|>",
            mutation_scale=10,
            lw=lw,
            color=col,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
            alpha=alpha,
        )
        ax.add_patch(arr)
        if edge_labels:
            mx, my = _bezier_mid(pos[s], pos[s2], rad)
            ax.text(
                mx,
                my,
                lbl,
                fontsize=6.8,
                ha="center",
                va="center",
                color=col if lw > 1.5 else "#495057",
                weight="bold" if lw > 1.5 else "normal",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.82),
            )

    # 3. Legenda na dole lub z boku
    handles = [
        Patch(fc=C_NODE, ec=C_INITIAL, lw=2.2, label="Start"),
        Patch(fc=C_GOAL_FILL, ec=C_NODE_EDGE, label="Cel"),
        Patch(fc=C_VISITED_FILL, ec=C_VISITED_EDGE, lw=1.8, label="Odwiedzony (Visited)"),
        Patch(fc=C_FRONTIER_FILL, ec=C_FRONTIER_EDGE, lw=1.8, label="W kolejce / na stosie (Frontier)"),
        Patch(fc=C_NODE, ec=C_CURRENT, lw=2.8, label="Bieżący stan"),
        Line2D([], [], color=C_TREE_EDGE, lw=2.2, label="Drzewo przeszukiwania"),
    ]
    if step.solution:
        handles.append(Line2D([], [], color="#2b8a3e", lw=3.0, label="Znaleziona ścieżka"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )
    return fig


# --------------------------------------------------------------------------
# Rysowanie struktury danych: Kolejka FIFO / Stos LIFO / Stos wywolan (Call Stack)
# --------------------------------------------------------------------------
def draw_structure_panel(ax, problem: Problem, step: SearchStep, algo: str):
    """Rysuje zawartosc kolejki FIFO, stosu LIFO lub stosu wywolan rekurencyjnych."""
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    info = ALGO_INFO[algo]
    kind = info["struct_kind"]
    ax.text(5.0, 9.65, info["struct_title"], ha="center", va="top", fontsize=10.5, weight="bold", color="#212529")

    if kind == "queue":
        # Kolejka FIFO: wyjmujemy z GORY/PRZODU (popleft), dopisujemy na DOL/KONIEC (append)
        ax.text(
            5.0,
            9.05,
            "⬆ popleft() – pobieramy z przodu kolejki",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#1971c2",
            weight="bold",
        )
        items = list(step.frontier)
        max_show = 7
        shown = items[:max_show]
        # Rysujemy szyny kolejki
        ax.plot([1.2, 1.2], [1.4, 8.5], color="#868e96", lw=2, linestyle="--")
        ax.plot([8.8, 8.8], [1.4, 8.5], color="#868e96", lw=2, linestyle="--")

        if not shown:
            ax.text(5.0, 5.0, "(kolejka pusta)", ha="center", va="center", fontsize=10, color="#868e96")
        else:
            box_h = 0.85
            gap = 0.12
            top_y = 7.6
            for i, (s, par, act, d, c) in enumerate(shown):
                y = top_y - i * (box_h + gap)
                is_first = i == 0
                fc = "#e7f5ff" if is_first else C_FRONTIER_FILL
                ec = "#1971c2" if is_first else C_FRONTIER_EDGE
                rect = FancyBboxPatch(
                    (1.5, y),
                    7.0,
                    box_h,
                    boxstyle="round,pad=0.03,rounding_size=0.12",
                    fc=fc,
                    ec=ec,
                    lw=2.0 if is_first else 1.2,
                )
                ax.add_patch(rect)
                lbl = problem.state_label(s)
                via = f"z {problem.state_label(par)} ({problem.action_short(act)})" if par is not None else "start"
                tag = "PRZÓD →" if is_first else f"#{i + 1}"
                ax.text(1.75, y + box_h / 2, tag, ha="left", va="center", fontsize=8, weight="bold", color=ec)
                ax.text(5.0, y + box_h / 2 + 0.1, lbl, ha="center", va="center", fontsize=9.5, weight="bold", color="#212529")
                ax.text(5.0, y + box_h / 2 - 0.22, f"{via} · d={d}", ha="center", va="center", fontsize=7.5, color="#495057")

            if len(items) > max_show:
                ax.text(5.0, 1.55, f"... + jeszcze {len(items) - max_show} w kolejce", ha="center", va="center", fontsize=8, color="#495057")

        ax.text(
            5.0,
            1.1,
            "⬆ append() – nowe stany trafiają na koniec",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#9c36b5",
            weight="bold",
        )

    elif kind == "stack":
        # Stos LIFO: zamkniety od dolu, wkladamy i zdejmujemy od GORY!
        ax.text(
            5.0,
            9.05,
            "⬆ pop() / ⬇ push() – WIERZCH STOSU",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#9c36b5",
            weight="bold",
        )
        # Kubel stosu (U-ksztaltny)
        ax.plot([1.2, 1.2, 8.8, 8.8], [8.5, 1.1, 1.1, 8.5], color="#495057", lw=2.5)
        ax.text(5.0, 0.75, "DNO STOSU (pierwsze włożone elementy czekają na dole)", ha="center", va="top", fontsize=7.5, color="#868e96")

        items = list(step.frontier)
        max_show = 7
        shown = items[-max_show:]  # pokazujemy gore stosu
        if not shown:
            ax.text(5.0, 4.8, "(stos pusty)", ha="center", va="center", fontsize=10, color="#868e96")
        else:
            box_h = 0.85
            gap = 0.10
            base_y = 1.25
            for i, (s, par, act, d, c) in enumerate(shown):
                y = base_y + i * (box_h + gap)
                is_top = i == len(shown) - 1
                fc = "#fff3bf" if is_top else C_FRONTIER_FILL
                ec = "#e8590c" if is_top else C_FRONTIER_EDGE
                rect = FancyBboxPatch(
                    (1.5, y),
                    7.0,
                    box_h,
                    boxstyle="round,pad=0.03,rounding_size=0.12",
                    fc=fc,
                    ec=ec,
                    lw=2.2 if is_top else 1.2,
                )
                ax.add_patch(rect)
                lbl = problem.state_label(s)
                via = f"z {problem.state_label(par)} ({problem.action_short(act)})" if par is not None else "start"
                tag = "WIERZCH ⬆" if is_top else f"poziom {i + 1}"
                ax.text(1.75, y + box_h / 2, tag, ha="left", va="center", fontsize=7.8, weight="bold", color=ec)
                ax.text(5.3, y + box_h / 2 + 0.1, lbl, ha="center", va="center", fontsize=9.5, weight="bold", color="#212529")
                ax.text(5.3, y + box_h / 2 - 0.22, f"{via} · d={d}", ha="center", va="center", fontsize=7.5, color="#495057")

    elif kind in ("call_stack", "bfs_rec"):
        # Stos wywolan rekurencyjnych (Call Stack)
        ax.text(
            5.0,
            9.05,
            "⬆ Aktywna ramka wywołania (Call Stack)",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#f08c00",
            weight="bold",
        )
        ax.plot([1.0, 1.0, 9.0, 9.0], [8.5, 1.1, 1.1, 8.5], color="#f08c00", lw=2.2)
        ax.text(
            5.0,
            0.75,
            "DNO STOSU WYWOŁAŃ (główne wywołanie funkcji)",
            ha="center",
            va="top",
            fontsize=7.5,
            color="#868e96",
        )

        frames = list(step.call_stack)
        max_show = 7
        shown = frames[-max_show:]
        if not shown:
            ax.text(5.0, 4.8, "(brak aktywnych wywołań)", ha="center", va="center", fontsize=10, color="#868e96")
        else:
            box_h = 0.85
            gap = 0.10
            base_y = 1.25
            for i, frame in enumerate(shown):
                y = base_y + i * (box_h + gap)
                is_top = i == len(shown) - 1
                fc = "#d3f9d8" if frame.get("status") == "goal" else ("#fff3bf" if is_top else "#fff9db")
                ec = "#2b8a3e" if frame.get("status") == "goal" else ("#e8590c" if is_top else "#f59f00")
                rect = FancyBboxPatch(
                    (1.3, y),
                    7.4,
                    box_h,
                    boxstyle="round,pad=0.03,rounding_size=0.12",
                    fc=fc,
                    ec=ec,
                    lw=2.2 if is_top else 1.2,
                )
                ax.add_patch(rect)
                ax.text(
                    1.55,
                    y + box_h / 2 + 0.12,
                    f"#{i + 1}  {frame['func']}",
                    ha="left",
                    va="center",
                    fontsize=8.8,
                    weight="bold",
                    family="monospace",
                    color="#212529",
                )
                ax.text(
                    1.55,
                    y + box_h / 2 - 0.2,
                    frame.get("note", ""),
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color="#495057",
                )
                if is_top:
                    ax.text(8.5, y + box_h / 2, "◀ TU", ha="right", va="center", fontsize=8, weight="bold", color=ec)

    elif kind == "pq":
        # Kolejka priorytetowa (Dijkstra)
        ax.text(
            5.0,
            9.05,
            "⬆ heappop() – najniższy koszt łączny g(s)",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#1971c2",
            weight="bold",
        )
        items = list(step.frontier)
        max_show = 7
        shown = items[:max_show]
        ax.plot([1.2, 1.2], [1.4, 8.5], color="#868e96", lw=2, linestyle=":")
        ax.plot([8.8, 8.8], [1.4, 8.5], color="#868e96", lw=2, linestyle=":")
        if not shown:
            ax.text(5.0, 5.0, "(kolejka priorytetowa pusta)", ha="center", va="center", fontsize=10, color="#868e96")
        else:
            box_h = 0.85
            gap = 0.12
            top_y = 7.6
            for i, (s, par, act, d, c) in enumerate(shown):
                y = top_y - i * (box_h + gap)
                is_first = i == 0
                fc = "#e7f5ff" if is_first else "#f8f9fa"
                ec = "#1971c2" if is_first else "#868e96"
                rect = FancyBboxPatch(
                    (1.5, y),
                    7.0,
                    box_h,
                    boxstyle="round,pad=0.03,rounding_size=0.12",
                    fc=fc,
                    ec=ec,
                    lw=2.0 if is_first else 1.2,
                )
                ax.add_patch(rect)
                lbl = problem.state_label(s)
                ax.text(
                    1.75,
                    y + box_h / 2,
                    f"g = {c}",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    weight="bold",
                    color="#1971c2",
                )
                ax.text(5.2, y + box_h / 2 + 0.1, lbl, ha="center", va="center", fontsize=9.5, weight="bold", color="#212529")
                ax.text(5.2, y + box_h / 2 - 0.22, f"głębokość d={d}", ha="center", va="center", fontsize=7.5, color="#495057")


# --------------------------------------------------------------------------
# Tabela przebiegu "jak na kartce" (HTML)
# --------------------------------------------------------------------------
def trace_table_html(problem: Problem, steps: list[SearchStep], active_idx: int = -1, algo: str = "bfs_iter") -> str:
    """Generuje tabele HTML krok po kroku (dokladnie taka, jaka studenci wypelniaja na kartce)."""
    struct_col = "Stos wywołań (Call Stack)" if algo == "dfs_rec" else (
        "Stos LIFO (od dołu → WIERZCH)" if algo == "dfs_iter" else "Kolejka (PRZÓD → koniec)"
    )
    head = (
        f"<tr><th>Krok</th><th>Faza</th><th>Bieżący stan</th>"
        f"<th>{struct_col}</th><th>Odwiedzone (Visited)</th><th>Komentarz</th></tr>"
    )
    rows = []
    for st in steps:
        is_cur = st.step_idx == active_idx
        cls = "goal" if st.phase == "goal" else ("ini" if is_cur else "")
        style = "outline:2px solid #e03131;font-weight:bold;" if is_cur else ""
        cur_lbl = html.escape(problem.state_label(st.current)) if st.current is not None else "—"
        if algo == "dfs_rec":
            front_str = " → ".join(html.escape(f["func"]) for f in st.call_stack) or "∅"
        else:
            front_str = ", ".join(
                html.escape(problem.state_label(item[0]))
                + (f" (g={item[4]})" if algo == "dijkstra" else f" (d={item[3]})")
                for item in st.frontier
            ) or "∅"
        vis_str = ", ".join(html.escape(problem.state_label(s)) for s in st.visited) or "∅"
        rows.append(
            f"<tr class='{cls}' style='{style}'>"
            f"<td style='text-align:center'>{st.step_idx}</td>"
            f"<td><code>{html.escape(st.phase)}</code></td>"
            f"<td><b>{cur_lbl}</b></td>"
            f"<td><code>[{front_str}]</code></td>"
            f"<td>{vis_str}</td>"
            f"<td>{st.message}</td></tr>"
        )
    return _CSS + f"<table class='ls'>{head}{''.join(rows)}</table>"


# --------------------------------------------------------------------------
# Sekcja 0 (Powtorka): Pokaz ze Dijkstra przy rownych wagach = BFS
# --------------------------------------------------------------------------
def dijkstra_vs_bfs_demo(problem: Problem | None = None):
    """Rysuje obok siebie wynik Dijkstry (z Lab 2) oraz BFS na tym samym grafie."""
    from IPython.display import HTML, display

    if problem is None:
        problem = Elevator(floors=3, initial=(0, "o"), goal=(2, "o"))

    space = StateSpace(problem)
    bfs_steps = trace_bfs_iterative(problem, detailed=False)
    dij_steps = trace_dijkstra(problem, detailed=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    draw_search_step(
        space,
        bfs_steps[-1],
        ax=ax1,
        title="BFS (Przeszukiwanie wszerz – Kolejka FIFO)",
    )
    draw_search_step(
        space,
        dij_steps[-1],
        ax=ax2,
        title="Dijkstra / UCS z Lab 2 (gdy każda akcja ma koszt = 1)",
    )
    fig.tight_layout()
    display(fig)
    plt.close(fig)

    bfs_order = " → ".join(problem.state_label(s) for s in bfs_steps[-1].visited)
    dij_order = " → ".join(problem.state_label(s) for s in dij_steps[-1].visited)
    html_summary = _CSS + (
        "<table class='ls'>"
        "<tr><th>Algorytm</th><th>Kolejność odwiedzania stanów</th><th>Znalezione rozwiązanie</th></tr>"
        f"<tr><td><b>BFS (wszerz)</b></td><td><code>{html.escape(bfs_order)}</code></td>"
        f"<td><b>{html.escape(problem.solution_text(bfs_steps[-1].solution or []))}</b> "
        f"(d = {len(bfs_steps[-1].solution or [])})</td></tr>"
        f"<tr><td><b>Dijkstra (wagi = 1)</b></td><td><code>{html.escape(dij_order)}</code></td>"
        f"<td><b>{html.escape(problem.solution_text(dij_steps[-1].solution or []))}</b> "
        f"(koszt g = {len(dij_steps[-1].solution or [])})</td></tr>"
        "</table>"
    )
    display(HTML(html_summary))


# --------------------------------------------------------------------------
# Glowna aplikacja interaktywna: Symulator BFS / DFS / Dijkstra
# --------------------------------------------------------------------------
class SearchSimulatorApp:
    """Interaktywny symulator i animacja przeszukiwania grafu (BFS, DFS, Dijkstra)."""

    def __init__(self, default_problem: str = "elevator", default_algo: str = "bfs_iter"):
        import ipywidgets as W

        self.W = W
        self._busy = False
        st = {"description_width": "initial"}

        self.w_prob = W.Dropdown(
            options=[
                ("🛗 Winda (3 piętra)", "elevator"),
                ("🛗 Winda (4 piętra)", "elevator4"),
                ("🧹 Świat odkurzacza (2 pola)", "vacuum2"),
                ("🧹 Świat odkurzacza (3 pola)", "vacuum3"),
                ("🪣 Pojemniki z wodą (4 l i 3 l, koszt=1)", "jugs_unit"),
                ("🪣 Pojemniki z wodą (4 l i 3 l, koszt=litry wody)", "jugs_water"),
            ],
            value=default_problem,
            description="Problem z Lab 2:",
            style=st,
            layout=W.Layout(width="360px"),
        )
        self.w_algo = W.Dropdown(
            options=[
                ("BFS – wszerz (iteracyjny, Kolejka FIFO)", "bfs_iter"),
                ("DFS – w głąb (iteracyjny, Stos LIFO)", "dfs_iter"),
                ("DFS – w głąb (rekurencyjny, Stos wywołań)", "dfs_rec"),
                ("BFS – wszerz (rekurencyjny, warstwami)", "bfs_rec"),
                ("Dijkstra / UCS (Kolejka priorytetowa wg kosztu g)", "dijkstra"),
            ],
            value=default_algo,
            description="Algorytm:",
            style=st,
            layout=W.Layout(width="390px"),
        )
        self.w_detailed = W.Checkbox(
            value=False,
            description="Tryb szczegółowy (pokaż też sprawdzanie każdego sąsiada)",
            indent=False,
        )
        self.w_stop_goal = W.Checkbox(
            value=True,
            description="Zatrzymaj po znalezieniu celu",
            indent=False,
        )

        # Kontrolki odtwarzania animacji
        self.play = W.Play(
            value=0,
            min=0,
            max=10,
            step=1,
            interval=900,
            description="Animacja",
            disabled=False,
        )
        self.slider = W.IntSlider(
            value=0,
            min=0,
            max=10,
            step=1,
            description="Krok:",
            style=st,
            layout=W.Layout(width="340px"),
        )
        W.jslink((self.play, "value"), (self.slider, "value"))

        self.btn_prev = W.Button(description="◀ Wstecz", layout=W.Layout(width="90px"))
        self.btn_next = W.Button(description="Dalej ▶", layout=W.Layout(width="90px"))
        self.w_speed = W.Dropdown(
            options=[("Wolno (1.4 s)", 1400), ("Normalnie (0.9 s)", 900), ("Szybko (0.45 s)", 450)],
            value=900,
            description="Tempo:",
            style=st,
            layout=W.Layout(width="200px"),
        )

        self.btn_prev.on_click(self._prev_step)
        self.btn_next.on_click(self._next_step)
        self.w_speed.observe(self._on_speed, names="value")
        for w in (self.w_prob, self.w_algo, self.w_detailed, self.w_stop_goal):
            w.observe(self.rebuild, names="value")
        self.slider.observe(self._render_step, names="value")

        self.status_banner = W.HTML()
        self.out_vis = W.Output()
        self.out_code = W.Output()
        self.out_compare = W.Output()
        self.out_table = W.Output()

        tab1 = W.VBox([
            self.status_banner,
            self.out_vis,
            W.HTML("<b>Podgląd wykonywanego kodu w tym kroku:</b>"),
            self.out_code,
        ])
        tabs = W.Tab(children=[tab1, self.out_compare, self.out_table])
        tabs.set_title(0, "🎬 Wizualizacja + Stos/Kolejka")
        tabs.set_title(1, "⚖️ Porównanie: BFS vs DFS vs Dijkstra")
        tabs.set_title(2, "📝 Tabela kroków (jak na kartce)")

        top_bar = W.VBox([
            W.HBox([self.w_prob, self.w_algo], layout=W.Layout(flex_flow="row wrap", grid_gap="10px")),
            W.HBox([self.w_detailed, self.w_stop_goal, self.w_speed], layout=W.Layout(flex_flow="row wrap", grid_gap="16px")),
            W.HBox([self.play, self.btn_prev, self.btn_next, self.slider], layout=W.Layout(flex_flow="row wrap", align_items="center", grid_gap="8px")),
        ])
        self.widget = W.VBox([
            W.HTML("<h3 style='margin:4px 0'>🔍 Interaktywny symulator przeszukiwania: BFS, DFS i Dijkstra</h3>"),
            top_bar,
            tabs,
        ])
        self.rebuild()

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def _make_problem(self) -> Problem:
        key = self.w_prob.value
        if key == "elevator":
            return Elevator(floors=3, initial=(0, "o"), goal=(2, "o"))
        if key == "elevator4":
            return Elevator(floors=4, initial=(0, "o"), goal=(3, "o"))
        if key == "vacuum2":
            return VacuumWorld(n=2, initial=(0, (1, 1)))
        if key == "vacuum3":
            return VacuumWorld(n=3, initial=(1, (1, 1, 1)))
        if key == "jugs_unit":
            return WaterJugs(cap_a=4, cap_b=3, initial=(0, 0), goal=(2, 0), cost="unit")
        if key == "jugs_water":
            return WaterJugs(cap_a=4, cap_b=3, initial=(0, 0), goal=(2, 0), cost="water")
        return Elevator()

    def _on_speed(self, change):
        self.play.interval = change["new"]

    def _prev_step(self, _=None):
        if self.slider.value > 0:
            self.slider.value -= 1

    def _next_step(self, _=None):
        if self.slider.value < self.slider.max:
            self.slider.value += 1

    def rebuild(self, *_):
        if self._busy:
            return
        self._busy = True
        try:
            self.problem = self._make_problem()
            self.space = StateSpace(self.problem)
            self.algo = self.w_algo.value
            self.steps = run_trace(
                self.problem,
                algo=self.algo,
                detailed=self.w_detailed.value,
                stop_on_goal=self.w_stop_goal.value,
            )
            max_idx = max(0, len(self.steps) - 1)
            self.play.max = max_idx
            self.slider.max = max_idx
            self.slider.value = 0
        finally:
            self._busy = False
        self._render_step()
        self._render_comparison()

    def _render_step(self, *_):
        if self._busy or not self.steps:
            return
        idx = min(self.slider.value, len(self.steps) - 1)
        step = self.steps[idx]

        phase_colors = {
            "init": ("#e7f5ff", "#1971c2"),
            "pop": ("#fff9db", "#e8590c"),
            "neighbor": ("#f3d9fa", "#9c36b5"),
            "skip": ("#f1f3f5", "#868e96"),
            "backtrack": ("#fff3bf", "#f08c00"),
            "goal": ("#d3f9d8", "#2b8a3e"),
            "done": ("#e9ecef", "#495057"),
        }
        bg, border = phase_colors.get(step.phase, ("#f8f9fa", "#495057"))
        self.status_banner.value = (
            f"<div style='background:{bg};border-left:5px solid {border};padding:8px 12px;"
            f"margin:4px 0;border-radius:4px;color:#212529;font-size:13.5px'>"
            f"<b>Krok {idx} / {len(self.steps) - 1} — {html.escape(step.title)}</b><br>"
            f"{step.message}</div>"
        )

        with self.out_vis:
            self.out_vis.clear_output(wait=True)
            fig, (ax_g, ax_s) = plt.subplots(
                1,
                2,
                figsize=(13.5, 5.4),
                gridspec_kw={"width_ratios": [1.85, 1.0]},
            )
            draw_search_step(
                self.space,
                step,
                ax=ax_g,
                title=f"Graf przestrzeni stanów — {ALGO_INFO[self.algo]['short']}",
            )
            draw_structure_panel(ax_s, self.problem, step, self.algo)
            fig.tight_layout()
            from IPython.display import display
            display(fig)
            plt.close(fig)

        with self.out_code:
            self.out_code.clear_output(wait=True)
            from IPython.display import HTML, display
            display(HTML(format_code_html(self.algo, highlight_line=step.code_line)))

        with self.out_table:
            self.out_table.clear_output(wait=True)
            from IPython.display import HTML, display
            display(HTML(trace_table_html(self.problem, self.steps, active_idx=idx, algo=self.algo)))

    def _render_comparison(self):
        with self.out_compare:
            self.out_compare.clear_output(wait=True)
            from IPython.display import HTML, display

            p = self.problem
            bfs_st = trace_bfs_iterative(p, detailed=False, stop_on_goal=self.w_stop_goal.value)[-1]
            dfs_st = trace_dfs_iterative(p, detailed=False, stop_on_goal=self.w_stop_goal.value)[-1]
            dij_st = trace_dijkstra(p, detailed=False, stop_on_goal=self.w_stop_goal.value)[-1]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
            draw_search_step(self.space, bfs_st, ax=ax1, title="BFS (wszerz – Kolejka FIFO)")
            draw_search_step(self.space, dfs_st, ax=ax2, title="DFS (w głąb – Stos LIFO)")
            fig.tight_layout()
            display(fig)
            plt.close(fig)

            rows_html = []
            for name, st_end in (
                ("BFS (wszerz – kolejka FIFO)", bfs_st),
                ("DFS (w głąb – stos LIFO)", dfs_st),
                ("Dijkstra / UCS (kolejka priorytetowa)", dij_st),
            ):
                sol = st_end.solution
                if sol is not None:
                    cost = sum(p.action_cost(*e) for e in sol)
                    sol_str = f"<b>{html.escape(p.solution_text(sol))}</b> ({n_steps(len(sol))}, koszt = {cost})"
                else:
                    sol_str = "—"
                order_str = " → ".join(html.escape(p.state_label(s)) for s in st_end.visited)
                rows_html.append(
                    f"<tr><td><b>{name}</b></td>"
                    f"<td style='text-align:center'><b>{len(st_end.visited)}</b> / {len(self.space.reachable)}</td>"
                    f"<td>{sol_str}</td>"
                    f"<td><span style='font-size:90%'>{order_str}</span></td></tr>"
                )
            tbl = _CSS + (
                "<table class='ls'>"
                "<tr><th>Algorytm</th><th>Odwiedzone stany</th><th>Znaleziona ścieżka do celu</th>"
                "<th>Kolejność odwiedzania stanów</th></tr>"
                + "".join(rows_html)
                + "</table>"
            )
            display(HTML(tbl))


def search_app(default_problem: str = "elevator", default_algo: str = "bfs_iter"):
    """Uruchamia glowny interaktywny symulator przeszukiwania (BFS, DFS, Dijkstra)."""
    return SearchSimulatorApp(default_problem=default_problem, default_algo=default_algo)


# --------------------------------------------------------------------------
# Sekcja 2: Porownanie Iteracja (jawny stos/kolejka) vs Rekurencja (Call Stack)
# --------------------------------------------------------------------------
class RecursionVsIterationApp:
    """Zestawia obok siebie wersje iteracyjna i rekurencyjna dla DFS oraz BFS."""

    def __init__(self):
        import ipywidgets as W

        self.W = W
        st = {"description_width": "initial"}
        self.w_family = W.ToggleButtons(
            options=[
                ("DFS: Stos LIFO (iteracja) vs Call Stack (rekurencja)", "dfs"),
                ("BFS: Kolejka FIFO (iteracja) vs Warstwy (rekurencja)", "bfs"),
            ],
            value="dfs",
            description="Porównaj:",
            style=st,
        )
        self.w_prob = W.Dropdown(
            options=[
                ("🛗 Winda (3 piętra)", "elevator"),
                ("🧹 Świat odkurzacza (2 pola)", "vacuum2"),
                ("🪣 Pojemniki z wodą (4 l i 3 l)", "jugs"),
            ],
            value="elevator",
            description="Problem:",
            style=st,
        )
        self.slider = W.IntSlider(value=0, min=0, max=5, step=1, description="Krok:", style=st, layout=W.Layout(width="420px"))
        self.w_family.observe(self.rebuild, names="value")
        self.w_prob.observe(self.rebuild, names="value")
        self.slider.observe(self._render, names="value")

        self.out = W.Output()
        self.widget = W.VBox([
            W.HTML("<h3 style='margin:4px 0'>🔄 Dwa sposoby implementacji: Iteracja vs Rekurencja</h3>"),
            W.HBox([self.w_family, self.w_prob], layout=W.Layout(flex_flow="row wrap", grid_gap="12px")),
            self.slider,
            self.out,
        ])
        self.rebuild()

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def rebuild(self, *_):
        key = self.w_prob.value
        if key == "elevator":
            self.problem = Elevator(floors=3, initial=(0, "o"), goal=(2, "o"))
        elif key == "vacuum2":
            self.problem = VacuumWorld(n=2, initial=(0, (1, 1)))
        else:
            self.problem = WaterJugs(cap_a=4, cap_b=3, initial=(0, 0), goal=(2, 0))

        self.space = StateSpace(self.problem)
        if self.w_family.value == "dfs":
            self.algo_iter, self.algo_rec = "dfs_iter", "dfs_rec"
            self.steps_iter = trace_dfs_iterative(self.problem, detailed=False)
            self.steps_rec = trace_dfs_recursive(self.problem, detailed=False)
        else:
            self.algo_iter, self.algo_rec = "bfs_iter", "bfs_rec"
            self.steps_iter = trace_bfs_iterative(self.problem, detailed=False)
            self.steps_rec = trace_bfs_recursive(self.problem, detailed=False)

        self.slider.max = max(len(self.steps_iter) - 1, len(self.steps_rec) - 1)
        self.slider.value = 0
        self._render()

    def _render(self, *_):
        i_it = min(self.slider.value, len(self.steps_iter) - 1)
        i_rc = min(self.slider.value, len(self.steps_rec) - 1)
        st_it = self.steps_iter[i_it]
        st_rc = self.steps_rec[i_rc]

        with self.out:
            self.out.clear_output(wait=True)
            from IPython.display import HTML, display

            fig, (ax_g, ax_it, ax_rc) = plt.subplots(
                1,
                3,
                figsize=(14.2, 4.9),
                gridspec_kw={"width_ratios": [1.5, 1.0, 1.05]},
            )
            draw_search_step(
                self.space,
                st_it,
                ax=ax_g,
                title=f"Odwiedzany graf (krok {self.slider.value})",
            )
            draw_structure_panel(ax_it, self.problem, st_it, self.algo_iter)
            draw_structure_panel(ax_rc, self.problem, st_rc, self.algo_rec)
            fig.tight_layout()
            display(fig)
            plt.close(fig)

            code_side_by_side = (
                "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>"
                f"<div><b>1. Wersja iteracyjna ({ALGO_INFO[self.algo_iter]['short']}):</b>"
                f"<div style='font-size:12.5px;margin:2px 0;color:#495057'>{st_it.message}</div>"
                f"{format_code_html(self.algo_iter, st_it.code_line)}</div>"
                f"<div><b>2. Wersja rekurencyjna ({ALGO_INFO[self.algo_rec]['short']}):</b>"
                f"<div style='font-size:12.5px;margin:2px 0;color:#495057'>{st_rc.message}</div>"
                f"{format_code_html(self.algo_rec, st_rc.code_line)}</div>"
                "</div>"
            )
            display(HTML(code_side_by_side))


def recursion_vs_iteration_app():
    """Uruchamia interaktywne porownanie implementacji iteracyjnej i rekurencyjnej."""
    return RecursionVsIterationApp()


# --------------------------------------------------------------------------
# Sekcja 3: Przeszukiwanie drzewa uwagi (Attention Heads) w modelach Transformer
# --------------------------------------------------------------------------
ATTENTION_SCENARIOS = {
    "coref": {
        "title": "1. Rozwiązywanie zaimków (Coreference): „Jan dał książkę Marii, bo ona...”",
        "tokens": ["Jan", "dał", "książkę", "Marii", "bo", "ona"],
        "query_idx": 5,  # "ona" w warstwie wyjsciowej L3
        "description": (
            "Model przetwarza token <b>„ona”</b> w warstwie 3 i musi ustalić, do kogo odnosi się zaimek "
            "(żeby poprawnie przewidzieć kolejne słowo). Przeszukując graf uwagi wstecz od węzła <code>[L3] ona</code> "
            "przez kolejne warstwy i głowice uwagi (Attention Heads), odkrywamy ścieżki przepływu informacji!"
        ),
        "heads": {
            "H0 (Składnia / Czasownik)": "#1c7ed6",
            "H1 (Zaimki / Podmiot-Dopełnienie)": "#e8590c",
            "H2 (Kontekst lokalny)": "#2f9e44",
        },
        # Krawedzie drzewa uwagi: (warstwa_zrodlowa, idx_tokenu) -> [(warstwa_cel, idx_tokenu, nazwa_glowicy, waga_alpha), ...]
        "edges": {
            (3, 5): [
                (2, 3, "H1 (Zaimki / Podmiot-Dopełnienie)", 0.56),   # ona -> Marii
                (2, 1, "H0 (Składnia / Czasownik)", 0.28),          # ona -> dał
                (2, 4, "H2 (Kontekst lokalny)", 0.16),              # ona -> bo
            ],
            (2, 3): [
                (1, 3, "H1 (Zaimki / Podmiot-Dopełnienie)", 0.52),   # Marii -> Marii (tozsamosc)
                (1, 2, "H0 (Składnia / Czasownik)", 0.33),          # Marii -> książkę
                (1, 1, "H2 (Kontekst lokalny)", 0.15),              # Marii -> dał
            ],
            (2, 1): [
                (1, 0, "H0 (Składnia / Czasownik)", 0.58),          # dał -> Jan (kto dal?)
                (1, 2, "H0 (Składnia / Czasownik)", 0.29),          # dał -> książkę (co dal?)
                (1, 3, "H1 (Zaimki / Podmiot-Dopełnienie)", 0.13),
            ],
            (2, 4): [
                (1, 3, "H2 (Kontekst lokalny)", 0.65),
                (1, 1, "H0 (Składnia / Czasownik)", 0.35),
            ],
            (1, 3): [
                (0, 3, "H1 (Zaimki / Podmiot-Dopełnienie)", 0.78),   # Warstwa 0: token "Marii"
                (0, 2, "H2 (Kontekst lokalny)", 0.22),
            ],
            (1, 2): [
                (0, 2, "H0 (Składnia / Czasownik)", 0.68),          # Warstwa 0: token "książkę"
                (0, 1, "H2 (Kontekst lokalny)", 0.32),
            ],
            (1, 0): [
                (0, 0, "H0 (Składnia / Czasownik)", 0.85),          # Warstwa 0: token "Jan"
                (0, 1, "H2 (Kontekst lokalny)", 0.15),
            ],
            (1, 1): [
                (0, 1, "H0 (Składnia / Czasownik)", 0.72),
                (0, 0, "H0 (Składnia / Czasownik)", 0.28),
            ],
        },
        "target_nodes": {(0, 3)},  # zrodlowy token "Marii" w warstwie wejsciowej L0
    },
    "induction": {
        "title": "2. Obwód indukcyjny (Induction Circuit): „Kognitywistyka bada umysł. AI bada → [?]”",
        "tokens": ["Kognitywistyka", "bada", "umysł", "AI", "bada"],
        "query_idx": 4,  # drugie "bada"
        "description": (
            "Klasyczny obwód odkryty w interpretowalności mechanistycznej (Olsson et al.): "
            "przy drugim słowie <b>„bada”</b> głowica indukcyjna (Induction Head) szuka poprzedniego wystąpienia "
            "słowa „bada” i sprawdza, jakie słowo wystąpiło <i>zaraz po nim</i> (<b>„umysł”</b>)!"
        ),
        "heads": {
            "H0 (Previous Token Head)": "#1c7ed6",
            "H1 (Induction / Name Mover Head)": "#9c36b5",
            "H2 (Self / Context Head)": "#2f9e44",
        },
        "edges": {
            (3, 4): [
                (2, 2, "H1 (Induction / Name Mover Head)", 0.64),   # drugie "bada" patrzy na "umysł"
                (2, 3, "H0 (Previous Token Head)", 0.24),          # patrzy na "AI"
                (2, 4, "H2 (Self / Context Head)", 0.12),
            ],
            (2, 2): [
                (1, 1, "H0 (Previous Token Head)", 0.72),          # "umysł" niesie info, ze stalo po "bada" (#1)
                (1, 2, "H2 (Self / Context Head)", 0.28),
            ],
            (2, 3): [
                (1, 3, "H2 (Self / Context Head)", 0.65),
                (1, 0, "H1 (Induction / Name Mover Head)", 0.35),
            ],
            (2, 4): [
                (1, 3, "H0 (Previous Token Head)", 0.75),
                (1, 4, "H2 (Self / Context Head)", 0.25),
            ],
            (1, 1): [
                (0, 1, "H2 (Self / Context Head)", 0.62),          # pierwsze "bada"
                (0, 0, "H0 (Previous Token Head)", 0.38),          # "Kognitywistyka"
            ],
            (1, 2): [
                (0, 2, "H2 (Self / Context Head)", 0.80),          # "umysł"
                (0, 1, "H0 (Previous Token Head)", 0.20),
            ],
            (1, 3): [
                (0, 3, "H2 (Self / Context Head)", 0.82),
                (0, 2, "H0 (Previous Token Head)", 0.18),
            ],
            (1, 0): [
                (0, 0, "H2 (Self / Context Head)", 0.90),
            ],
            (1, 4): [
                (0, 4, "H2 (Self / Context Head)", 0.88),
            ],
        },
        "target_nodes": {(0, 2), (0, 1)},
    },
    "multihop": {
        "title": "3. Wnioskowanie wielokrokowe (Multi-hop): „Klucz jest w skrzyni, skrzynia w piwnicy, więc klucz jest w...”",
        "tokens": ["Klucz", "w_skrzyni", "skrzynia", "w_piwnicy", "klucz", "jest_w"],
        "query_idx": 5,
        "description": (
            "Aby odpowiedzieć, gdzie jest klucz, pojedyncza warstwa uwagi nie wystarczy! "
            "Model musi wykonać <b>2 skoki (2-hop traversal)</b>: w wyższej warstwie połączyć <code>klucz → skrzynia</code>, "
            "a w niższej warstwie <code>skrzynia → w_piwnicy</code>. To dokładnie przeszukiwanie grafu w głąb / wszerz!"
        ),
        "heads": {
            "H0 (Powiązanie obiektu)": "#1c7ed6",
            "H1 (Lokalizacja / Miejsce)": "#e8590c",
            "H2 (Tło zdania)": "#868e96",
        },
        "edges": {
            (3, 5): [
                (2, 4, "H0 (Powiązanie obiektu)", 0.55),            # jest_w -> klucz
                (2, 3, "H1 (Lokalizacja / Miejsce)", 0.30),         # jest_w -> w_piwnicy
                (2, 5, "H2 (Tło zdania)", 0.15),
            ],
            (2, 4): [
                (1, 1, "H0 (Powiązanie obiektu)", 0.62),            # klucz -> w_skrzyni
                (1, 0, "H0 (Powiązanie obiektu)", 0.26),            # klucz -> Klucz
                (1, 2, "H2 (Tło zdania)", 0.12),
            ],
            (2, 3): [
                (1, 2, "H1 (Lokalizacja / Miejsce)", 0.68),         # w_piwnicy -> skrzynia
                (1, 3, "H1 (Lokalizacja / Miejsce)", 0.32),
            ],
            (2, 5): [
                (1, 4, "H2 (Tło zdania)", 0.70),
                (1, 5, "H2 (Tło zdania)", 0.30),
            ],
            (1, 1): [
                (0, 2, "H0 (Powiązanie obiektu)", 0.58),            # w_skrzyni -> skrzynia
                (0, 1, "H1 (Lokalizacja / Miejsce)", 0.42),
            ],
            (1, 2): [
                (0, 3, "H1 (Lokalizacja / Miejsce)", 0.74),         # skrzynia -> w_piwnicy (cel!)
                (0, 2, "H0 (Powiązanie obiektu)", 0.26),
            ],
            (1, 0): [(0, 0, "H0 (Powiązanie obiektu)", 0.85)],
            (1, 3): [(0, 3, "H1 (Lokalizacja / Miejsce)", 0.88)],
            (1, 4): [(0, 4, "H2 (Tło zdania)", 0.80)],
            (1, 5): [(0, 5, "H2 (Tło zdania)", 0.80)],
        },
        "target_nodes": {(0, 3)},
    },
}


def trace_attention_traversal(scenario_key: str, mode: str = "bfs", threshold: float = 0.2) -> list[dict]:
    """Generuje kroki przeszukiwania drzewa uwagi (BFS warstwami vs DFS sciezkami)."""
    scen = ATTENTION_SCENARIOS[scenario_key]
    tokens = scen["tokens"]
    start = (3, scen["query_idx"])
    edges = scen["edges"]

    def node_label(node):
        layer, tidx = node
        return f"[L{layer}] „{tokens[tidx]}”"

    visited = []
    visited_set = set()
    tree_edges = []  # (u, v, head, alpha)
    cum_score = {start: 1.0}
    steps = []

    if mode == "bfs":
        queue = deque([(start, None, None, 1.0)])
        visited_set.add(start)
        steps.append({
            "visited": [],
            "frontier": [item[0] for item in queue],
            "current": None,
            "tree_edges": [],
            "cum_score": dict(cum_score),
            "title": "Inicjalizacja BFS na drzewie uwagi",
            "msg": (f"Zaczynamy od tokenu wyjściowego <b>{node_label(start)}</b> w najwyższej warstwie (L3). "
                    f"Wstawiamy go do <b>kolejki FIFO</b> – będziemy badać rozpływ uwagi warstwa po warstwie (L3 → L2 → L1 → L0)."),
        })
        while queue:
            u, par, head, sc = queue.popleft()
            visited.append(u)
            added = []
            for l2, t2, h_name, alpha in edges.get(u, []):
                if alpha < threshold:
                    continue
                v = (l2, t2)
                tree_edges.append((u, v, h_name, alpha))
                cum_score[v] = max(cum_score.get(v, 0.0), sc * alpha)
                if v not in visited_set:
                    visited_set.add(v)
                    queue.append((v, u, h_name, sc * alpha))
                    added.append(f"{node_label(v)} (α={alpha:.2f}, {h_name.split()[0]})")
            steps.append({
                "visited": list(visited),
                "frontier": [item[0] for item in queue],
                "current": u,
                "tree_edges": list(tree_edges),
                "cum_score": dict(cum_score),
                "title": f"BFS (wszerz): analiza węzła {node_label(u)}",
                "msg": (f"Pobrano z kolejki FIFO <b>{node_label(u)}</b>. "
                        f"Krawędzie uwagi powyżej progu α ≥ {threshold:.2f}: "
                        f"<b>{', '.join(added) if added else 'brak (warstwa wejściowa L0)'}</b>."),
            })
    else:
        # DFS (w glab - sledzenie pojedynczego obwodu od L3 az do L0)
        call_stack = []

        steps.append({
            "visited": [],
            "frontier": [start],
            "current": None,
            "tree_edges": [],
            "cum_score": dict(cum_score),
            "title": "Inicjalizacja DFS na drzewie uwagi (śledzenie obwodów / circuits)",
            "msg": (f"Zaczynamy od <b>{node_label(start)}</b>. "
                    f"Za pomocą <b>DFS (stosu / rekurencji)</b> podążamy najmocniejszą ścieżką uwagi od razu aż do warstwy wejściowej L0!"),
        })

        def _dfs_att(u, sc):
            visited.append(u)
            visited_set.add(u)
            call_stack.append(u)
            children = [(l2, t2, h_name, alpha) for l2, t2, h_name, alpha in edges.get(u, []) if alpha >= threshold]
            steps.append({
                "visited": list(visited),
                "frontier": list(call_stack),
                "current": u,
                "tree_edges": list(tree_edges),
                "cum_score": dict(cum_score),
                "title": f"DFS (w głąb): wejście do {node_label(u)} [ścieżka: {' → '.join(node_label(x) for x in call_stack)}]",
                "msg": (f"Na stosie wywołań mamy aktywną ścieżkę o długości {len(call_stack)}. "
                        + (f"Schodzimy głębiej do warstwy L{u[0]-1}!" if children else "Dotarliśmy do tokenu wejściowego w warstwie L0 – cofamy się (backtrack)!")),
            })
            for l2, t2, h_name, alpha in children:
                v = (l2, t2)
                tree_edges.append((u, v, h_name, alpha))
                cum_score[v] = max(cum_score.get(v, 0.0), sc * alpha)
                if v not in visited_set:
                    _dfs_att(v, sc * alpha)
            call_stack.pop()

        _dfs_att(start, 1.0)

    return steps


class AttentionTraversalApp:
    """Interaktywna wizualizacja przeszukiwania glowic uwagi (Attention Heads) za pomoca BFS i DFS."""

    def __init__(self):
        import ipywidgets as W

        self.W = W
        st = {"description_width": "initial"}
        self.w_scen = W.Dropdown(
            options=[(v["title"], k) for k, v in ATTENTION_SCENARIOS.items()],
            value="coref",
            description="Przykład (zdanie):",
            style=st,
            layout=W.Layout(width="620px"),
        )
        self.w_mode = W.ToggleButtons(
            options=[
                ("BFS (wszerz: warstwa po warstwie – Attention Rollout)", "bfs"),
                ("DFS (w głąb: śledzenie obwodów – Circuit Tracing)", "dfs"),
            ],
            value="bfs",
            description="Strategia:",
            style=st,
        )
        self.w_thresh = W.FloatSlider(
            value=0.20,
            min=0.10,
            max=0.60,
            step=0.05,
            description="Próg wagi uwagi (α ≥):",
            style=st,
            readout_format=".2f",
            layout=W.Layout(width="340px"),
        )
        self.play = W.Play(value=0, min=0, max=5, step=1, interval=950)
        self.slider = W.IntSlider(value=0, min=0, max=5, step=1, description="Krok:", style=st, layout=W.Layout(width="340px"))
        W.jslink((self.play, "value"), (self.slider, "value"))

        for w in (self.w_scen, self.w_mode, self.w_thresh):
            w.observe(self.rebuild, names="value")
        self.slider.observe(self._render, names="value")

        self.desc_html = W.HTML()
        self.step_html = W.HTML()
        self.out_plot = W.Output()

        self.widget = W.VBox([
            W.HTML("<h3 style='margin:4px 0'>🧠 Przeszukiwanie głowic uwagi (Attention Heads) w modelu Transformer</h3>"),
            self.w_scen,
            self.desc_html,
            W.HBox([self.w_mode, self.w_thresh], layout=W.Layout(flex_flow="row wrap", grid_gap="14px", align_items="center")),
            W.HBox([self.play, self.slider], layout=W.Layout(align_items="center", grid_gap="10px")),
            self.step_html,
            self.out_plot,
        ])
        self.rebuild()

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def rebuild(self, *_):
        self.scen_key = self.w_scen.value
        self.scen = ATTENTION_SCENARIOS[self.scen_key]
        self.desc_html.value = (
            f"<div style='background:#f8f9fa;border:1px solid #dee2e6;padding:6px 10px;"
            f"border-radius:4px;font-size:13px;color:#212529;margin:4px 0'>{self.scen['description']}</div>"
        )
        self.steps = trace_attention_traversal(
            self.scen_key,
            mode=self.w_mode.value,
            threshold=self.w_thresh.value,
        )
        self.play.max = max(0, len(self.steps) - 1)
        self.slider.max = max(0, len(self.steps) - 1)
        self.slider.value = len(self.steps) - 1  # domyslnie pokaz caly przebieg, suwakiem mozna cofnac
        self._render()

    def _render(self, *_):
        idx = min(self.slider.value, len(self.steps) - 1)
        st = self.steps[idx]
        scen = self.scen
        tokens = scen["tokens"]
        n_tok = len(tokens)

        self.step_html.value = (
            f"<div style='background:#fff9db;border-left:4px solid #f08c00;padding:6px 10px;"
            f"margin:4px 0;font-size:13px;color:#212529'>"
            f"<b>Krok {idx} / {len(self.steps) - 1} — {html.escape(st['title'])}</b><br>{st['msg']}</div>"
        )

        with self.out_plot:
            self.out_plot.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(12.2, 5.2))
            xgap = 2.0
            ygap = 1.45
            node_w, node_h = 1.45, 0.52

            # Pozycje (layer, tidx)
            pos = {(layer, t_i): (t_i * xgap, layer * ygap) for layer in range(4) for t_i in range(n_tok)}

            # Etykiety warstw po lewej
            layer_names = {
                3: "Warstwa 3\n(Wyjście / Predykcja)",
                2: "Warstwa 2\n(Głowice uwagi L2)",
                1: "Warstwa 1\n(Głowice uwagi L1)",
                0: "Warstwa 0\n(Tokeny wejściowe)",
            }
            for layer, lname in layer_names.items():
                ax.text(-1.15, layer * ygap, lname, ha="right", va="center", fontsize=8.5, weight="bold", color="#495057")

            visited_order = {node: i + 1 for i, node in enumerate(st["visited"])}
            frontier_set = set(st["frontier"])
            active_edges = {(u, v): (h_name, alpha) for u, v, h_name, alpha in st["tree_edges"]}

            # Wszystkie krawedzie w tle (jasnoszare, jezeli ponizej progu lub jeszcze nieodwiedzone)
            for u, edge_list in scen["edges"].items():
                for l2, t2, h_name, alpha in edge_list:
                    v = (l2, t2)
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    if (u, v) in active_edges:
                        col = scen["heads"][h_name]
                        lw = 1.5 + 3.8 * alpha
                        arr = FancyArrowPatch(
                            (x1, y1 - node_h / 2),
                            (x2, y2 + node_h / 2),
                            arrowstyle="-|>",
                            mutation_scale=11,
                            lw=lw,
                            color=col,
                            alpha=0.9,
                            connectionstyle="arc3,rad=0.06",
                            zorder=3,
                        )
                        ax.add_patch(arr)
                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.text(
                            mx,
                            my,
                            f"α={alpha:.2f}",
                            fontsize=7,
                            weight="bold",
                            color=col,
                            ha="center",
                            va="center",
                            zorder=6,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85),
                        )
                    else:
                        arr = FancyArrowPatch(
                            (x1, y1 - node_h / 2),
                            (x2, y2 + node_h / 2),
                            arrowstyle="-|>",
                            mutation_scale=8,
                            lw=0.8,
                            color="#dee2e6",
                            linestyle="dashed" if alpha < self.w_thresh.value else "solid",
                            alpha=0.55,
                            connectionstyle="arc3,rad=0.06",
                            zorder=1,
                        )
                        ax.add_patch(arr)

            # Rysowanie wezlow (tokenow w warstwach L0..L3)
            for (layer, t_i), (x, y) in pos.items():
                node = (layer, t_i)
                is_start = node == (3, scen["query_idx"])
                is_target = node in scen["target_nodes"]
                is_vis = node in visited_order
                is_front = node in frontier_set
                is_cur = node == st["current"]

                fc = "#f8f9fa"
                ec, lw = "#ced4da", 1.0
                txt_col = "#868e96"

                if layer == 0:
                    fc = "#ffffff"
                    ec = "#adb5bd"
                    txt_col = "#495057"
                if is_front and not is_vis:
                    fc = C_FRONTIER_FILL
                    ec, lw = C_FRONTIER_EDGE, 1.8
                    txt_col = "#212529"
                if is_vis:
                    fc = C_GOAL_FILL if is_target else C_VISITED_FILL
                    ec, lw = ("#2b8a3e", 2.4) if is_target else (C_VISITED_EDGE, 1.9)
                    txt_col = "#212529"
                if is_start:
                    ec, lw = C_INITIAL, 2.8
                    txt_col = "#212529"
                if is_cur:
                    ec, lw = C_CURRENT, 3.4
                    txt_col = "#212529"

                box = FancyBboxPatch(
                    (x - node_w / 2, y - node_h / 2),
                    node_w,
                    node_h,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    fc=fc,
                    ec=ec,
                    lw=lw,
                    zorder=4,
                )
                ax.add_patch(box)
                ax.text(x, y + 0.05, tokens[t_i], ha="center", va="center", fontsize=9, weight="bold", color=txt_col, zorder=5)
                ax.text(x, y - 0.15, f"L{layer}", ha="center", va="center", fontsize=6.8, color="#868e96", zorder=5)

                if is_vis:
                    ax.text(
                        x + node_w / 2,
                        y + node_h / 2 + 0.03,
                        f"#{visited_order[node]}",
                        fontsize=7,
                        weight="bold",
                        color="white",
                        ha="right",
                        va="bottom",
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.14", fc="#1971c2", ec="none"),
                    )

            handles = [Line2D([], [], color=c, lw=2.8, label=h_name) for h_name, c in scen["heads"].items()]
            handles += [
                Patch(fc=C_VISITED_FILL, ec=C_VISITED_EDGE, label="Odwiedzony token w warstwie"),
                Patch(fc=C_GOAL_FILL, ec="#2b8a3e", label="Kluczowy token źródłowy (L0)"),
            ]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8, frameon=False)
            ax.set_xlim(-2.6, (n_tok - 1) * xgap + 1.1)
            ax.set_ylim(-0.6, 3 * ygap + 0.65)
            ax.axis("off")
            fig.tight_layout()
            from IPython.display import display
            display(fig)
            plt.close(fig)


def attention_traversal_app():
    """Uruchamia interaktywna wizualizacje przeszukiwania drzewa uwagi (Attention Heads)."""
    return AttentionTraversalApp()
