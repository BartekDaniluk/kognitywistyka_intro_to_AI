"""
lab_traversal_quiz.py - cwiczenie "BFS i DFS na kartce", ale w przegladarce.

Student dostaje losowy graf (5-10 wezlow, numery 0-9), losowy wezel START i CEL, i:
  1. klikajac wezly zapisuje kolejnosc odwiedzania BFS,
  2. tak samo dla DFS - jako preorder (kolejnosc wejscia) ALBO postorder (kolejnosc wyjscia);
     odpowiedz zaczynajaca sie od START jest sprawdzana jako preorder, kazda inna jako postorder,
  3. klikajac wezly zapisuje najkrotsza sciezke START -> CEL,
a potem klika "Sprawdz" - odpowiedzi robia sie zielone 😀 albo czerwone 😢.

Dwa tryby:
  * "exam" (📝 Egzamin) - sasiadow rozpatrujemy w kolejnosci rosnacych numerow, wiec jest
    dokladnie jedna poprawna kolejnosc (da sie ocenic). DFS = wersja rekurencyjna (ta sama
    kolejnosc co iteracyjny DFS z lab_search, ktory odklada sasiadow na stos odwrotnie).
  * "playground" (🔓 Plac zabaw) - akceptowana jest kazda poprawna kolejnosc BFS/DFS.
W obu trybach po poprawnej odpowiedzi mozna obejrzec WSZYSTKIE poprawne kolejnosci
(galeria), a dla sciezki - wszystkie najkrotsze sciezki. W obu trybach przechodzimy caly
graf (nie zatrzymujemy sie na celu).

Sprawdzanie i galeria korzystaja z tej samej funkcji "dozwolone nastepne wezly":
  BFS - nieodwiedzone wezly odkryte najwczesniej (rodzic najwczesniej na liscie),
  DFS - nieodwiedzeni sasiedzi wezla na szczycie stosu (po cofnieciu ze slepych zaulkow).
Postorder sprawdza postSearch: przeszukiwanie z nawrotami po wyborach DFS (z pamiecia martwych stanow).

Cale cwiczenie dziala w JavaScripcie w komorce wyjsciowej (Colab, JupyterLab, VS Code),
bez ipywidgets.

Uzycie w notebooku:
    import lab_traversal_quiz as quiz
    quiz.traversal_exercise()                          # losowy graf, tryb egzaminu
    quiz.traversal_exercise(n_nodes=10, seed=2026)      # ten sam graf dla calej grupy
    quiz.traversal_exercise(mode="playground")          # dowolna poprawna kolejnosc
"""
from __future__ import annotations

import json
import random
import uuid

DENSITIES = ("sparse", "medium", "dense")
MODES = ("exam", "playground")

# --------------------------------------------------------------------------
# Logika (czysty JS, bez DOM) - testowalna w node: module.exports
# --------------------------------------------------------------------------
_LOGIC_JS = r"""
var TQ = (function () {
  var W = 560, H = 360, COLS = 4, ROWS = 3, R = 19;
  var DENS = { sparse: 0.2, medium: 0.45, dense: 0.8 };
  var STORE_CAP = 5000, COUNT_CAP = 100000;

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
  function shuffle(arr, r) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(r() * (i + 1)), tmp = arr[i];
      arr[i] = arr[j]; arr[j] = tmp;
    }
    return arr;
  }
  function range(n) { var a = []; for (var i = 0; i < n; i++) a.push(i); return a; }
  function asc(x, y) { return x - y; }
  function segPointDist(a, b, p) {
    var dx = b[0] - a[0], dy = b[1] - a[1];
    var t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy);
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
  }
  function orient(a, b, c) { return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]); }
  function segsCross(p1, p2, p3, p4) {
    var d1 = orient(p3, p4, p1), d2 = orient(p3, p4, p2), d3 = orient(p1, p2, p3), d4 = orient(p1, p2, p4);
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
  }

  function bfsDist(adj, s) {
    var dist = adj.map(function () { return -1; }), q = [s];
    dist[s] = 0;
    while (q.length) {
      var u = q.shift();
      adj[u].forEach(function (v) { if (dist[v] < 0) { dist[v] = dist[u] + 1; q.push(v); } });
    }
    return dist;
  }
  // BFS: kolejka FIFO, sasiedzi rosnaco (kolejnosc egzaminacyjna)
  function bfsOrder(adj, s) {
    var seen = {}, order = [], q = [s];
    seen[s] = true;
    while (q.length) {
      var u = q.shift();
      order.push(u);
      adj[u].forEach(function (v) { if (!seen[v]) { seen[v] = true; q.push(v); } });
    }
    return order;
  }
  // DFS rekurencyjny, sasiedzi rosnaco (kolejnosc egzaminacyjna)
  function dfsOrder(adj, s) {
    var seen = {}, order = [];
    (function go(u) {
      seen[u] = true; order.push(u);
      adj[u].forEach(function (v) { if (!seen[v]) go(v); });
    })(s);
    return order;
  }
  // DFS rekurencyjny, postorder: wezel zapisujemy, gdy WYCHODZIMY z niego (wszyscy sasiedzi juz odwiedzeni)
  function dfsPostOrder(adj, s) {
    var seen = {}, order = [];
    (function go(u) {
      seen[u] = true;
      adj[u].forEach(function (v) { if (!seen[v]) go(v); });
      order.push(u);
    })(s);
    return order;
  }
  function shortestPath(adj, s, t) {
    var par = {}, q = [s];
    par[s] = -1;
    while (q.length) {
      var u = q.shift();
      if (u === t) break;
      adj[u].forEach(function (v) { if (!(v in par)) { par[v] = u; q.push(v); } });
    }
    var path = [];
    for (var x = t; x !== -1; x = par[x]) path.unshift(x);
    return path;
  }
  function allShortestPaths(adj, s, t, cap) {
    var dist = bfsDist(adj, s), out = [];
    (function go(path) {
      if (out.length >= (cap || 50)) return;
      var u = path[path.length - 1];
      if (u === t) { out.push(path.slice()); return; }
      adj[u].forEach(function (v) {
        if (dist[v] === dist[u] + 1 && dist[v] <= dist[t]) { path.push(v); go(path); path.pop(); }
      });
    })([s]);
    return out;
  }

  // ---- "dozwolone nastepne wezly" - wspolne dla sprawdzania i wyliczania kolejnosci ----
  function walker(n, kind) {
    var vis = [], ppos = [];
    for (var i = 0; i < n; i++) { vis.push(false); ppos.push(-1); }
    return { kind: kind, order: [], vis: vis, ppos: ppos, stack: [], post: [] };
  }
  function clone(w) {
    return { kind: w.kind, order: w.order.slice(), vis: w.vis.slice(), ppos: w.ppos.slice(), stack: w.stack.slice(),
             post: w.post.slice() };
  }
  function visit(w, adj, v) {
    var p = w.order.length;
    w.order.push(v); w.vis[v] = true;
    if (w.kind === "bfs") adj[v].forEach(function (x) { if (!w.vis[x] && w.ppos[x] < 0) w.ppos[x] = p; });
    else w.stack.push(v);
  }
  // BFS: nieodwiedzone wezly, ktorych rodzic (najwczesniej odwiedzony sasiad) stoi najwczesniej
  // DFS: nieodwiedzeni sasiedzi wezla na szczycie stosu (puste galezie zdejmujemy - cofanie)
  function candidates(w, adj) {
    if (w.kind === "bfs") {
      var best = Infinity, c = [];
      for (var v = 0; v < adj.length; v++) {
        if (w.vis[v] || w.ppos[v] < 0) continue;
        if (w.ppos[v] < best) { best = w.ppos[v]; c = [v]; } else if (w.ppos[v] === best) c.push(v);
      }
      return c;
    }
    while (w.stack.length) {
      var top = w.stack[w.stack.length - 1];
      var c2 = adj[top].filter(function (x) { return !w.vis[x]; });
      if (c2.length) return c2;
      w.post.push(w.stack.pop());   // wychodzimy z wezla -> postorder
    }
    return [];
  }
  // Analiza odpowiedzi: pierwszy niemozliwy krok i pierwsze zlamanie zasady "rosnaco"
  function analyze(adj, s, kind, given) {
    var res = { invalid: null, ruleBreak: null };
    if (given[0] !== s) { res.invalid = { idx: 0, v: given[0], why: "start" }; return res; }
    var w = walker(adj.length, kind);
    visit(w, adj, s);
    for (var i = 1; i < given.length; i++) {
      var c = candidates(w, adj), v = given[i];
      if (c.indexOf(v) < 0) {
        var inv = { idx: i, v: v };
        if (kind === "bfs") {
          if (w.ppos[v] < 0) inv.why = "notadj";
          else { inv.why = "queue"; inv.waitingFor = w.order[w.ppos[c[0]]]; }
        } else { inv.why = "dfs"; inv.top = w.stack[w.stack.length - 1]; }
        res.invalid = inv;
        return res;
      }
      var m = Math.min.apply(null, c);
      if (v !== m && !res.ruleBreak) res.ruleBreak = { idx: i, v: v, cands: c.slice().sort(asc) };
      visit(w, adj, v);
    }
    return res;
  }
  // Postorder: czy "given" (albo jego poczatek) to postorder jakiegos przebiegu DFS?
  // Przeszukiwanie z nawrotami po wyborach DFS; ucinamy galaz, gdy zakonczony wezel nie zgadza
  // sie z odpowiedzia. asc=true -> tylko przebieg "rosnaco" (egzamin).
  // best = dlugosc najdluzszego pasujacego poczatku odpowiedzi.
  function postSearch(adj, s, given, asc) {
    var n = adj.length, best = 0, dead = {};
    if (!given.length) return { best: 0, ok: true };
    function rec(vis, stack, k) {
      stack = stack.slice();
      while (stack.length) {
        var top = stack[stack.length - 1];
        if (adj[top].some(function (x) { return !vis[x]; })) break;
        stack.pop();
        if (given[k] !== top) return false;
        k++; if (k > best) best = k;
        if (k === given.length) return true;
      }
      if (!stack.length) return false;
      var key = vis.join("") + "|" + stack.join(",");
      if (dead[key]) return false;
      var c = adj[stack[stack.length - 1]].filter(function (x) { return !vis[x]; });
      if (asc) c = [Math.min.apply(null, c)];
      for (var i = 0; i < c.length; i++) {
        var v2 = vis.slice(); v2[c[i]] = 1;
        if (rec(v2, stack.concat([c[i]]), k)) return true;
      }
      dead[key] = true;
      return false;
    }
    var vis = [];
    for (var i = 0; i < n; i++) vis.push(0);
    vis[s] = 1;
    var ok = rec(vis, [s], 0);
    return { best: best, ok: ok };
  }
  // Wszystkie poprawne kolejnosci (wg preorder leksykograficznie: #1 = kolejnosc egzaminacyjna).
  // kind: "bfs" | "dfs" (preorder) | "dfspost" (rozne postordery, w kolejnosci pierwszego wystapienia)
  function enumerate(adj, s, kind, storeCap, countCap) {
    storeCap = storeCap || STORE_CAP; countCap = countCap || COUNT_CAP;
    var post = kind === "dfspost", seenPost = {}, runs = 0;
    var list = [], count = 0, w0 = walker(adj.length, post ? "dfs" : kind);
    visit(w0, adj, s);
    (function rec(w) {
      if (runs >= countCap) return;
      var c = candidates(w, adj);
      if (!c.length) {
        runs++;
        var seq = post ? w.post : w.order;
        if (post) { var key = seq.join(","); if (seenPost[key]) return; seenPost[key] = true; }
        count++;
        if (list.length < storeCap) list.push(seq.slice());
        return;
      }
      c.sort(asc).forEach(function (v) { var w2 = clone(w); visit(w2, adj, v); rec(w2); });
    })(w0);
    return { count: count, capped: runs >= countCap, list: list };
  }

  function generate(n, density, seed) {
    var r = rng(seed), cw = W / COLS, ch = H / ROWS;
    var cells = shuffle(range(COLS * ROWS), r).slice(0, n);
    var pos = cells.map(function (c) {
      var col = c % COLS, row = Math.floor(c / COLS);
      return [(col + 0.5 + (r() - 0.5) * 0.45) * cw, (row + 0.5 + (r() - 0.5) * 0.4) * ch];
    });
    var edges = [], has = {};
    function key(a, b) { return a < b ? a + "-" + b : b + "-" + a; }
    function d(a, b) { return Math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]); }
    function ok(a, b) {
      if (has[key(a, b)]) return false;
      for (var k = 0; k < n; k++)
        if (k !== a && k !== b && segPointDist(pos[a], pos[b], pos[k]) < R * 1.7) return false;
      for (var i = 0; i < edges.length; i++) {
        var u = edges[i][0], v = edges[i][1];
        if (u === a || u === b || v === a || v === b) continue;
        if (segsCross(pos[a], pos[b], pos[u], pos[v])) return false;
      }
      return true;
    }
    function add(a, b) { edges.push(a < b ? [a, b] : [b, a]); has[key(a, b)] = true; }
    // drzewo rozpinajace: kazdy kolejny wezel laczymy z najblizszym juz polaczonym
    var order = shuffle(range(n), r), conn = [order[0]];
    for (var i = 1; i < n; i++) {
      var v = order[i];
      var cands = conn.slice().sort(function (x, y) { return d(v, x) - d(v, y); });
      var u = cands.filter(function (c) { return ok(v, c); })[0];
      add(v, u === undefined ? cands[0] : u);
      conn.push(v);
    }
    // dodatkowe krotkie krawedzie (cykle), bez przeciec
    var pairs = [];
    for (var a = 0; a < n; a++) for (var b = a + 1; b < n; b++) pairs.push([a, b]);
    pairs.sort(function (x, y) { return d(x[0], x[1]) - d(y[0], y[1]); });
    var p = DENS[density] === undefined ? DENS.medium : DENS[density], maxLen = Math.hypot(cw, ch) * 1.3;
    pairs.forEach(function (pr) {
      if (d(pr[0], pr[1]) <= maxLen && r() < p && ok(pr[0], pr[1])) add(pr[0], pr[1]);
    });
    var adj = range(n).map(function () { return []; });
    edges.forEach(function (e) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); });
    adj.forEach(function (l) { l.sort(asc); });
    var start = Math.floor(r() * n), dist = bfsDist(adj, start);
    var far = range(n).filter(function (x) { return dist[x] >= 2; });
    if (!far.length) far = range(n).filter(function (x) { return x !== start; });
    var target = far[Math.floor(r() * far.length)];
    return { n: n, pos: pos, edges: edges, adj: adj, start: start, target: target, seed: seed,
             dist: dist, bfs: bfsOrder(adj, start), dfs: dfsOrder(adj, start), dfspost: dfsPostOrder(adj, start),
             path: shortestPath(adj, start, target), paths: allShortestPaths(adj, start, target, 50) };
  }

  function plural(k, one, few, many) {
    if (k === 1) return one;
    return (k % 10 >= 2 && k % 10 <= 4 && (k % 100 < 12 || k % 100 > 14)) ? few : many;
  }
  function nSteps(k) { return k + " " + plural(k, "krok", "kroki", "kroków"); }
  function fmt(k) { return String(k).replace(/\B(?=(\d{3})+(?!\d))/g, " "); }

  var KIND = { bfs: "BFS", dfs: "DFS" };
  var PRE = "preorder (kolejność wejścia do węzłów)", POST = "postorder (kolejność wyjścia z węzłów)";
  // DFS: odpowiedz zaczynajaca sie od START sprawdzamy jako preorder, kazda inna - jako postorder
  function checkPost(g, given, mode) {
    var res = postSearch(g.adj, g.start, given, false);
    if (mode === "exam") {
      var ex = postSearch(g.adj, g.start, given, true);
      if (!ex.ok && (res.ok || res.best > ex.best))
        return { ok: false, rule: true, which: "post", msg: "Taki postorder jest możliwy w DFS, ale łamie zasadę „rosnąco”: " +
                 "pierwsza różnica na pozycji " + (ex.best + 1) + " (węzeł " + given[ex.best] + ") – idąc w głąb, " +
                 "DFS zawsze wybiera najmniejszego nieodwiedzonego sąsiada." };
      res = ex;
    }
    if (!res.ok) {
      var v = given[res.best];
      var why = v === g.start
        ? "z węzła START wychodzimy na samym końcu – dopiero gdy cały graf jest odwiedzony."
        : "w tym momencie DFS nie może skończyć węzła " + v + ". W postorder zapisujemy węzeł, gdy z niego " +
          "<b>wychodzimy</b> (cofamy się) – czyli gdy wszyscy jego sąsiedzi są już odwiedzeni.";
      return { ok: false, which: "post", msg: "Odpowiedź nie zaczyna się od START, więc sprawdzam ją jako " + POST +
               ". Pierwszy błąd na pozycji " + (res.best + 1) + " (węzeł " + v + "): " + why };
    }
    if (given.length < g.n)
      return { ok: false, which: "post", msg: "Dotąd dobrze – to początek poprawnego postorder, ale brakuje węzłów: " +
               given.length + " z " + g.n + "." };
    return { ok: true, which: "post", msg: "Poprawny " + POST + "!" };
  }
  function checkTraversal(g, kind, given, mode) {
    if (!given.length) return { ok: false, msg: "Lista jest pusta – kliknij węzły na grafie." };
    if (kind === "dfs" && given[0] !== g.start && g.n > 1) return checkPost(g, given, mode);
    var a = analyze(g.adj, g.start, kind, given), inv = a.invalid, rb = a.ruleBreak;
    if (mode === "exam" && rb && (!inv || rb.idx < inv.idx))
      return { ok: false, rule: true, which: "pre", msg: (kind === "dfs" ? "Taki preorder jest możliwy w DFS" : "Taka kolejność jest możliwa w BFS") + ", ale łamie zasadę „rosnąco”: na pozycji " +
               (rb.idx + 1) + " do wyboru były węzły " + rb.cands.join(", ") + " – bierzemy najmniejszy, czyli " + rb.cands[0] + "." };
    if (inv) {
      var why = {
        start: "kolejność musi zaczynać się od węzła START (" + g.start + ").",
        notadj: "ten węzeł nie sąsiaduje z żadnym z dotąd odwiedzonych.",
        queue: "w kolejce FIFO czekają jeszcze węzły odkryte wcześniej (przez węzeł " + inv.waitingFor + ") – one idą pierwsze.",
        dfs: "DFS jest teraz w węźle " + inv.top + " i musi wejść do jednego z jego nieodwiedzonych sąsiadów (cofamy się dopiero ze ślepego zaułka)."
      }[inv.why];
      return { ok: false, which: "pre", msg: "Pierwszy błąd na pozycji " + (inv.idx + 1) + " (węzeł " + inv.v + "): " + why };
    }
    if (given.length < g.n)
      return { ok: false, which: "pre", msg: "Dotąd dobrze" + (kind === "dfs" ? " – to początek poprawnego preorder" : "") +
               ", ale brakuje węzłów: odwiedzono " + given.length + " z " + g.n + "." };
    return { ok: true, which: "pre", msg: kind === "dfs" ? "Poprawny " + PRE + "!" : "Kolejność poprawna!" };
  }
  // zgodnosc wsteczna: porownanie z jedna oczekiwana kolejnoscia
  function checkOrder(expected, given) {
    if (!given.length) return { ok: false, msg: "Lista jest pusta – kliknij węzły na grafie." };
    for (var i = 0; i < given.length; i++)
      if (i >= expected.length || given[i] !== expected[i])
        return { ok: false, msg: "Pierwszy błąd na pozycji " + (i + 1) + " (węzeł " + given[i] + ")." };
    if (given.length < expected.length)
      return { ok: false, msg: "Dotąd dobrze, ale brakuje węzłów: odwiedzono " + given.length + " z " + expected.length + "." };
    return { ok: true, msg: "Kolejność poprawna!" };
  }
  function checkPath(g, path) {
    if (!path.length) return { ok: false, msg: "Ścieżka jest pusta – kliknij węzły od START do CEL." };
    if (path[0] !== g.start) return { ok: false, msg: "Ścieżka musi zaczynać się w węźle START (" + g.start + ")." };
    for (var i = 0; i + 1 < path.length; i++)
      if (g.adj[path[i]].indexOf(path[i + 1]) < 0)
        return { ok: false, msg: "Między " + path[i] + " a " + path[i + 1] + " nie ma krawędzi." };
    if (path[path.length - 1] !== g.target)
      return { ok: false, msg: "Ścieżka musi kończyć się w węźle CEL (" + g.target + ")." };
    var best = g.dist[g.target], len = path.length - 1;
    if (len > best)
      return { ok: false, msg: "Ścieżka poprawna, ale nie najkrótsza: " + nSteps(len) + ", a da się w " + nSteps(best) + "." };
    return { ok: true, msg: "Najkrótsza ścieżka – " + nSteps(len) + "!" };
  }

  return { W: W, H: H, R: R, rng: rng, generate: generate, bfsOrder: bfsOrder, dfsOrder: dfsOrder,
           bfsDist: bfsDist, shortestPath: shortestPath, allShortestPaths: allShortestPaths,
           dfsPostOrder: dfsPostOrder, postSearch: postSearch, analyze: analyze, enumerate: enumerate, checkTraversal: checkTraversal, checkOrder: checkOrder,
           checkPath: checkPath, nSteps: nSteps, plural: plural, fmt: fmt };
})();
if (typeof module === "object" && module && module.exports && typeof window === "undefined") module.exports = TQ;
"""

# --------------------------------------------------------------------------
# Interfejs (DOM + SVG)
# --------------------------------------------------------------------------
_UI_JS = r"""
(function () {
  var root = document.getElementById("__ROOT__");
  var CFG = __CFG__;
  var TASKS = [
    { id: "bfs", color: "#1971c2", tint: "#d0ebff", title: "1. Kolejność odwiedzania – BFS (wszerz)" },
    { id: "dfs", color: "#e8590c", tint: "#ffe8cc", title: "2. DFS (w głąb) – preorder albo postorder" },
    { id: "path", color: "#9c36b5", tint: "#f3d9fa", title: "3. Najkrótsza ścieżka START → CEL" }
  ];
  var RULES = {
    exam: "📏 <b>Zasady (egzamin):</b> zaczynamy w węźle <b>START</b>; sąsiadów rozpatrujemy zawsze w kolejności " +
          "<b>rosnących numerów</b> – dzięki temu jest dokładnie jedna poprawna kolejność; odwiedzamy <b>cały graf</b> " +
          "(nie zatrzymujemy się na celu). DFS – wersja rekurencyjna (tak samo jak stos LIFO w symulatorze). Graf jest nieskierowany.",
    playground: "🔓 <b>Plac zabaw:</b> zaczynamy w węźle <b>START</b> i odwiedzamy <b>cały graf</b>, ale sąsiadów możesz brać " +
          "w <b>dowolnej kolejności</b> – zaliczona jest każda kolejność, którą naprawdę mógłby dać BFS albo DFS. " +
          "Po poprawnej odpowiedzi zobaczysz, ile takich kolejności w ogóle jest."
  };
  var DFS_NOTE = "<br>🔁 <b>DFS</b> możesz zapisać na dwa sposoby: <b>preorder</b> – węzeł zapisujemy, gdy do niego " +
          "<b>wchodzimy</b> (lista zaczyna się od START), albo <b>postorder</b> – gdy z niego <b>wychodzimy</b>, " +
          "bo wszyscy jego sąsiedzi są już odwiedzeni (lista kończy się na START). Program sam rozpozna, który zapisano.";
  var KEYNAME = {
    bfs: "BFS", dfs: "DFS preorder", dfspost: "DFS postorder"
  };
  var S = { g: null, mode: CFG.mode, active: "bfs", ans: { bfs: [], dfs: [], path: [] }, res: {}, reveal: false,
            note: "", enumc: {}, gal: null, timer: null };
  var NS = "http://www.w3.org/2000/svg";
  function $(sel) { return root.querySelector(sel); }
  function task(id) { return TASKS.filter(function (t) { return t.id === id; })[0]; }
  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function chipsHtml(seq, color) {
    return seq.map(function (v) { return "<span class='tq-chip' style='background:" + color + "'>" + v + "</span>"; })
              .join("<span class='tq-arrow'>→</span>");
  }
  function orders(kind) {
    if (!S.enumc[kind]) S.enumc[kind] = TQ.enumerate(S.g.adj, S.g.start, kind);
    return S.enumc[kind];
  }
  function countText(e) {
    var k = e.count;
    return (e.capped ? "ponad " : "") + TQ.fmt(k) + " " +
           (e.capped ? "poprawnych kolejności" : TQ.plural(k, "poprawna kolejność", "poprawne kolejności", "poprawnych kolejności"));
  }

  // ---- galeria wszystkich kolejnosci ----
  function stopPlay() { if (S.timer) { clearInterval(S.timer); S.timer = null; } }
  function closeGallery() { stopPlay(); S.gal = null; }
  function openGallery(task, key) {
    stopPlay();
    var list = orders(key).list, mine = S.ans[task].join(","), idx = 0;
    for (var i = 0; i < list.length; i++) if (list[i].join(",") === mine) { idx = i; break; }
    S.gal = { task: task, key: key, idx: idx };
    S.active = task;
    render();
  }
  function galStep(d) {
    var list = orders(S.gal.key).list;
    S.gal.idx = (S.gal.idx + d + list.length) % list.length;
    render();
  }
  function togglePlay() {
    if (S.timer) { stopPlay(); render(); return; }
    S.timer = setInterval(function () {
      if (!document.body.contains(root) || !S.gal) { stopPlay(); return; }
      galStep(1);
    }, 700);
    render();
  }

  function newGraph(seed) {
    closeGallery();
    var n = parseInt($(".tq-n").value, 10), dens = $(".tq-dens").value;
    S.g = TQ.generate(n, dens, seed);
    $(".tq-seed").value = seed;
    S.ans = { bfs: [], dfs: [], path: [] }; S.res = {}; S.reveal = false; S.note = ""; S.enumc = {};
    $(".tq-show").textContent = "👁 Pokaż rozwiązanie";
    render();
  }
  function setMode(m) {
    if (S.mode === m) return;
    closeGallery();
    S.mode = m; S.res = {}; S.note = "";
    render();
  }
  function randomSeed() { return Math.floor(Math.random() * 100000); }

  function clickNode(v) {
    if (S.gal) closeGallery();
    var list = S.ans[S.active];
    if (list.indexOf(v) >= 0) { S.note = "Węzeł " + v + " już jest na tej liście."; render(); return; }
    list.push(v); S.note = ""; delete S.res[S.active]; render();
  }

  function drawGraph() {
    var g = S.g, svg = $(".tq-svg"), t = task(S.active);
    var list = (S.gal && S.gal.task === S.active) ? orders(S.gal.key).list[S.gal.idx] : S.ans[S.active];
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    function k(a, b) { return Math.min(a, b) + "-" + Math.max(a, b); }
    var pathEdges = {}, shortEdges = {};
    if (S.active === "path") {
      for (var i = 0; i + 1 < list.length; i++) pathEdges[k(list[i], list[i + 1])] = true;
      if (S.res.path || S.reveal)
        g.paths.forEach(function (p) { for (var j = 0; j + 1 < p.length; j++) shortEdges[k(p[j], p[j + 1])] = true; });
    }
    g.edges.forEach(function (e) {
      var key = k(e[0], e[1]), on = pathEdges[key], sh = shortEdges[key], a = g.pos[e[0]], b = g.pos[e[1]];
      if (sh && !on)
        el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: t.color, "stroke-opacity": 0.55,
                     "stroke-width": 4, "stroke-dasharray": "7 6", "stroke-linecap": "round" }, svg);
      else
        el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: on ? t.color : "#adb5bd",
                     "stroke-width": on ? 5 : 2.5, "stroke-linecap": "round" }, svg);
    });
    for (var v = 0; v < g.n; v++) (function (v) {
      var p = g.pos[v], idx = list.indexOf(v), grp = el("g", { "class": "tq-node" }, svg);
      grp.addEventListener("click", function () { clickNode(v); });
      var fill = idx >= 0 ? t.tint : (v === g.target ? "#ffe066" : "#ffffff");
      var stroke = v === g.start ? "#2f9e44" : (idx >= 0 ? t.color : "#495057");
      el("circle", { cx: p[0], cy: p[1], r: TQ.R, fill: fill, stroke: stroke,
                     "stroke-width": v === g.start ? 4.5 : (idx >= 0 ? 3 : 2) }, grp);
      el("text", { x: p[0], y: p[1] + 6, "text-anchor": "middle", "font-size": 17, "font-weight": 700,
                   fill: "#212529" }, grp, String(v));
      if (v === g.start || v === g.target)
        el("text", { x: p[0], y: p[1] + TQ.R + 15, "text-anchor": "middle", "font-size": 11, "font-weight": 700,
                     fill: v === g.start ? "#2f9e44" : "#b08900" }, grp, v === g.start ? "START" : "CEL");
      if (idx >= 0) {
        el("circle", { cx: p[0] + 16, cy: p[1] - 16, r: 10, fill: t.color, stroke: "#fff", "stroke-width": 1.5 }, grp);
        el("text", { x: p[0] + 16, y: p[1] - 12, "text-anchor": "middle", "font-size": 11, "font-weight": 700,
                     fill: "#fff" }, grp, String(idx + 1));
      }
    })(v);
  }

  // dodatkowe informacje pod zadaniem BFS/DFS: liczba kolejnosci + galeria
  // (dla DFS osobno preorder i postorder; najpierw ten, ktory student zapisal)
  function extraTraversal(t) {
    var res = S.res[t.id], html = "";
    if (!((res && res.ok) || S.reveal)) return "";
    var keys = t.id === "bfs" ? ["bfs"] : (res && res.ok && res.which === "post" ? ["dfspost", "dfs"] : ["dfs", "dfspost"]);
    keys.forEach(function (key) {
      var e = orders(key), name = KEYNAME[key], mineKey = res && res.ok && (key === "dfspost") === (res.which === "post");
      if (S.mode === "exam")
        html += "<div class='tq-more'>🔢 Bez zasady „rosnąco” " + name + " mógłby dać na tym grafie <b>" + countText(e) + "</b>.";
      else {
        var mine = S.ans[t.id].join(","), pos = -1;
        if (mineKey) for (var i = 0; i < e.list.length; i++) if (e.list[i].join(",") === mine) { pos = i; break; }
        html += "<div class='tq-more'>🔢 " + name + " ma na tym grafie <b>" + countText(e) + "</b>" +
                (pos >= 0 ? " – Twoja to <b>#" + TQ.fmt(pos + 1) + "</b>." : ".") +
                (pos === 0 ? " To akurat kolejność „egzaminacyjna” (rosnąco)." : "");
      }
      if (!(S.gal && S.gal.key === key))
        html += " <button class='tq-gal-open' data-key='" + key + "'>🎞 Pokaż wszystkie</button>";
      html += "</div>";
      if (S.gal && S.gal.key === key) html += galleryHtml(t, e);
    });
    return html;
  }
  function galleryHtml(t, e) {
    var html = "";
    {
      var shown = e.list.length, seq = e.list[S.gal.idx];
      html += "<div class='tq-gal'>" +
        "<button class='tq-gal-prev' title='Poprzednia'>◀</button>" +
        "<button class='tq-gal-play'>" + (S.timer ? "⏸ Pauza" : "⏯ Odtwarzaj") + "</button>" +
        "<button class='tq-gal-next' title='Następna'>▶</button>" +
        "<span class='tq-gal-count'>#" + TQ.fmt(S.gal.idx + 1) + " / " + TQ.fmt(shown) +
        (e.capped || shown < e.count ? " (pokazuję pierwsze " + TQ.fmt(shown) + ")" : "") + "</span>" +
        "<button class='tq-gal-close' title='Zamknij galerię'>✕</button>" +
        "<div class='tq-seq'>" + chipsHtml(seq, t.color) + "</div>" +
        (S.gal.idx === 0 ? "<div class='tq-sol'>#1 = kolejność egzaminacyjna (sąsiedzi rosnąco)</div>" : "") +
        "</div>";
    }
    return html;
  }
  function extraPath(t) {
    if (!S.res.path && !S.reveal) return "";
    var g = S.g, ps = g.paths;
    return "<div class='tq-more'>📍 " + (ps.length === 1 ? "Jest tylko <b>jedna</b> najkrótsza ścieżka" :
             "Najkrótszych ścieżek jest <b>" + ps.length + "</b>") + " (" + TQ.nSteps(g.dist[g.target]) + "; na grafie przerywaną linią):" +
           ps.map(function (p) { return "<div class='tq-seq'>" + chipsHtml(p, t.color) + "</div>"; }).join("") + "</div>";
  }

  function drawTasks() {
    var box = $(".tq-tasks");
    box.innerHTML = "";
    TASKS.forEach(function (t) {
      var res = S.res[t.id], active = S.active === t.id;
      var row = document.createElement("div");
      row.className = "tq-row" + (active ? " tq-active" : "") + (res ? (res.ok ? " tq-ok" : " tq-bad") : "");
      row.style.setProperty("--c", t.color);
      var seq = S.ans[t.id];
      var chips = seq.length ? chipsHtml(seq, t.color)
        : "<span class='tq-ph'>" + (active ? "kliknij węzły na grafie…" : "kliknij tutaj, żeby wybrać to zadanie") + "</span>";
      var tag = (S.mode === "exam" && t.id !== "path") ? "<span class='tq-tag'>↑ rosnąco</span>" : "";
      var sol = "";
      if (S.reveal && t.id !== "path") {
        var lead = S.mode === "exam" ? "Rozwiązanie" : "Przykład (kolejność egzaminacyjna)";
        sol = t.id === "bfs"
          ? "<div class='tq-sol'>" + lead + ": <b>" + S.g.bfs.join(" → ") + "</b></div>"
          : "<div class='tq-sol'>" + lead + " – preorder: <b>" + S.g.dfs.join(" → ") + "</b></div>" +
            "<div class='tq-sol'>" + lead + " – postorder: <b>" + S.g.dfspost.join(" → ") + "</b></div>";
      }
      row.innerHTML =
        "<div class='tq-head'><span class='tq-radio'></span><b>" + t.title + "</b>" + tag +
        "<span class='tq-btns'><button class='tq-undo' title='Cofnij ostatni'>⌫</button><button class='tq-clear' title='Wyczyść'>✕</button></span></div>" +
        "<div class='tq-seq'>" + chips + "</div>" +
        (res ? "<div class='tq-res'><span class='tq-face'>" + (res.ok ? "😀" : "😢") + "</span><span>" + res.msg + "</span></div>" : "") +
        sol + (t.id === "path" ? extraPath(t) : extraTraversal(t));
      row.addEventListener("click", function () { if (S.active !== t.id) { S.active = t.id; S.note = ""; render(); } });
      function on(sel, fn) {
        row.querySelectorAll(sel).forEach(function (b) {
          b.addEventListener("click", function (ev) { ev.stopPropagation(); fn(b); });
        });
      }
      on(".tq-undo", function () { if (S.gal) closeGallery(); S.active = t.id; seq.pop(); delete S.res[t.id]; render(); });
      on(".tq-clear", function () { if (S.gal) closeGallery(); S.active = t.id; seq.length = 0; delete S.res[t.id]; render(); });
      on(".tq-gal-open", function (b) { openGallery(t.id, b.getAttribute("data-key")); });
      on(".tq-gal-prev", function () { stopPlay(); galStep(-1); });
      on(".tq-gal-next", function () { stopPlay(); galStep(1); });
      on(".tq-gal-play", togglePlay);
      on(".tq-gal-close", function () { closeGallery(); render(); });
      box.appendChild(row);
    });
  }

  function render() {
    var g = S.g;
    root.querySelectorAll(".tq-mode button").forEach(function (b) {
      b.className = b.getAttribute("data-mode") === S.mode ? "tq-on" : "";
    });
    $(".tq-rule").innerHTML = RULES[S.mode] + DFS_NOTE;
    $(".tq-info").innerHTML = "START: <b class='tq-start'>" + g.start + "</b> &nbsp;·&nbsp; CEL: <b class='tq-goal'>" + g.target +
      "</b> &nbsp;·&nbsp; węzłów: " + g.n + ", krawędzi: " + g.edges.length;
    $(".tq-note").textContent = S.note;
    drawGraph(); drawTasks();
    var all = TASKS.every(function (t) { return S.res[t.id] && S.res[t.id].ok; });
    var any = TASKS.some(function (t) { return S.res[t.id]; });
    $(".tq-summary").innerHTML = all ? "🎉 Wszystko dobrze! Zauważ: kolejność odwiedzania BFS to <i>nie</i> to samo co najkrótsza ścieżka – " +
      "ścieżkę odczytujemy z drzewa przeszukiwania (kto kogo odkrył), a nie z listy odwiedzin."
      : any ? "Popraw czerwone odpowiedzi i sprawdź jeszcze raz." : "";
  }

  function check() {
    var g = S.g;
    S.res = { bfs: TQ.checkTraversal(g, "bfs", S.ans.bfs, S.mode), dfs: TQ.checkTraversal(g, "dfs", S.ans.dfs, S.mode),
              path: TQ.checkPath(g, S.ans.path) };
    S.note = ""; render();
  }

  $(".tq-new").addEventListener("click", function () { newGraph(randomSeed()); });
  $(".tq-seed").addEventListener("change", function () { newGraph(parseInt(this.value, 10) || 0); });
  $(".tq-n").addEventListener("change", function () { newGraph(S.g.seed); });
  $(".tq-dens").addEventListener("change", function () { newGraph(S.g.seed); });
  $(".tq-check").addEventListener("click", check);
  $(".tq-show").addEventListener("click", function () {
    S.reveal = !S.reveal; this.textContent = S.reveal ? "🙈 Ukryj rozwiązanie" : "👁 Pokaż rozwiązanie"; render();
  });
  root.querySelectorAll(".tq-mode button").forEach(function (b) {
    b.addEventListener("click", function () { setMode(b.getAttribute("data-mode")); });
  });
  $(".tq-n").value = String(CFG.n);
  $(".tq-dens").value = CFG.density;
  newGraph(CFG.seed);
})();
"""

_CSS = r"""
<style>
.tq-card{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#212529;background:#ffffff;
  border:1px solid #ced4da;border-radius:10px;padding:12px 14px;max-width:1000px;font-size:14px;line-height:1.4}
.tq-card *{box-sizing:border-box}
.tq-card h4{margin:0 0 8px 0;font-size:17px;color:#212529}
.tq-top{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between;margin-bottom:8px}
.tq-mode{display:inline-flex;border:1px solid #adb5bd;border-radius:8px;overflow:hidden}
.tq-card .tq-mode button{border:none;border-radius:0;background:#f8f9fa;padding:5px 12px}
.tq-card .tq-mode button + button{border-left:1px solid #adb5bd}
.tq-card .tq-mode button.tq-on{background:#343a40;color:#fff;font-weight:700}
.tq-controls{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin-bottom:6px}
.tq-controls label{display:flex;gap:6px;align-items:center;color:#212529}
.tq-card select,.tq-card input{font-size:14px;padding:3px 6px;border:1px solid #adb5bd;border-radius:6px;
  background:#fff;color:#212529}
.tq-card input.tq-seed{width:90px}
.tq-card button{font-size:14px;padding:5px 12px;border:1px solid #adb5bd;border-radius:6px;background:#f8f9fa;
  color:#212529;cursor:pointer}
.tq-card button:hover{background:#e9ecef}
.tq-card button.tq-check{background:#2f9e44;border-color:#2b8a3e;color:#fff;font-weight:700}
.tq-card button.tq-check:hover{background:#2b8a3e}
.tq-rule{background:#f1f3f5;border-radius:6px;padding:6px 10px;margin:6px 0;color:#343a40;font-size:13px}
.tq-info{margin:4px 0;color:#212529}
.tq-start{color:#2f9e44}.tq-goal{color:#b08900}
.tq-body{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
.tq-svgbox{flex:1 1 360px;min-width:260px;border:1px solid #e9ecef;border-radius:8px;background:#fff}
.tq-svg{width:100%;height:auto;display:block}
.tq-node{cursor:pointer}
.tq-node:hover circle:first-child{filter:brightness(0.93)}
.tq-tasks{flex:1 1 340px;min-width:240px;display:flex;flex-direction:column;gap:8px}
.tq-row{border:2px solid #dee2e6;border-left:6px solid var(--c);border-radius:8px;padding:6px 8px;cursor:pointer;background:#fff}
.tq-row.tq-active{border-color:var(--c);box-shadow:0 0 0 2px rgba(0,0,0,0.04)}
.tq-row.tq-ok{border-color:#2f9e44;background:#ebfbee}
.tq-row.tq-bad{border-color:#e03131;background:#fff5f5}
.tq-head{display:flex;align-items:center;gap:6px;color:#212529;flex-wrap:wrap}
.tq-radio{width:12px;height:12px;border-radius:50%;border:2px solid var(--c);flex:none}
.tq-active .tq-radio{background:var(--c)}
.tq-tag{font-size:11px;font-weight:700;color:#495057;background:#e9ecef;border-radius:10px;padding:1px 7px}
.tq-btns{margin-left:auto;display:flex;gap:4px}
.tq-card .tq-btns button,.tq-card .tq-gal button,.tq-card .tq-more button{padding:1px 8px;font-size:13px}
.tq-seq{margin-top:6px;min-height:28px;display:flex;flex-wrap:wrap;align-items:center;gap:3px}
.tq-chip{display:inline-block;min-width:24px;text-align:center;color:#fff;border-radius:12px;padding:2px 7px;font-weight:700}
.tq-arrow{color:#868e96}
.tq-ph{color:#868e96;font-style:italic}
.tq-res{margin-top:6px;display:flex;align-items:center;gap:8px;color:#212529}
.tq-face{font-size:28px;line-height:1;flex:none}
.tq-sol{margin-top:4px;color:#495057;font-size:13px}
.tq-more{margin-top:6px;color:#212529;font-size:13px}
.tq-gal{margin-top:6px;padding:6px;border:1px dashed var(--c);border-radius:6px;background:#fff;
  display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.tq-gal .tq-seq{flex-basis:100%;margin-top:2px}
.tq-gal-count{font-variant-numeric:tabular-nums;color:#212529;margin:0 6px;font-size:13px}
.tq-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}
.tq-note{color:#e8590c;min-height:18px;font-size:13px}
.tq-summary{margin-top:6px;color:#212529;min-height:20px}
</style>
"""

_HTML = r"""
<div class="tq-card" id="__ROOT__">
  <div class="tq-top">
    <h4>🎲 BFS i DFS na losowym grafie</h4>
    <div class="tq-mode">
      <button data-mode="exam">📝 Egzamin</button><button data-mode="playground">🔓 Plac zabaw</button>
    </div>
  </div>
  <div class="tq-controls">
    <label>Węzłów: <select class="tq-n">
      <option>5</option><option>6</option><option>7</option><option>8</option><option>9</option><option>10</option>
    </select></label>
    <label>Krawędzi: <select class="tq-dens">
      <option value="sparse">mało</option><option value="medium">średnio</option><option value="dense">dużo</option>
    </select></label>
    <label>Numer grafu: <input class="tq-seed" type="number" min="0" step="1"></label>
    <button class="tq-new">🎲 Nowy graf</button>
  </div>
  <div class="tq-rule"></div>
  <div class="tq-info"></div>
  <div class="tq-body">
    <div class="tq-svgbox"><svg class="tq-svg" viewBox="0 0 560 360" xmlns="http://www.w3.org/2000/svg"></svg></div>
    <div class="tq-tasks"></div>
  </div>
  <div class="tq-note"></div>
  <div class="tq-actions">
    <button class="tq-check">✔ Sprawdź</button>
    <button class="tq-show">👁 Pokaż rozwiązanie</button>
  </div>
  <div class="tq-summary"></div>
</div>
"""


def build_html(n_nodes: int = 8, density: str = "medium", seed: int | None = None, mode: str = "exam") -> str:
    """Zwraca samodzielny fragment HTML (CSS + SVG + JS) z cwiczeniem."""
    if not 5 <= n_nodes <= 10:
        raise ValueError("n_nodes musi byc w zakresie 5-10")
    if density not in DENSITIES:
        raise ValueError(f"density musi byc jednym z {DENSITIES}")
    if mode not in MODES:
        raise ValueError(f"mode musi byc jednym z {MODES}")
    if seed is None:
        seed = random.randrange(100000)
    root = "tq-" + uuid.uuid4().hex[:10]
    cfg = json.dumps({"n": n_nodes, "density": density, "seed": int(seed), "mode": mode})
    ui = _UI_JS.replace("__ROOT__", root).replace("__CFG__", cfg)
    return (_CSS + _HTML.replace("__ROOT__", root)
            + "<script>(function(){\n" + _LOGIC_JS + "\n" + ui + "\n})();</script>")


def traversal_exercise(n_nodes: int = 8, density: str = "medium", seed: int | None = None, mode: str = "exam"):
    """Wyswietla cwiczenie.
    density: 'sparse' | 'medium' | 'dense';  mode: 'exam' (sasiedzi rosnaco) | 'playground' (dowolna kolejnosc).
    Podaj seed (np. 2026), zeby cala grupa dostala ten sam graf."""
    from IPython.display import HTML, display
    display(HTML(build_html(n_nodes, density, seed, mode)))
