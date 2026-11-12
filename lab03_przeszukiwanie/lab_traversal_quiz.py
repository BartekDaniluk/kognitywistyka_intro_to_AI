"""
lab_traversal_quiz.py - cwiczenie "BFS i DFS na kartce", ale w przegladarce.

Student dostaje losowy graf (5-10 wezlow, numery 0-9), losowy wezel START i CEL, i:
  1. klikajac wezly zapisuje kolejnosc odwiedzania BFS,
  2. tak samo dla DFS,
  3. klikajac wezly zapisuje najkrotsza sciezke START -> CEL,
a potem klika "Sprawdz" - odpowiedzi robia sie zielone 😀 albo czerwone 😢.

Konwencja (jedna poprawna odpowiedz): sasiadow rozpatrujemy w kolejnosci rosnacych
numerow; przechodzimy caly graf (nie zatrzymujemy sie na celu); DFS = wersja rekurencyjna
(daje te sama kolejnosc co iteracyjny DFS z lab_search, ktory odklada sasiadow na stos
w odwrotnej kolejnosci). Najkrotsza sciezka: akceptowana jest kazda najkrotsza.

Cale cwiczenie (losowanie grafu, klikanie, sprawdzanie) dziala w JavaScripcie w komorce
wyjsciowej, wiec dziala tak samo w Google Colab, JupyterLab i VS Code - bez ipywidgets.

Uzycie w notebooku:
    import lab_traversal_quiz as quiz
    quiz.traversal_exercise()                      # losowy graf
    quiz.traversal_exercise(n_nodes=10, seed=2026)  # ten sam graf dla calej grupy
"""
from __future__ import annotations

import json
import random
import uuid

DENSITIES = ("sparse", "medium", "dense")

# --------------------------------------------------------------------------
# Logika (czysty JS, bez DOM) - testowalna w node: module.exports
# --------------------------------------------------------------------------
_LOGIC_JS = r"""
var TQ = (function () {
  var W = 560, H = 360, COLS = 4, ROWS = 3, R = 19;
  var DENS = { sparse: 0.2, medium: 0.45, dense: 0.8 };

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
  // BFS: kolejka FIFO, sasiedzi rosnaco
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
  // DFS rekurencyjny, sasiedzi rosnaco
  function dfsOrder(adj, s) {
    var seen = {}, order = [];
    (function go(u) {
      seen[u] = true; order.push(u);
      adj[u].forEach(function (v) { if (!seen[v]) go(v); });
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
    adj.forEach(function (l) { l.sort(function (x, y) { return x - y; }); });
    var start = Math.floor(r() * n), dist = bfsDist(adj, start);
    var far = range(n).filter(function (x) { return dist[x] >= 2; });
    if (!far.length) far = range(n).filter(function (x) { return x !== start; });
    var target = far[Math.floor(r() * far.length)];
    return { n: n, pos: pos, edges: edges, adj: adj, start: start, target: target, seed: seed,
             dist: dist, bfs: bfsOrder(adj, start), dfs: dfsOrder(adj, start),
             path: shortestPath(adj, start, target) };
  }

  function nSteps(k) {
    var w = k === 1 ? "krok" : (k % 10 >= 2 && k % 10 <= 4 && (k % 100 < 12 || k % 100 > 14)) ? "kroki" : "kroków";
    return k + " " + w;
  }
  function checkOrder(expected, given) {
    if (!given.length) return { ok: false, msg: "Lista jest pusta – kliknij węzły na grafie." };
    for (var i = 0; i < given.length; i++) {
      if (i >= expected.length || given[i] !== expected[i])
        return { ok: false, msg: "Pierwszy błąd na pozycji " + (i + 1) + " (węzeł " + given[i] + ")." };
    }
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
           bfsDist: bfsDist, shortestPath: shortestPath, checkOrder: checkOrder, checkPath: checkPath,
           nSteps: nSteps };
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
    { id: "dfs", color: "#e8590c", tint: "#ffe8cc", title: "2. Kolejność odwiedzania – DFS (w głąb)" },
    { id: "path", color: "#9c36b5", tint: "#f3d9fa", title: "3. Najkrótsza ścieżka START → CEL" }
  ];
  var S = { g: null, active: "bfs", ans: { bfs: [], dfs: [], path: [] }, res: {}, reveal: false, note: "" };
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

  function newGraph(seed) {
    var n = parseInt($(".tq-n").value, 10), dens = $(".tq-dens").value;
    S.g = TQ.generate(n, dens, seed);
    $(".tq-seed").value = seed;
    S.ans = { bfs: [], dfs: [], path: [] }; S.res = {}; S.reveal = false; S.note = "";
    render();
  }
  function randomSeed() { return Math.floor(Math.random() * 100000); }

  function clickNode(v) {
    var list = S.ans[S.active];
    if (list.indexOf(v) >= 0) { S.note = "Węzeł " + v + " już jest na tej liście."; render(); return; }
    list.push(v); S.note = ""; delete S.res[S.active]; render();
  }

  function drawGraph() {
    var g = S.g, svg = $(".tq-svg"), t = task(S.active), list = S.ans[S.active];
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var pathEdges = {};
    if (S.active === "path")
      for (var i = 0; i + 1 < list.length; i++) pathEdges[Math.min(list[i], list[i + 1]) + "-" + Math.max(list[i], list[i + 1])] = true;
    g.edges.forEach(function (e) {
      var on = pathEdges[e[0] + "-" + e[1]], a = g.pos[e[0]], b = g.pos[e[1]];
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

  function expectedText(id) {
    var g = S.g, seq = id === "bfs" ? g.bfs : id === "dfs" ? g.dfs : g.path;
    return seq.join(" → ") + (id === "path" ? "  (" + TQ.nSteps(seq.length - 1) + ")" : "");
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
      var chips = seq.length ? seq.map(function (v) { return "<span class='tq-chip' style='background:" + t.color + "'>" + v + "</span>"; }).join("<span class='tq-arrow'>→</span>")
                             : "<span class='tq-ph'>" + (active ? "kliknij węzły na grafie…" : "kliknij tutaj, żeby wybrać to zadanie") + "</span>";
      row.innerHTML =
        "<div class='tq-head'><span class='tq-radio'></span><b>" + t.title + "</b>" +
        "<span class='tq-btns'><button class='tq-undo' title='Cofnij ostatni'>⌫</button><button class='tq-clear' title='Wyczyść'>✕</button></span></div>" +
        "<div class='tq-seq'>" + chips + "</div>" +
        (res ? "<div class='tq-res'><span class='tq-face'>" + (res.ok ? "😀" : "😢") + "</span>" + res.msg + "</div>" : "") +
        (S.reveal ? "<div class='tq-sol'>Rozwiązanie: <b>" + expectedText(t.id) + "</b>" + (t.id === "path" ? " – lub inna ścieżka tej samej długości" : "") + "</div>" : "");
      row.addEventListener("click", function () { if (S.active !== t.id) { S.active = t.id; S.note = ""; render(); } });
      row.querySelector(".tq-undo").addEventListener("click", function (ev) { ev.stopPropagation(); S.active = t.id; seq.pop(); delete S.res[t.id]; render(); });
      row.querySelector(".tq-clear").addEventListener("click", function (ev) { ev.stopPropagation(); S.active = t.id; seq.length = 0; delete S.res[t.id]; render(); });
      box.appendChild(row);
    });
  }

  function render() {
    var g = S.g;
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
    S.res = { bfs: TQ.checkOrder(g.bfs, S.ans.bfs), dfs: TQ.checkOrder(g.dfs, S.ans.dfs), path: TQ.checkPath(g, S.ans.path) };
    S.note = ""; render();
  }

  $(".tq-new").addEventListener("click", function () { newGraph(randomSeed()); });
  $(".tq-seed").addEventListener("change", function () { newGraph(parseInt(this.value, 10) || 0); });
  $(".tq-n").addEventListener("change", function () { newGraph(S.g.seed); });
  $(".tq-dens").addEventListener("change", function () { newGraph(S.g.seed); });
  $(".tq-check").addEventListener("click", check);
  $(".tq-show").addEventListener("click", function () { S.reveal = !S.reveal; this.textContent = S.reveal ? "🙈 Ukryj rozwiązanie" : "👁 Pokaż rozwiązanie"; render(); });
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
.tq-svgbox{flex:1 1 360px;min-width:280px;border:1px solid #e9ecef;border-radius:8px;background:#fff}
.tq-svg{width:100%;height:auto;display:block}
.tq-node{cursor:pointer}
.tq-node:hover circle:first-child{filter:brightness(0.93)}
.tq-tasks{flex:1 1 320px;min-width:260px;display:flex;flex-direction:column;gap:8px}
.tq-row{border:2px solid #dee2e6;border-left:6px solid var(--c);border-radius:8px;padding:6px 8px;cursor:pointer;background:#fff}
.tq-row.tq-active{border-color:var(--c);box-shadow:0 0 0 2px rgba(0,0,0,0.04)}
.tq-row.tq-ok{border-color:#2f9e44;background:#ebfbee}
.tq-row.tq-bad{border-color:#e03131;background:#fff5f5}
.tq-head{display:flex;align-items:center;gap:6px;color:#212529}
.tq-radio{width:12px;height:12px;border-radius:50%;border:2px solid var(--c);flex:none}
.tq-active .tq-radio{background:var(--c)}
.tq-btns{margin-left:auto;display:flex;gap:4px}
.tq-card .tq-btns button{padding:1px 8px;font-size:13px}
.tq-seq{margin-top:6px;min-height:28px;display:flex;flex-wrap:wrap;align-items:center;gap:3px}
.tq-chip{display:inline-block;min-width:24px;text-align:center;color:#fff;border-radius:12px;padding:2px 7px;font-weight:700}
.tq-arrow{color:#868e96}
.tq-ph{color:#868e96;font-style:italic}
.tq-res{margin-top:6px;display:flex;align-items:center;gap:8px;color:#212529}
.tq-face{font-size:28px;line-height:1}
.tq-sol{margin-top:4px;color:#495057;font-size:13px}
.tq-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}
.tq-note{color:#e8590c;min-height:18px;font-size:13px}
.tq-summary{margin-top:6px;color:#212529;min-height:20px}
</style>
"""

_HTML = r"""
<div class="tq-card" id="__ROOT__">
  <h4>🎲 BFS i DFS na losowym grafie</h4>
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
  <div class="tq-rule">📏 <b>Zasady:</b> zaczynamy w węźle <b>START</b>; sąsiadów rozpatrujemy zawsze w kolejności
    <b>rosnących numerów</b>; odwiedzamy <b>cały graf</b> (nie zatrzymujemy się na celu). DFS – wersja rekurencyjna
    (tak samo jak stos LIFO w symulatorze). Graf jest nieskierowany.</div>
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


def build_html(n_nodes: int = 8, density: str = "medium", seed: int | None = None) -> str:
    """Zwraca samodzielny fragment HTML (CSS + SVG + JS) z cwiczeniem."""
    if not 5 <= n_nodes <= 10:
        raise ValueError("n_nodes musi byc w zakresie 5-10")
    if density not in DENSITIES:
        raise ValueError(f"density musi byc jednym z {DENSITIES}")
    if seed is None:
        seed = random.randrange(100000)
    root = "tq-" + uuid.uuid4().hex[:10]
    cfg = json.dumps({"n": n_nodes, "density": density, "seed": int(seed)})
    ui = _UI_JS.replace("__ROOT__", root).replace("__CFG__", cfg)
    return (_CSS + _HTML.replace("__ROOT__", root)
            + "<script>(function(){\n" + _LOGIC_JS + "\n" + ui + "\n})();</script>")


def traversal_exercise(n_nodes: int = 8, density: str = "medium", seed: int | None = None):
    """Wyswietla cwiczenie. density: 'sparse' | 'medium' | 'dense'.
    Podaj seed (np. 2026), zeby cala grupa dostala ten sam graf."""
    from IPython.display import HTML, display
    display(HTML(build_html(n_nodes, density, seed)))
