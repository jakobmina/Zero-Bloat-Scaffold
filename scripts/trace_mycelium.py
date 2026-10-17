"""Búsqueda tipo micelio en "superposición" + 3 logs de diagnóstico.

Corre el código REAL de `my_agent.py` sin arcengine ni el framework (se
sustituyen por stubs mínimos) sobre un juego mock: laberinto de 2 niveles con
barra de HUD, límite de pasos (muertes) y acciones inútiles (5 y 6).

Micelio: en vez de ir a la punta de frontera más cercana (BFS), todas las
puntas coexisten con una amplitud

    a(punta) = nutriente(punta) · exp(-distancia / λ)
    P(punta) = a² / Σ a²          (regla tipo Born, solo como analogía)

y cada acción real "colapsa" hacia la punta de mayor amplitud.
nutriente = acciones sin probar × (0.5 + respuesta), donde respuesta es la
fracción de acciones ya probadas en esa punta que cambiaron el estado.

Salidas (JSONL, una línea por paso) en h7_logs/<agente>/:
  1. graph.jsonl       nodos, aristas, muertes (+ volcado final del grafo)
  2. perception.jsonl  hash, máscara de HUD, candidatos, acciones sin probar
  3. actions.jsonl     GameAction emitidos con su reasoning

uso:  python trace_mycelium.py [--agent ruta/my_agent.py] [--steps 600]
"""
from __future__ import annotations

import argparse
import enum
import importlib.util
import json
import math
import random
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ───────────────────── stubs de arcengine y del framework ─────────────────────
class GameState(str, enum.Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class GameAction:
    """Mismo contrato que usa el agente: from_id, is_complex, set_data, reasoning."""

    def __init__(self, id: int, x: int = -1, y: int = -1) -> None:
        self.id, self.x, self.y, self.reasoning = id, x, y, {}

    @classmethod
    def from_id(cls, id: int) -> "GameAction":
        return cls(id)

    def is_complex(self) -> bool:
        return self.id == 6

    def set_data(self, d: dict) -> None:
        self.x, self.y = d["x"], d["y"]

    def to_json(self) -> dict:
        out = {"id": self.id, "name": "RESET" if self.id == 0 else f"ACTION{self.id}"}
        if self.is_complex():
            out.update(x=self.x, y=self.y)
        out["reasoning"] = self.reasoning
        return out


@dataclass
class FrameData:
    frame: list = field(default_factory=list)
    state: GameState = GameState.NOT_PLAYED
    levels_completed: int = 0
    win_levels: int = 0
    available_actions: list = field(default_factory=list)


class Agent:
    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        self.game_id, self.frames = game_id, [FrameData()]


def install_stubs() -> None:
    arc = types.ModuleType("arcengine")
    arc.GameState, arc.GameAction, arc.FrameData = GameState, GameAction, FrameData
    pkg, mod = types.ModuleType("agents"), types.ModuleType("agents.agent")
    mod.Agent, pkg.agent = Agent, mod
    sys.modules.update({"arcengine": arc, "agents": pkg, "agents.agent": mod})


# ─────────────────────────────── juego mock ───────────────────────────────────
MAZES = [  # '#' muro, '.' piso, P jugador, E salida
    ["########",
     "#P..#..#",
     "#.#.#.##",
     "#.#....#",
     "#.####.#",
     "#....#.#",
     "####..E#",
     "########"],
    ["############",
     "#P..#......#",
     "#.#.#.####.#",
     "#.#......#.#",
     "#.######.#.#",
     "#......#.#.#",
     "######.#.#.#",
     "#....#.#.#.#",
     "#.##.#.#.#.#",
     "#.#..#...#.#",
     "#...######E#",
     "############"],
]
LIMIT = 60  # pasos por vida; la barra del HUD se consume 1 px por acción


class MockGame:
    def __init__(self) -> None:
        self.level, self.state, self.left = 0, GameState.NOT_PLAYED, LIMIT
        self.pos = None

    def _start(self) -> None:
        rows = MAZES[self.level]
        self.walls = {(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == "#"}
        self.pos = next((x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == "P")
        self.exit = next((x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == "E")
        self.size, self.left, self.state = len(rows), LIMIT, GameState.NOT_FINISHED

    def render(self) -> np.ndarray:
        n = self.size
        s = 64 // n
        pad = (64 - s * n) // 2
        g = np.full((64, 64), 3, np.uint8)
        for y in range(n):
            for x in range(n):
                c = 5 if (x, y) in self.walls else 0
                c = 9 if (x, y) == self.exit else c
                c = 8 if (x, y) == self.pos else c
                g[pad + y * s: pad + (y + 1) * s, pad + x * s: pad + (x + 1) * s] = c
        g[0:2, :] = 0
        g[0:2, : self.left] = 11  # HUD
        return g

    def frame(self) -> FrameData:
        f = [self.render()] if self.state is not GameState.NOT_PLAYED else []
        return FrameData(f, self.state, self.level, len(MAZES), [1, 2, 3, 4, 5, 6])

    def step(self, a: GameAction) -> FrameData:
        if a.id == 0:
            self._start()
            return self.frame()
        if self.state is not GameState.NOT_FINISHED:
            return self.frame()
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(a.id, (0, 0))
        nxt = (self.pos[0] + dx, self.pos[1] + dy)
        if nxt not in self.walls:
            self.pos = nxt
        self.left -= 1
        if self.pos == self.exit:
            self.level += 1
            if self.level == len(MAZES):
                self.state = GameState.WIN
            else:
                self._start()
        elif self.left <= 0:
            self.state = GameState.GAME_OVER
        return self.frame()


# ──────────────────────────── agente micelio ──────────────────────────────────
def make_mycelium(Base):
    class MyceliumAgent(Base):
        HYPHA_LEN = 1.5  # λ: qué tanto pesa la distancia al crecer
        # Cómo colapsa la superposición:
        #   "max"    → siempre la punta de mayor amplitud (determinista)
        #   "sample" → muestrea P = a²/Σa² en CADA paso (puede oscilar entre puntas)
        #   "commit" → muestrea P una vez y sigue a esa punta hasta llegar o perderla
        COLLAPSE = "max"
        target = None

        def _bfs_all(self, cur):
            """Distancia y primer paso hacia cada nodo conocido (incluye RESET virtual)."""
            dist, first, q = {cur: 0}, {cur: None}, [cur]
            for u in q:
                nbrs = [(k, v) for k, v in self.edges.get(u, {}).items() if v is not None and v != u]
                if self.root is not None and u != self.root and (u != cur or self.steps_since_reset > 0):
                    nbrs.append(((0, -1, -1), self.root))
                for k, v in nbrs:
                    if v not in dist:
                        dist[v], first[v] = dist[u] + 1, first[u] or k
                        q.append(v)
            return dist, first

        def _nutrient(self, n, untried):
            tried = [v for v in self.edges.get(n, {}).values()]
            resp = (sum(v is not None and v != n for v in tried) / len(tried)) if tried else 0.5
            return len(untried) * (0.5 + resp)

        def _plan(self, cur, tier):
            dist, first = self._bfs_all(cur)
            amps = {}
            for n, d in dist.items():
                un = self._untried(n, tier)
                if un:
                    amps[n] = (self._nutrient(n, un) * math.exp(-d / self.HYPHA_LEN), un)
            if not amps:
                self.superposition = []
                return None
            z = sum(a * a for a, _ in amps.values())
            tips = list(amps)
            born = [amps[n][0] ** 2 / z for n in tips]
            if self.COLLAPSE == "max":
                best = max(tips, key=lambda n: (amps[n][0], -dist[n]))
            elif self.COLLAPSE == "commit" and self.target in amps:
                best = self.target  # evolución determinista hasta la siguiente "medición"
            else:
                best = self.rng.choices(tips, weights=born, k=1)[0]
            self.target = None if best == cur else best
            self.superposition = sorted(
                ({"tip": n.hex()[:8], "d": dist[n], "untried": len(un), "P": round(a * a / z, 3)}
                 for n, (a, un) in amps.items()), key=lambda t: -t["P"])[:5]
            if best == cur:
                return amps[cur][1][0], "explore"
            return first[best], "collapse"

        def _emit(self, key, mode, grid):
            action = super()._emit(key, mode, grid)
            if mode in ("explore", "collapse"):
                action.reasoning["superposition"] = getattr(self, "superposition", [])
            return action

    return MyceliumAgent


# ──────────────────────────────── corrida ─────────────────────────────────────
def run(AgentCls, name: str, steps: int, outdir: Path, seed: int | None = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    fg, fp, fa = (open(outdir / f, "w") for f in ("graph.jsonl", "perception.jsonl", "actions.jsonl"))
    env = MockGame()
    ag = AgentCls(card_id="mock", game_id="mock", agent_name=name, ROOT_URL="", record=False, arc_env=env)
    if seed is not None:
        ag.rng = random.Random(seed)
    fd, per_level, count = env.frame(), [], 0
    for t in range(steps):
        if fd.state is GameState.WIN:
            break
        act = ag.choose_action([], fd)
        # ── 2. percepción (estado sobre el que se decidió) ──
        if fd.frame:
            g = ag._grid(fd)
            h = ag._h(g)
            cands = ag._cands(h) if h in ag.rep else []
            fp.write(json.dumps({
                "t": t, "hash": h.hex(), "level": ag.level,
                "mask_px": int(ag.mask.sum()),
                "mask_rows": np.where(ag.m_row)[0].tolist(), "mask_cols": np.where(ag.m_col)[0].tolist(),
                "avail": ag.avail.get(h, []), "n_cands": len(cands),
                "cands": [list(c) for c in cands[:8]],
                "untried": len(ag._untried(h, 1)) if h in ag.rep else None,
            }) + "\n")
        # ── 3. traza de acciones ──
        fa.write(json.dumps({"t": t, **act.to_json()}) + "\n")
        prev_level = fd.levels_completed
        fd = env.step(act)
        count += 1
        if fd.levels_completed > prev_level:
            per_level.append(count)
            count = 0
        # ── 1. grafo ──
        fg.write(json.dumps({
            "t": t, "level": ag.level, "nodes": len(ag.edges),
            "edges": sum(len(v) for v in ag.edges.values()),
            "deaths": sum(ag.deaths.values()),
            "forbidden": sum(v is None for e in ag.edges.values() for v in e.values()),
            "state": fd.state.value,
        }) + "\n")
    fg.write(json.dumps({"final_graph": {
        "root": ag.root.hex() if ag.root else None,
        "edges": [[a.hex()[:8], list(k), b.hex()[:8] if b else None]
                  for a, e in ag.edges.items() for k, b in e.items()],
        "deaths": [[a.hex()[:8], list(k), n] for (a, k), n in ag.deaths.items()],
    }}) + "\n")
    for f in (fg, fp, fa):
        f.close()
    return {"agent": name, "levels": fd.levels_completed, "actions": t + 1 if fd.state is not GameState.WIN else t,
            "per_level": per_level, "state": fd.state.value}


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    default = next((p for p in (root / "agent" / "my_agent.py", root / "my_agent.py", here / "my_agent.py") if p.exists()), None)
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default=str(default) if default else None)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out", default="h7_logs")
    p.add_argument("--collapse", choices=["max", "sample", "commit"], default="max")
    p.add_argument("--seed", type=int, default=None, help="semilla del rng (para sample/commit)")
    a = p.parse_args()
    if not a.agent:
        raise SystemExit("No encontré my_agent.py; pásalo con --agent")

    install_stubs()
    spec = importlib.util.spec_from_file_location("my_agent_under_test", a.agent)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Base = mod.MyAgent

    out = Path(a.out)
    Myc = make_mycelium(Base)
    Myc.COLLAPSE = a.collapse
    rows = [run(Base, "bfs", a.steps, out / "bfs"),
            run(Myc, f"myc-{a.collapse}", a.steps, out / "mycelium", a.seed)]
    print(f"\n{'agente':10} {'niveles':>8} {'acciones':>9}  por nivel       estado")
    for r in rows:
        print(f"{r['agent']:10} {r['levels']:>5}/{len(MAZES)}  {r['actions']:>9}  {str(r['per_level']):14} {r['state']}")
    print(f"\nlogs en {out.resolve()}/{{bfs,mycelium}}/{{graph,perception,actions}}.jsonl")


if __name__ == "__main__":
    main()
