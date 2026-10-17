"""H7 · ARC-AGI-3 — explorador de grafo de estados (sin ML, sin LLM).

Idea central
------------
Cada frame (64×64, 16 colores) es un nodo; cada acción es una arista.
El agente prueba primero las acciones que aún no ha probado en el estado
actual. Cuando ya no quedan, viaja por BFS al estado conocido más cercano
que todavía tenga acciones sin probar. RESET funciona como una arista
virtual que regresa al inicio del nivel.

Tres filtros evitan que el grafo explote:
  1. Máscara de HUD: filas/columnas de borde y píxeles que cambian en casi
     todas las transiciones (contadores de pasos, barras de energía) se
     ignoran al calcular el hash del estado.
  2. Acciones muertas: una acción que nunca cambió nada después de varios
     intentos baja de prioridad (solo se usa si no queda otra cosa).
  3. Clicks (ACTION6) solo sobre objetos: componentes conexas, priorizando
     colores raros y formas pequeñas, y descartando colores que nunca
     respondieron a un click.

Contrato del framework ARC-AGI-3-Agents:
  - clase llamada `MyAgent`, subclase de `agents.agent.Agent`
  - `is_done(frames, latest_frame) -> bool`
  - `choose_action(frames, latest_frame) -> GameAction`
"""
from __future__ import annotations

import hashlib
import random
import zlib
from collections import deque
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

# Una acción concreta: (id, x, y). Para acciones simples x = y = -1.
Key = tuple
RESET_KEY: Key = (0, -1, -1)
EXPLORE_IDS = (1, 2, 3, 4, 5, 6)  # ACTION7 (undo) no se usa para explorar
EDGE_BAND = 4                      # el HUD por filas/columnas solo se busca en los 4 px del borde


class MyAgent(Agent):
    MAX_ACTIONS = 2000      # tope de acciones por juego (lo aplica el framework)
    MAX_CLICKS = 24         # candidatos ACTION6 por estado
    DEAD_AFTER = 4          # intentos sin efecto para declarar muerta una acción/color
    DEATHS_TO_FORBID = 2    # una arista se prohíbe tras 2 muertes (1 puede ser el reloj)
    HUD_ON, HUD_OFF = 0.90, 0.70   # histéresis de la máscara por píxel
    HUD_MIN_SAMPLES = 12
    HUD_WINDOW = 16         # ventana deslizante para filas/columnas de borde

    # ───────────────────────────── estado ─────────────────────────────
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rng = random.Random(zlib.crc32(self.game_id.encode()))
        # estadística global del juego (sobrevive entre niveles)
        self.n_trans = 0
        self.pix_freq = np.zeros((64, 64), np.int32)
        self.recent_rows: deque = deque(maxlen=self.HUD_WINDOW)
        self.recent_cols: deque = deque(maxlen=self.HUD_WINDOW)
        self.m_pix = np.zeros((64, 64), bool)
        self.m_row = np.zeros(64, bool)
        self.m_col = np.zeros(64, bool)
        self.mask = np.zeros((64, 64), bool)
        self.act_tries = np.zeros(8, np.int32)
        self.act_hits = np.zeros(8, np.int32)
        self.color_tries = np.zeros(16, np.int32)
        self.color_hits = np.zeros(16, np.int32)
        # memoria de corto plazo
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_key: Optional[Key] = None
        self.steps_since_reset = 0
        self.level = 0
        self._new_level(0, None)

    def _new_level(self, level: int, grid: Optional[np.ndarray]) -> None:
        self.level = level
        self.log: list[tuple[np.ndarray, Key, Optional[np.ndarray]]] = []
        self.root_grid = grid
        self.avail: dict[bytes, list[int]] = {}
        self.avail_by_grid: dict[bytes, tuple[np.ndarray, list[int]]] = {}
        self.steps_since_reset = 0
        self._rebuild()

    # ─────────────────────────── percepción ───────────────────────────
    @staticmethod
    def _grid(fd: FrameData) -> Optional[np.ndarray]:
        if not fd.frame:
            return None
        g = np.asarray(fd.frame[-1], dtype=np.uint8)  # último sub-frame = estado final
        if g.shape != (64, 64):
            out = np.zeros((64, 64), np.uint8)
            h, w = min(64, g.shape[0]), min(64, g.shape[1])
            out[:h, :w] = g[:h, :w]
            g = out
        return g

    def _h(self, g: np.ndarray) -> bytes:
        masked = np.where(self.mask, 255, g).astype(np.uint8)
        return hashlib.blake2b(masked.tobytes(), digest_size=8).digest()

    def _update_mask(self) -> bool:
        """Recalcula la máscara de HUD. Devuelve True si cambió."""
        if self.n_trans < self.HUD_MIN_SAMPLES:
            return False
        # (a) píxeles sueltos que cambian casi siempre (dígitos, parpadeos)
        f = self.pix_freq / float(self.n_trans)
        p = (f >= self.HUD_ON) | (self.m_pix & (f >= self.HUD_OFF))
        if p.sum() > 1024:
            p[:] = False
        # (b) filas/columnas de borde que cambiaron en casi toda la ventana
        #     reciente: barras que se consumen un píxel por acción. Pegajoso.
        band = np.zeros(64, bool)
        band[:EDGE_BAND] = band[-EDGE_BAND:] = True
        r, c = self.m_row.copy(), self.m_col.copy()
        if len(self.recent_rows) == self.HUD_WINDOW:
            r |= (np.mean(self.recent_rows, axis=0) >= self.HUD_ON) & band
            c |= (np.mean(self.recent_cols, axis=0) >= self.HUD_ON) & band
        if r.sum() > 2 * EDGE_BAND:   # cambia casi todo: es scroll, no HUD
            r[:] = False
        if c.sum() > 2 * EDGE_BAND:
            c[:] = False
        new = p | r[:, None] | c[None, :]
        changed = not np.array_equal(new, self.mask)
        self.m_row, self.m_col, self.m_pix, self.mask = r, c, p, new
        return changed

    # ───────────────────────────── grafo ──────────────────────────────
    def _rebuild(self) -> None:
        """Reconstruye el grafo del nivel desde el log (necesario si cambia la máscara)."""
        old_avail = self.avail_by_grid
        self.edges: dict[bytes, dict[Key, Optional[bytes]]] = {}
        self.rep: dict[bytes, np.ndarray] = {}
        self.cands_cache: dict[bytes, list[Key]] = {}
        self.deaths: dict[tuple[bytes, Key], int] = {}
        for a, k, b in self.log:
            self._add(a, k, b)
        self.root = None
        if self.root_grid is not None:
            self.root = self._h(self.root_grid)
            self.rep.setdefault(self.root, self.root_grid)
            self.edges.setdefault(self.root, {})
        # re-mapear acciones disponibles a los hashes nuevos
        self.avail = {self._h(g): v for g, v in old_avail.values()}

    def _add(self, a: np.ndarray, k: Key, b: Optional[np.ndarray]) -> None:
        ha = self._h(a)
        self.rep.setdefault(ha, a)
        out = self.edges.setdefault(ha, {})
        if b is None:  # muerte: solo se prohíbe si se repite (la 1ª puede ser el reloj)
            n = self.deaths[(ha, k)] = self.deaths.get((ha, k), 0) + 1
            if n >= self.DEATHS_TO_FORBID:
                out[k] = None
            return
        hb = self._h(b)
        self.rep.setdefault(hb, b)
        self.edges.setdefault(hb, {})
        out[k] = hb

    def _record(self, a: np.ndarray, k: Key, b: Optional[np.ndarray]) -> None:
        changed = b is None or self._h(a) != self._h(b)
        self.act_tries[k[0]] += 1
        self.act_hits[k[0]] += int(changed)
        if k[0] == 6:
            c = int(a[k[2], k[1]])
            self.color_tries[c] += 1
            self.color_hits[c] += int(changed)
        if b is not None:
            diff = a != b
            self.n_trans += 1
            self.pix_freq += diff
            self.recent_rows.append(diff.any(axis=1))
            self.recent_cols.append(diff.any(axis=0))
        self.log.append((a, k, b))
        self._add(a, k, b)

    # ─────────────────────── candidatos de acción ─────────────────────
    def _dead(self, k: Key, g: np.ndarray) -> bool:
        if k[0] == 6:
            c = int(g[k[2], k[1]])
            return self.color_tries[c] >= self.DEAD_AFTER and self.color_hits[c] == 0
        return self.act_tries[k[0]] >= self.DEAD_AFTER and self.act_hits[k[0]] == 0

    def _cands(self, h: bytes) -> list[Key]:
        if h in self.cands_cache:
            return self.cands_cache[h]
        avail = self.avail.get(h) or list(EXPLORE_IDS)
        out: list[Key] = [(i, -1, -1) for i in avail if i in EXPLORE_IDS and i != 6]
        if 6 in avail:
            out += self._click_cands(self.rep[h])
        self.cands_cache[h] = out
        return out

    def _click_cands(self, g: np.ndarray) -> list[Key]:
        """Un click por objeto (componente 4-conexa de un mismo color)."""
        valid = ~self.mask
        counts = np.bincount(g[valid], minlength=16)
        bg = int(counts.argmax())
        grid = g.tolist()
        seen = (self.mask | (g == bg)).tolist()
        comps = []
        for y in range(64):
            row_seen = seen[y]
            for x in range(64):
                if row_seen[x]:
                    continue
                col = grid[y][x]
                stack, pts = [(y, x)], []
                seen[y][x] = True
                while stack:
                    cy, cx = stack.pop()
                    pts.append((cy, cx))
                    for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                        if 0 <= ny < 64 and 0 <= nx < 64 and not seen[ny][nx] and grid[ny][nx] == col:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                my = sum(p[0] for p in pts) / len(pts)
                mx = sum(p[1] for p in pts) / len(pts)
                cy, cx = min(pts, key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2)
                comps.append((int(counts[col]), len(pts), cx, cy))
        comps.sort()  # colores raros primero, luego objetos pequeños
        return [(6, cx, cy) for _, _, cx, cy in comps[: self.MAX_CLICKS]]

    def _untried(self, h: bytes, tier: int) -> list[Key]:
        tried = self.edges.get(h, {})
        g = self.rep[h]
        out = [k for k in self._cands(h) if k not in tried]
        if tier == 1:
            out = [k for k in out if not self._dead(k, g)]
        return out

    # ──────────────────────────── planeación ──────────────────────────
    def _plan(self, cur: bytes, tier: int) -> Optional[tuple[Key, str]]:
        here = self._untried(cur, tier)
        if here:
            return here[0], "explore"
        prev: dict[bytes, Optional[tuple[bytes, Key]]] = {cur: None}
        q = deque([cur])
        while q:
            u = q.popleft()
            nbrs = [(k, v) for k, v in self.edges.get(u, {}).items() if v is not None and v != u]
            can_reset = self.root is not None and u != self.root and (u != cur or self.steps_since_reset > 0)
            if can_reset:
                nbrs.append((RESET_KEY, self.root))
            for k, v in nbrs:
                if v in prev:
                    continue
                prev[v] = (u, k)
                if self._untried(v, tier):
                    while prev[v][0] != cur:   # retroceder hasta el primer paso
                        v = prev[v][0]
                    return prev[v][1], "travel"
                q.append(v)
        return None

    # ───────────────────────────── interfaz ───────────────────────────
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        state = latest_frame.state
        grid = self._grid(latest_frame)

        # Inicio o muerte → RESET (registrando la arista mortal)
        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or grid is None:
            if state is GameState.GAME_OVER and self.prev_key not in (None, RESET_KEY):
                self._record(self.prev_grid, self.prev_key, None)
            return self._emit(RESET_KEY, "reset", grid)

        if latest_frame.levels_completed != self.level:
            self._new_level(latest_frame.levels_completed, grid)
        elif self.prev_key == RESET_KEY or self.root_grid is None:
            self.root_grid = grid
            self.root = self._h(grid)
            self.steps_since_reset = 0
        elif self.prev_key is not None:
            self._record(self.prev_grid, self.prev_key, grid)

        if self._update_mask():
            self._rebuild()

        cur = self._h(grid)
        self.rep.setdefault(cur, grid)
        self.edges.setdefault(cur, {})
        if cur not in self.avail:
            av = list(latest_frame.available_actions) or list(EXPLORE_IDS)
            self.avail[cur] = av
            self.avail_by_grid[cur] = (grid, av)

        plan = self._plan(cur, 1) or self._plan(cur, 2)
        if plan is None:  # todo explorado bajo esta abstracción: ruido controlado
            opts = [k for k in self._cands(cur)] or [(1, -1, -1)]
            return self._emit(self.rng.choice(opts), "random", grid)
        key, mode = plan
        return self._emit(key, mode, grid)

    def _emit(self, key: Key, mode: str, grid: Optional[np.ndarray]) -> GameAction:
        action = GameAction.from_id(key[0])
        if action.is_complex():
            action.set_data({"x": key[1], "y": key[2]})
        action.reasoning = {"mode": mode, "level": self.level, "nodes": len(self.edges)}
        self.prev_grid, self.prev_key = grid, key
        self.steps_since_reset = 0 if key == RESET_KEY else self.steps_since_reset + 1
        return action
