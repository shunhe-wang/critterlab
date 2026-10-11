"""Normative world rules, shared with the browser through cross-language fixtures.

0=grass, 1=rock, 2=hazard. Four actions: north, east, south, west.
The policy sees goal displacement and the 3x3 neighborhood, never a solver path.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np

SIZE = 7
LIMIT = 48
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class RNG:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296

    def integer(self, n: int) -> int:
        return int(self.next() * n)


@dataclass
class Garden:
    cells: list[int]
    start: int
    goal: int
    seed: int


def route(cells: list[int], start: int, goal: int) -> list[int] | None:
    """BFS expert, used only for demonstrations and map solvability."""
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        pos, actions = queue.popleft()
        if pos == goal:
            return actions
        x, y = pos % SIZE, pos // SIZE
        for action, (dx, dy) in enumerate(DELTAS):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < SIZE and 0 <= ny < SIZE):
                continue
            nxt = ny * SIZE + nx
            if cells[nxt] == 0 and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, actions + [action]))
    return None


def generate(seed: int, shifted: bool = False) -> Garden:
    rng = RNG(seed)
    for _ in range(100):
        cells = []
        for y in range(SIZE):
            for x in range(SIZE):
                r = rng.next()
                cells.append(1 if x in (0, SIZE - 1) or y in (0, SIZE - 1)
                             else 1 if r < (0.16 if shifted else 0.08)
                             else 2 if r < (0.26 if shifted else 0.14) else 0)
        start = (1 + rng.integer(5)) + (1 + rng.integer(5)) * SIZE
        goal = (1 + rng.integer(5)) + (1 + rng.integer(5)) * SIZE
        cells[start] = cells[goal] = 0
        path = route(cells, start, goal)
        if path is not None and len(path) >= 4:
            return Garden(cells, start, goal, seed)
    raise ValueError(f"Failed to generate solvable garden for seed {seed}")


def observe(cells: list[int], pos: int, goal: int) -> np.ndarray:
    x, y = pos % SIZE, pos // SIZE
    output = [(goal % SIZE - x) / SIZE, (goal // SIZE - y) / SIZE]
    for tile in (1, 2):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                value = cells[ny * SIZE + nx] if 0 <= nx < SIZE and 0 <= ny < SIZE else 1
                output.append(float(value == tile))
    return np.array(output, dtype=np.float32)


class World:
    def __init__(self, garden: Garden):
        self.garden = garden
        self.pos = garden.start
        self.steps = 0
        self.done = False
        self.outcome = "running"

    def obs(self) -> np.ndarray:
        return observe(self.garden.cells, self.pos, self.garden.goal)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if self.done:
            raise ValueError("Episode is already over")
        if action not in range(4):
            raise ValueError("Action must be 0..3")
        x, y = self.pos % SIZE, self.pos // SIZE
        dx, dy = DELTAS[action]
        nx, ny = x + dx, y + dy
        nxt = ny * SIZE + nx
        gx, gy = self.garden.goal % SIZE, self.garden.goal // SIZE
        before = abs(gx - x) + abs(gy - y)
        self.steps += 1
        reward = -0.02
        if not (0 <= nx < SIZE and 0 <= ny < SIZE) or self.garden.cells[nxt] == 1:
            reward = -0.08
        else:
            self.pos = nxt
            reward += 0.1 * (before - abs(gx - nx) - abs(gy - ny))
            if self.garden.cells[nxt] == 2:
                reward = -1.0
                self.done, self.outcome = True, "hazard"
            elif nxt == self.garden.goal:
                reward = 1.0
                self.done, self.outcome = True, "success"
        if self.steps >= LIMIT and not self.done:
            self.done, self.outcome = True, "timeout"
        return self.obs(), reward, self.done
