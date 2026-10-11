"""Vanishing-cue T garden; normative rules in docs/memory-protocol.md."""
from dataclasses import dataclass
import numpy as np
from .world import DELTAS, observe

HORIZON = 96


@dataclass(frozen=True)
class Task:
    column: int
    row: int
    delay: int
    cue: int  # 0 blue/left, 1 gold/right

    def __post_init__(self):
        if self.column not in (2, 3, 4) or self.row not in (1, 2, 3) or self.delay not in range(1, 33) or self.cue not in (0, 1):
            raise ValueError("Invalid memory task")


class MemoryWorld:
    def __init__(self, task: Task):
        self.task = task
        self.cells = [1] * 49
        for y in range(task.row, 6):
            self.cells[y * 7 + task.column] = 0
        for x in range(1, 6):
            self.cells[task.row * 7 + x] = 0
        self.junction = task.row * 7 + task.column
        self.start = 35 + task.column
        self.pos = self.start
        self.steps = 0
        self.done = False
        self.outcome = "running"

    def obs(self):
        return np.concatenate([observe(self.cells, self.pos, self.junction),
                               np.array([float(self.steps == 0 and self.task.cue == cue) for cue in (0, 1)], dtype=np.float32)])

    def expert(self):
        if self.steps < self.task.delay or self.pos // 7 > self.task.row:
            return 0
        return 3 if self.task.cue == 0 else 1

    def step(self, action):
        if self.done:
            raise ValueError("Episode is already over")
        if action not in range(4):
            raise ValueError("Action must be 0..3")
        if self.steps >= self.task.delay:
            dx, dy = DELTAS[action]
            nx, ny = self.pos % 7 + dx, self.pos // 7 + dy
            if 0 <= nx < 7 and 0 <= ny < 7 and self.cells[ny * 7 + nx] == 0:
                self.pos = ny * 7 + nx
            if self.pos in (self.task.row * 7 + 1, self.task.row * 7 + 5):
                chosen = int(self.pos % 7 == 5)
                self.done = True
                self.outcome = "success" if chosen == self.task.cue else "wrong-berry"
        self.steps += 1
        if self.steps >= HORIZON and not self.done:
            self.done, self.outcome = True, "timeout"
        return self.obs()


def tasks(rows, delays):
    return [Task(column, row, delay, cue) for column in (2, 3, 4)
            for row in rows for delay in delays for cue in (0, 1)]


TRAIN = tasks((2, 3), (2, 4, 8))
DEV = tasks((2, 3), (6,))
TEST = tasks((1,), (2, 4, 8))
LONG = tasks((1,), (16, 32))
