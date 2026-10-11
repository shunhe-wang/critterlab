# Exploratory failure traces

Selection rule: first failing frozen test world for the final seed-0 BC→PPO
actor trained on that family. This case selection is post-hoc; it is not
used to choose models or estimate effects. Full maps and actions are in the raw artifact.

## standard: timeout

World `02ba2808907354518f2e94a6bcafa9f483eb3cd2b2adb41a089fd006d100b52d` (generator seed 5000010).

```text
#######
#.....#
#..#S.#
#.#.!.#
#.#!!.#
#!G...#
#######
```

`#` rock, `!` hazard, `S` start, `G` goal. Coordinates are flat row-major indices.

Expert actions: `[1, 2, 2, 2, 3, 3, 3]`.

Policy actions: `[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]`.

Visited positions: `[18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]`.

Unique positions: 1; recorded actions: 48.

The same actor's separately seeded sampled evaluation on this world ended in **hazard**
after 5 steps. Sampled actions: `[3, 1, 3, 3, 2]`.

A repeated position under greedy reactive inference can produce a deterministic loop.
This trace establishes behavior, not whether entropy, aliasing, or optimization caused it.

## dense: timeout

World `c2dc61177b7aafb2660c10627ee1fada2c08d97f4b15785856a242f8cd5ed485` (generator seed 5100009).

```text
#######
#S....#
#..#..#
#!...##
#G.!..#
#...!.#
#######
```

`#` rock, `!` hazard, `S` start, `G` goal. Coordinates are flat row-major indices.

Expert actions: `[1, 2, 2, 2, 3]`.

Policy actions: `[2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1]`.

Visited positions: `[8, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16]`.

Unique positions: 3; recorded actions: 48.

The same actor's separately seeded sampled evaluation on this world ended in **success**
after 7 steps. Sampled actions: `[2, 1, 2, 2, 2, 0, 3]`.

A repeated position under greedy reactive inference can produce a deterministic loop.
This trace establishes behavior, not whether entropy, aliasing, or optimization caused it.

## barrier: timeout

World `19f7a181b9ba1fa174b8b0c3992238251306eae847de706a9020834e01c4d2ff` (generator seed 5200001).

```text
#######
#..#S##
#..#..#
##.#..#
#.G#..#
#.....#
#######
```

`#` rock, `!` hazard, `S` start, `G` goal. Coordinates are flat row-major indices.

Expert actions: `[2, 2, 2, 2, 3, 3, 0]`.

Policy actions: `[2, 2, 2, 2, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]`.

Visited positions: `[11, 18, 25, 32, 39, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37, 38, 37]`.

Unique positions: 7; recorded actions: 48.

The same actor's separately seeded sampled evaluation on this world ended in **success**
after 13 steps. Sampled actions: `[2, 2, 0, 2, 0, 2, 2, 2, 3, 3, 3, 0, 1]`.

A repeated position under greedy reactive inference can produce a deterministic loop.
This trace establishes behavior, not whether entropy, aliasing, or optimization caused it.

