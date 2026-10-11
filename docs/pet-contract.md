# Companion contract (v0.2)

The companion is a game layer around the unchanged navigation experiment.
Care, dialogue, growth and autobiography are authored game rules. Navigation
and teaching use the actual neural policy. Pet needs are not network inputs.

## Time and care

- A new pet starts with fullness 80, energy 85 and affection 70 (out of 100).
- Away-time decay is 4, 2 and 1 points per hour respectively. At most 24 hours
  elapse on one update. Needs bottom out at 20; the pet never dies from absence.
- A backwards clock advances nothing. The saved clock never moves backwards.
- Food restores 25 fullness; a cuddle restores 12 affection. Each has a
  60-second cooldown. Values cap at 100.
- A nap lasts 30 seconds and restores 0.75 energy per second actually slept.
  Waking early keeps only the energy already recovered. Exploring wakes the pet.
- Successful care earns 4 growth points per care type at most once per 15
  minutes. Repeated clicks cannot advance growth.
- A completed expedition costs 3 energy. Success adds 8 fullness and 3 affection.
  The first completion of each exact garden per UTC day earns 8 growth points
  for success or 2 for a failed attempt, for at most 128 unique gardens per day.
  Repeating that garden earns none.
- Growth stages start at 0 (Hatchling), 40 (Scout), and 120 (Trailkeeper).
  These cosmetic stages are not a measure of neural capability.

## Persistence and conversation

One versioned browser-local save contains the pet, up to 80 memories, 24 chat
messages, the garden, the current episode, up to 2,400 teaching actions and the
adapted neural weights. Returning resumes the episode in a paused state.
The selected r1 recurrent brain and optional reset-memory control are also saved.
Its hidden state is reconstructed from the saved garden and actual action prefix
using the fixed r1 checkpoint. Scrubbing the journal reconstructs the state at
that frame; later frames do not leak into earlier decisions. Browser teaching
updates reactive actors only. Comparison replays use the same garden without
awarding extra companion growth.
JSON backups can move a pet between browsers. Clearing site data removes it.
There is no account, server, background simulation or cross-device sync.

Imports are limited to 1 MB and validated before replacing state, including
policy dimensions/finite weights and replay legality. Broken saves are reported;
they are not silently presented as successfully restored.

Conversation recognizes a small set of commands: greeting, food, cuddle,
nap/wake, explore, needs, memories and lessons. Replies refer to actual saved
state and commands trigger visible game actions. Unknown language gets a clear
fallback. This is not a language model or evidence of language understanding.
