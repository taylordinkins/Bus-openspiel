# Bus — Action Catalogue

**Status:** DRAFT (Phase 0 unit 4). Companion to `docs/RULES_SPEC.md` (R-IDs refer to it). If this file and the spec disagree, the spec wins and this file is fixed. **Action IDs freeze when Phase 1 closes** (CLAUDE.md; replays and playtest reports depend on them). Board counts come from `data/board.yaml` through `bus/board.py`; nothing here hard-codes a node, street or site id.

Design (PLAN §5.4, D9): atomic typed sub-decisions, one small vocabulary reused across phases, legal-action masks do the work. Bus is deterministic with perfect information (R-073): there are **no chance nodes**, `max_chance_outcomes` = 0.

## 1. Conventions

1. **Seats** are numbered from 0. An action never names the acting seat: the acting seat is `current_player()`.
2. **Index symbols.** All sizes are read from the board at import time; the numbers in brackets are the counts for the current `board.yaml` (36 nodes, 70 streets, 47 sites, 2 stations) and are checked by a test.

   | Symbol | Range | Meaning |
   |---|---|---|
   | `N` | [36] | number of nodes; node `n` is `board.nodes[n]` (ids are contiguous 0..N-1) |
   | `S` | [70] | number of streets; street `s` is `board.streets[s]` |
   | `T` | [47] | number of sites; site `t` is `board.sites[t]` |
   | `K` | [2] | number of stations; station index `k` is the position in `board.stations` (ascending node id) |
   | action `a` | 0..6 | resolution order of R-022 / R-029: 0 Lines, 1 Buses, 2 Passengers, 3 Buildings, 4 Clock, 5 Vrrooomm!, 6 Starting Player |
   | end `e` | 0..1 | end of a line's ordered path (R-035): 0 = start of the path, 1 = its last street |
   | type `y` | 0..2 | building type, in clock-cycle order (R-050): 0 house, 1 office, 2 pub |

3. **Line orientation convention.** A line is stored as an ordered list of street ids. Its node walk starts at the endpoint of the first street that is not shared with the second street; for a **single-street line** end 0 is at the street's `a` node and end 1 at its `b` node (board.yaml order). Extending from end 0 prepends the new street, from end 1 appends it. (An encoding convention only; the spec requires just that ends are never ambiguous, R-035.)
4. **No `DONE`.** Every decision is one atomic action; a marker's resolution ends by itself when its units are used up or lost (R-034, R-040).
5. **Decision nodes are emitted even when only one action is legal.** The engine never skips a node that has at least one legal action, so action histories and replays are uniform. `bus.rules.forced_action(state)` returns the id when exactly one action is legal; the UI and bots auto-play it (PLAN §5.3). Steps with **no** legal action are applied automatically inside the previous transition (no node, no id): see §4 "automatic steps".
6. **Canonicalisation.** Equal outcomes should have one action sequence where it is cheap: only R-063 is applied (the lowest-id empty site of the clock's type at a node stands for all of them, §3 `DELIVER_TO`). Order of interchangeable units inside one marker (several builds, several passengers, several Vrrooomm! moves) is **not** canonicalised.
7. **Move number** (R-062, R-070): the move number of an action is the length of the OpenSpiel history after it is applied (the first action is move 1, counting every decision node including forced ones). It is recorded for each seat on every `DELIVER_TO` that scores.

## 2. Decision IDs (`num_distinct_actions` = 376)

Contiguous, no gaps. IDs `0..375`.

| IDs | Name | Formula | Count | Meaning | Rules |
|---|---|---|---|---|---|
| 0–6 | `SELECT(a)` | `0 + a` | 7 | place an Action marker on action `a` (it goes on the earliest free letter) | R-022..R-026 |
| 7 | `DROP_OUT` | `7` | 1 | drop out for the rest of the round | R-024 |
| 8–147 | `EXTEND(e, s)` | `8 + e*S + s` | 2S [140] | place a Bus Line marker on street `s` at end `e` of the acting seat's line | R-017, R-018, R-036..R-042 |
| 148–288 | `BUILD(t, y)` | `148 + t*3 + y` | 3T [141] | build a type-`y` building on empty site `t` | R-016, R-047, R-048 |
| 289–290 | `PASSENGER_TO(k)` | `289 + k` | K [2] | place one passenger from the supply on station `k` | R-045 |
| 291 | `CLOCK_ADVANCE` | `291` | 1 | advance the clock one step | R-050, R-051 |
| 292 | `CLOCK_PAUSE` | `292` | 1 | pause: keep the clock, take a time stone | R-050, R-052, R-053 |
| 293–328 | `PICK_PASSENGER(n)` | `293 + n` | N [36] | Vrrooomm!: choose a waiting passenger standing at node `n` | R-057, R-058 |
| 329–375 | `DELIVER_TO(t)` | `329 + t` | T [47] | Vrrooomm!: put the picked passenger into the empty building on site `t` | R-057, R-059, R-063 |

Named constants (`bus/actions.py`): `SELECT_BASE=0`, `DROP_OUT=7`, `EXTEND_BASE=8`, `BUILD_BASE=EXTEND_BASE+2*S`, `PASSENGER_TO_BASE=BUILD_BASE+3*T`, `CLOCK_ADVANCE=PASSENGER_TO_BASE+K`, `CLOCK_PAUSE=CLOCK_ADVANCE+1`, `PICK_PASSENGER_BASE=CLOCK_PAUSE+1`, `DELIVER_TO_BASE=PICK_PASSENGER_BASE+N`, `NUM_DISTINCT_ACTIONS=DELIVER_TO_BASE+T`. Helpers `encode_*`/`decode_action(id) -> (kind, params)` live there; no numeric literals elsewhere.

Reuse across phases:
- `BUILD` serves both setup buildings (R-016) and the Buildings action (R-047).
- `EXTEND` serves both setup line markers (R-017, R-018) and the Lines action. **The first setup marker has no end yet** (the line is empty): it is encoded as `EXTEND(0, s)`, with `e = 0` as a placeholder for "no end"; `EXTEND(1, s)` is never legal in that phase. The marker creates the single-street line `[s]`.
- Sizes and `K` are recomputed from the board; a change to `board.yaml` that alters `N`, `S`, `T` or `K` changes the layout, which is allowed only before Phase 1 closes.

## 3. Decision points

Phase codes (also used in scenario JSON, `docs/SCENARIOS.md`):

| Code | Name | Rules |
|---|---|---|
| `SB` | setup buildings | R-016 |
| `L1` | setup first line marker | R-017 |
| `L2` | setup second line marker (reverse order) | R-018 |
| `SEL` | Phase 1 action selection | R-022..R-028 |
| `LN` | Lines marker resolving | R-030..R-042 |
| `PX` | Passengers marker resolving | R-045 |
| `BD` | Buildings marker resolving | R-047, R-048 |
| `CK` | Clock marker resolving | R-050..R-055 |
| `VP` | Vrrooomm! pick step | R-056..R-062 |
| `VD` | Vrrooomm! deliver step | R-057, R-059, R-063 |
| `END` | terminal | R-066..R-072 |

Buses (R-043) and Starting Player (R-064) have no decision and no phase code.

### 3.1 `SB` — setup buildings (R-016)
- **Who:** seats in turn order from the start player (seat 0, R-019); each seat acts twice in a row.
- **Legal:** `BUILD(t, y)` for every empty site `t` of the lowest open tier (R-010; re-evaluated before each building) and every `y` in 0..2. Never a node with no legal action: 2n buildings fit in tier A.
- **Pending state:** `setup_builds_done[seat]` (0..2) or equivalently the global count `0..2n`; the actor is derived from it.
- Next: after the last building, `L1`.

### 3.2 `L1` — first line marker (R-017)
- **Who:** turn order from the start player, one marker each.
- **Legal:** `EXTEND(0, s)` for every street `s` (0..S-1), occupied or not.
- **Pending state:** count of seats done.

### 3.3 `L2` — second line marker (R-018)
- **Who:** reverse turn order (R-003).
- **Legal:** `EXTEND(e, s)` where `s` is a candidate street (R-037) at end `e` of the seat's one-street line, allowed by R-038 evaluated for that end and by the shared-end exception (R-039, which applies here per R-039's last line). Because the line has two distinct end nodes and a node pair has at most one street, `(e, s)` is determined by `s`.
- **Pending state:** count of seats done.
- Next: after the last seat, `SEL` of round 1.

### 3.4 `SEL` — action selection (R-022..R-028)
- **Who:** the seat whose turn it is in the pass; passes cycle through turn order from the round's start player. A seat with no Action markers left, or that has dropped out, is skipped with **no node** (R-027, R-024).
- **Legal:**
  - `SELECT(a)` for every action `a` with a free space (R-025, R-026), always legal otherwise (a blocking placement is allowed).
  - `DROP_OUT` from the third pass on (never in the two mandatory passes, R-023), as long as the seat has markers. If the seat has markers but every action space is full, `DROP_OUT` is the only legal action (R-027); the node is still emitted.
- **Pending state:** `round`, `start_player`, `pass_index` (0-based; passes 0 and 1 are mandatory), `current_player`, `dropped[seat]`, the action board (per action, the list of seats on letters a, b, … in order), `markers_left[seat]`. A seat with 1 marker left places it in pass 0 and is skipped in pass 1 (R-027).
- **Ends** (R-028) when every seat has dropped out or is skipped; Phase 2 starts at `a = 0` (Lines) with the first marker in resolution order (R-030) that exists.

### 3.5 `LN` — Lines marker (R-030..R-042)
- **Who:** the owner of the marker being resolved. Markers resolve highest letter first (R-030).
- **Power:** `p = M - k` (R-031), `+1` at 5 players (R-033), `M` read now (R-032). `p <= 0`: no node; the marker is spent.
- **Legal (one node per unit, `p` units):** `EXTEND(e, s)` such that
  1. `s` is a candidate street at end `e` (R-037: incident to that end's node, not already in the seat's line),
  2. `s` passes the placement rule for that end (R-038) or the shared-end exception (R-039), evaluated with the streets placed earlier in this same resolution counting as occupied,
  3. the seat has a Bus Line marker left (25 minus markers on the board),
  4. **maximal resolution (R-040, R-042):** the choice must keep the maximum achievable number of remaining placements reachable. Let `r` = remaining units including this one. For each candidate `(e, s)` compute `best(e, s)` = 1 + the maximum number of further placements (at most `r - 1`, bounded by markers in supply) over all legal continuations under 1–2. Only candidates with `best(e, s) = max over all candidates` are legal. This is a depth-first search of depth at most `r <= 6` and is computed on the fly.
  For a full-line loop (both ends at one node) both ends are listed with their own candidate sets (R-041); the two ends are different actions even where the geometry looks symmetrical.
- **No legal extension:** the remaining units are lost (R-040) and the node is not emitted. If no extension is legal at the first unit, the marker produces no node.
- **Pending state:** `resolving = (action 0, letter k, owner)`, `units_left` (1..6). Each `EXTEND` decrements it.
- After the last unit, resolution moves to the next marker, then to Buses.

### 3.6 Buses (R-043) — automatic
The owner gets +1 bus if they have fewer than 5; otherwise nothing. No decision, no id. M updates immediately (R-032).

### 3.7 `PX` — Passengers marker (R-045)
- **Who:** the marker owner; letter a first (R-030).
- **Legal (one node per unit):** `PASSENGER_TO(k)` for each station index `k`; the station need not be on the owner's line.
- **Power:** `p = M - k` (R-031); units beyond the supply are lost (no node when the supply is empty). `p <= 0`: no node.
- **Pending state:** `units_left`, `supply`.

### 3.8 `BD` — Buildings marker (R-047, R-048)
- **Who:** the marker owner; highest letter first.
- **Legal (one node per unit):** `BUILD(t, y)`: any empty site of the lowest open tier, recomputed before each unit, and any `y`. The site need not be on any line. When no empty site is left, the remaining units are lost; no node.
- **Pending state:** `units_left`. After the unit that fills the last site the sites-full condition is recorded (R-049) and resolution continues; the end check is at round end.

### 3.9 `CK` — Clock marker (R-050..R-055)
- **Who:** the marker owner.
- **Legal:** `CLOCK_ADVANCE` and `CLOCK_PAUSE`. Pause is legal because a stone is always on the clock when a marker resolves (R-052); the engine asserts it.
- `CLOCK_PAUSE` that takes the last stone ends the game immediately (R-053): terminal, nothing else resolves, the marker already spent.
- After the choice (if the game continues) passenger movement runs **automatically** with no further decision: step-out (R-054), then step-in at each node by ascending site id (R-055).
- **Pending state:** none beyond the marker owner.

### 3.10 `VP` / `VD` — Vrrooomm! (R-056..R-063)
- **Who:** the marker owner; letter a first. Not dwindling (R-056).
- **Forced count** (R-060): `moves = min(buses, waiting passengers on the owner's line, empty buildings of the clock's type on the owner's line)`, computed when the marker starts. If it is 0 no node is emitted. Each move reduces all three quantities by 1, so the count is always achievable whatever the owner picks; the owner cannot stop early.
- **`VP` legal:** `PICK_PASSENGER(n)` for each node `n` that is on the owner's line (R-058) and has at least one waiting passenger. Passengers inside buildings are not eligible.
- **`VD` legal:** `DELIVER_TO(t)` where `t` is the **canonical site**: for every node `m` on the owner's line that has at least one empty building of the clock's type, the **lowest site id** among those empty buildings at `m` (R-059, R-063). The passenger's own node is allowed (R-061).
- **Pending state:** `moves_left` (1..5), `picked_node` (none in `VP`, set in `VD`), the owner. `DELIVER_TO` moves the passenger, fills the building, adds 1 point, sets `last_score_move[owner]` to the move number (R-062), decrements `moves_left`; the next node is `VP` while `moves_left > 0`.
- No sequence of moves is ever blocked: after any `PICK_PASSENGER` at least one `DELIVER_TO` is legal.

### 3.11 Starting Player and round end (R-064..R-067) — automatic
The Starting Player marker (if any) sets the next start player; otherwise the start passes to the next seat. Spent markers are removed (R-065). End checks (R-066): if any hold, state becomes `END`; otherwise the next round's `SEL` begins with an empty action board. No decision, no id.

### 3.12 `END` — terminal
No legal actions. `returns()` per R-071 / R-072. The recorded end-type list (R-066) and winners (R-069) are exposed through the observer string and `game_result(state)`.

### Automatic steps (no node, no id)
Skipped seats in `SEL`; power ≤ 0 markers; Lines with no legal extension; the Buses action; Passengers with empty supply; Buildings with no empty site; Vrrooomm! with `moves = 0`; passenger step-out and step-in after the Clock choice; Starting Player and round-end bookkeeping. They are applied by the transition that produced them, so the next state is always a decision node or terminal.

## 4. `action_to_string` format

Seat labels are `P0`…`P4`. Board references use the ids from `board.yaml`: `n12` node, `s34 (n7-n12)` street with its endpoints in `a-b` order, `site 9 (B, n12)` site with tier and node. Format is `<seat>: <verb ...>`:

| Kind | String |
|---|---|
| `SELECT` | `P2: select Lines` (action names `Lines`, `Buses`, `Passengers`, `Buildings`, `Clock`, `Vrrooomm!`, `Starting Player`) |
| `DROP_OUT` | `P2: drop out` |
| `EXTEND` (setup `L1`) | `P0: start line on s34 (n7-n12)` |
| `EXTEND` (other) | `P1: extend end 0 via s34 (n7-n12) to n12` (end index, street, new end node) |
| `BUILD` | `P1: build house at site 9 (B, n12)`; type word `house`/`office`/`pub` |
| `PASSENGER_TO` | `P0: passenger to station n5` |
| `CLOCK_ADVANCE` | `P3: clock advance` |
| `CLOCK_PAUSE` | `P3: clock pause` |
| `PICK_PASSENGER` | `P2: pick passenger at n12` |
| `DELIVER_TO` | `P2: deliver to site 9 (B, n12)` |

`State.__str__` delegates to the observer string (CLAUDE.md); it also prints the phase and pending sub-decision, so distinct states never print alike.

## 5. `action_meta(state, action_id)`

Returns a JSON-serialisable dict for any legal action (and, via `decode_action`, a static dict for any id). Common keys: `id`, `kind`, `player` (acting seat), `label` (the §4 string). Targets are null when not applicable.

| `kind` | extra keys |
|---|---|
| `select` | `action` (0..6), `action_name`, `space` (letter index 0..5 the marker would take, R-025) |
| `drop_out` | none |
| `extend` | `end` (0/1), `street`, `from_node` (the end's node; null for the setup first marker), `to_node` (new end), `setup` (`"first"`, `"second"` or null), `shared_end` (true if only legal through R-039 or because every candidate was occupied, R-038: for UI hints) |
| `build` | `site`, `node`, `tier`, `building_type` (`"house"`,`"office"`,`"pub"`), `setup` (bool) |
| `passenger_to` | `station` (station index), `node` |
| `clock_advance` | `from` and `to` clock types |
| `clock_pause` | `stones_left_after`, `ends_game` (bool) |
| `pick_passenger` | `node`, `waiting` (count there) |
| `deliver_to` | `site`, `node`, `tier`, `building_type`, `picked_node`, `same_node` (bool) |

The UI makes every legal action reachable by clicking: streets for `extend`, sites for `build` (type chosen in a popover, `building_type` per id), station nodes for `passenger_to`, the clock for `clock_*`, passenger nodes then site nodes for `pick`/`deliver`, and the action board for `select`/`drop_out` (D9).

## 6. `max_game_length`

Counting decisions (one history entry per node, forced nodes included; no chance nodes). `n` = players (≤ 5), 20 Action markers each.

- **Setup:** `2n` buildings + `n` first markers + `n` second markers = `4n`.
- **Selection:** every `SELECT` places a marker, at most `20n` in the whole game (markers are never recovered, R-004, R-065). Every `DROP_OUT` follows at least two placements by the same seat in the same round (mandatory passes, R-023) and happens at most once per seat per round, so drop-outs ≤ (placed markers)/2 ≤ `10n`.
- **Resolution:** each placed marker produces at most 10 decision nodes: Lines ≤ 6 (power 5 + 1 at 5 players, minus 0 for letter a), Passengers ≤ 5, Buildings ≤ 5, Clock 1, Vrrooomm! ≤ 2·5 = 10 (pick + deliver per bus, 5 buses at most, R-044), Buses and Starting Player 0. So resolution ≤ `10 · 20n = 200n`.

Total ≤ `4n + 20n + 10n + 200n = 234n`, so **`max_game_length` = 234·n**, which is **1170** at `n = 5` (the `Game` object returns the bound for its configured `players`; 702 at n = 3, 936 at n = 4). The bound is loose; the random-simulation tests assert the bound holds and record the observed maximum.

## 7. Open points for encoding

Resolved in Phase 1 unit 13:
- **Observation tensor** (PLAN 5.2): absolute (perfect information, identical for every observing player), includes a one-hot of the seat to act. Flat float32 with named views (street x seat occupancy and end markers, node waiting and line membership, site type one-hot and occupied flag, clock one-hot, stones, buses, markers left, line markers left, points, action board action x letter x seat, phase one-hot, round, selection pass, dropped seats, end types, pending units/moves left, cursor letter, picked node). Counts are scaled to [0, 1] by their maximum, documented in the `bus/observer.py` docstring. `last_score_move` and move numbers are not in the tensor (the string shows them).
- `bus.rules.forced_action(state)` accepts a state or a core; None at END. `action_meta` and `game_result` live in `bus/game.py`. `shared_end` is true whenever the street is already in another seat's line (R-038 all-occupied case and R-039), false for the first setup marker; the first setup marker has `from_node` and `to_node` null, `end` 0, `setup` "first"; `deliver_to.building_type` is the clock's type.
- `game_result(state)` returns `{"end_types", "winners", "final_scores", "returns"}` at END, None before. The observer string at END appends `end: <types> winners: P.. scores: [...]`. `current` is 0 at END (canonical) so dump/load round-trips.

Performance note for Phase 1: the R-042 lookahead is a depth-first search of depth <= 6 on every Lines node; memoise per node if profiling shows a cost.
