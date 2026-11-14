# Bus — Scenario Format

**Status:** DRAFT (Phase 0 unit 4). Companion to `docs/RULES_SPEC.md` (R-IDs) and `docs/ACTIONS.md` (action ids, phase codes). If this file disagrees with the spec, the spec wins and this file is fixed. Board facts (node, street, site, station ids, counts) come from `data/board.yaml`; this file never states a rule in terms of real ids.

A **scenario** is a JSON document describing a position in one game of Bus, plus optionally the action ids to play from it. One format serves three users:

1. **Rule tests** load a scenario as the start state of a fixture (files in `tests/scenarios/*.json`, loader `bus/scenario.py`, harness `tests/harness.py`).
2. **The setup editor** (Phase 5) produces and consumes scenarios. Per PLAN D11 an editor-built game is persisted as **the scenario JSON plus the action history after it**; the server never accepts a `serialize()` string from a client and validates scenarios with the same loader as the tests.
3. **Mirroring a physical table**: a scenario transcribes a real table at a decision point, so a human game can be continued in the engine.

A scenario is **a position, not a history**. It need not be reachable by legal play; the loader checks only the invariants of §6. Seats are numbered from **0** everywhere (`seats[0]` is P0). Files are UTF-8 JSON; the loader also accepts YAML with the same schema. Bus has no chance, so there is no seed, no `chance` list and no sampling.

## 1. Conventions
- **Ids** are the integer ids of `board.yaml`: nodes `0..N-1`, streets `0..S-1`, sites `0..T-1`. Unknown ids are errors (`S-ID`).
- **Types:** `"house"`, `"office"`, `"pub"`. **Clock** value is one of those. **Actions:** `"lines"`, `"buses"`, `"passengers"`, `"buildings"`, `"clock"`, `"vrrooomm"`, `"starting_player"`.
- **Lines** are ordered lists of street ids, oriented as ACTIONS §1.3 (end 0 = start of the list; a one-street line has end 0 at the street's `a` node). Consecutive streets must share a node.
- Unknown keys are errors (`S-SCHEMA`). A scenario may rest at **any decision node** (not only at a turn boundary): the pending sub-decision (§4) carries what is needed. `dump_scenario(state)` always emits a resolved scenario (all fields explicit).

## 2. Top-level fields

| Field | Type | Req | Default | Meaning |
|---|---|---|---|---|
| `format` | `1` | yes | – | format version |
| `description` | string | no | `""` | free text |
| `parameters` | object | no | R-001 defaults | `players` 3..5 (4), `returns` `"win_loss"`\|`"margin"` (`"win_loss"`) |
| `round` | int ≥ 0 | no | 0 (with `phase` default `SB`) | current round number (`0` while in setup phases `SB`, `L1`, `L2`) |
| `phase` | code | no | `SEL` if `round`≥1 else `SB` | phase code of ACTIONS §3: `SB`, `L1`, `L2`, `SEL`, `LN`, `PX`, `BD`, `CK`, `VP`, `VD`, `END` |
| `current_player` | seat | no | derived (§4) | seat to act; required in `END`-free phases when not derivable |
| `start_player` | seat | no | 0 | start player of the current round (R-019, R-064); also the setup turn-order anchor |
| `clock` | type | no | `"house"` | clock hand (R-014) |
| `stones_on_clock` | int ≥ 0 | no | 5 (4 if 3 players) | stones still on the clock (R-006) |
| `seats` | list of seat objects (or map `"0"`…) | no | defaults | one per seat, §3 |
| `buildings` | list | no | `[]` | §3.2 |
| `waiting` | map node → int | cond. | the 4 roundabout passengers (R-015) in `SB`/`L1`/`L2`; **required** when `round` ≥ 1 (`S-SCHEMA` if missing; `{}` is allowed) | waiting passengers by node (§3.3) |
| `supply` | int | no | derived (15 minus all passengers on the board) | passengers in supply (R-005); if given it must equal the derived value |
| `action_board` | map action → list | no | all empty | markers on the board (§3.4) |
| `pending` | object | cond. | `{}` | sub-decision state (§4) |
| `end` | object | cond. | – | in `END` only: `{"types": [...]}`; terminal check data (§4.7) |
| `move_offset` | int ≥ 0 | no | max of the seats' `last_score_move` (0 if all null) | move number of the scenario position itself: an action applied after load has move number `move_offset` + its index in the post-load history + 1 (ACTIONS §1.7, R-062, R-070) |
| `allow_unreachable` | bool | no | false | relaxes only the `points` ↔ `last_score_move` link of §3.1 |
| `strict` | bool | no | false | §6: tier-fill order (R-010) becomes an error |
| `history` | list of ints | no | `[]` | action ids applied after the scenario |
| `expect` | object | no | none | assertions after `history` (§5) |

## 3. Objects

### 3.1 Seat object (`seats[i]`)

| Field | Type | Default | Meaning / rule |
|---|---|---|---|
| `points` | int ≥ 0 | 0 | R-008 |
| `stones` | int ≥ 0 | 0 | time stones held (R-006, R-052) |
| `buses` | 1..5 | 1 | R-044 |
| `markers_left` | 0..20 | 20 | Action markers still in hand (not those lying on the action board this round; R-004, R-065) |
| `line` | list of street ids | `[]` | ordered path (R-035); empty means no line yet (valid only in `SB`, `L1`, and for seats not yet served in `L1`) |
| `last_score_move` | int ≥ 1 \| null | null | move number of the last scoring move (R-062, R-070); null if never scored |

Bus Line markers left = 25 − `len(line)`; it is derived, not stored. `last_score_move` must be null iff `points` = 0 (an unreachable fixture may set both freely only with `"allow_unreachable": true` at top level).

### 3.2 Buildings

```json
{"site": 12, "type": "pub", "occupied": true}
```
`occupied` (default false) = a passenger is inside (R-054..R-059). Sites not listed are empty. Buildings have no owner. A site appears at most once.

### 3.3 Waiting passengers
`waiting` maps node id (as a string key) to a count ≥ 1. Passengers are identical (R-005). Conservation: `sum(waiting) + count(occupied buildings) + supply = 15`.

### 3.4 Action board
For each action a list of length equal to its spaces (1 for Buses, Clock, Starting Player; 6 for the rest, R-022): each entry is the seat that placed the marker on that letter or `null`. Filled letters must be contiguous from the start (R-025): `[0, 2, null, null, null, null]` is valid, `[null, 1, …]` is not. Outside Phase 1/2 the board must be empty. During resolution the board still shows the round's markers; which of them are done is derived from `pending.cursor` (§4.2).

## 4. Phases and pending sub-decision

`pending` holds exactly the fields below for the phase; extra or missing keys are errors (`S-PHASE`). `current_player` defaults are given in the second column.

| Phase | Default `current_player` | `pending` fields |
|---|---|---|
| `SB` | derived from `done` | `{"done": int}` buildings placed so far, 0..2n-1; seat = turn order index `done // 2` |
| `L1` | derived | `{"done": int}` 0..n-1 first markers placed |
| `L2` | derived | `{"done": int}` 0..n-1 second markers placed (reverse order) |
| `SEL` | next seat per pass | `{"pass": int, "dropped": [seats]}` (see §4.1) |
| `LN` | marker owner | `{"cursor": {"letter": k}, "units_left": int}` |
| `PX` | marker owner | same shape as `LN` |
| `BD` | marker owner | same shape as `LN` |
| `CK` | marker owner | `{"cursor": {"letter": 0}}` |
| `VP` | marker owner | `{"cursor": {"letter": k}, "moves_left": int}` |
| `VD` | marker owner | `{"cursor": {"letter": k}, "moves_left": int, "picked_node": node}` |
| `END` | – | `{}` plus top-level `end` |

### 4.1 `SEL`
`pass` ≥ 0 is the 0-based pass; passes 0 and 1 are mandatory (R-023). `dropped` lists seats that have dropped out this round. Consistency (`S-SEL`): a dropped seat has placed ≥ 2 markers; every seat before `current_player` in turn order has placed exactly `pass + 1` markers and every seat from `current_player` on has placed exactly `pass` (dropped seats and seats with `markers_left` 0 are exempt). `current_player` is the first such active seat with `pass` markers placed. Markers placed are counted from `action_board`.

### 4.2 Resolution phases and `cursor`
`cursor.letter` is the 0-based letter index (a = 0) of the marker being resolved; the action is implied by the phase (`LN` → Lines, `PX` → Passengers, `BD` → Buildings, `CK` → Clock, `VP`/`VD` → Vrrooomm!). Markers that resolve before the cursor (earlier actions entirely; same action in R-030 order) are done and must not hold stale state; the loader does not alter the board for them. The owner is `action_board[action][letter]` and must not be null. `units_left` is the remaining units of the current marker including the pending one (1..6); `moves_left` likewise (1..5). The M used for the current marker's power is recomputed from the seats' `buses` and is not stored.

### 4.3 Round 0
Setup phases use `round` 0. `current_player` is derived from turn order starting at `start_player` (reverse in `L2`).

### 4.7 `END`
`end.types` is a list drawn from `"sites_full"`, `"last_stone"`, `"markers"` (R-066), non-empty; `"last_stone"` must stand alone (R-053).

## 5. History and expectations

`history` is applied after load with the ACTIONS ids; every id must be legal at that point (`S-HISTORY`). `expect` assertions (all optional) are checked after `history`: `phase`, `current_player`, `legal_actions` (exact sorted ids), `legal_includes`, `legal_excludes`, `points` (list per seat), `final_scores`, `returns`, `winners`, `buses`, `stones`, `clock`, `lines` (per seat), `buildings`, `waiting`, `supply`, `start_player`, `terminal` (bool), `end_types`. Rule tests build their fixtures here, never in test code, so they survive refactors (`.claude/rules/engine.md`).

## 6. Validation rules (`ScenarioError` with code)

| Code | Check |
|---|---|
| `S-SCHEMA` | unknown or missing keys, wrong types, `format` ≠ 1 |
| `S-ID` | node/street/site id out of range; seat out of range |
| `S-PLAYERS` | `players` not 3..5; `seats` length ≠ players |
| `S-LINE` | consecutive streets of a line do not share a node; a street appears twice in one line (R-037); line longer than 25 (R-004); a loop with a repeated street |
| `S-BUS` | `buses` outside 1..5 (R-044) |
| `S-MARKERS` | `markers_left` + markers of that seat on the action board > 20; `markers_left` outside 0..20 |
| `S-STONES` | `stones_on_clock` + sum of seat `stones` ≠ 5 (4 at 3 players) (R-006); `stones_on_clock` = 0 outside phase `END` (R-053) |
| `S-SITES` | duplicate sites in `buildings`; invalid type |
| `S-PASSENGERS` | conservation fails (§3.3); `waiting` key not a node; `occupied` building count + waiting + supply ≠ 15 (R-005) |
| `S-BOARD` | action board shape wrong; letters not contiguous (R-025); non-empty board outside phases `SEL`..`VD`, except `END` with `end.types` = `["last_stone"]` (the immediate end of R-053 keeps the round's markers on the board) |
| `S-PHASE` | `phase` not compatible with `round` (`SB`/`L1`/`L2` need round 0; others need ≥ 1); pending keys wrong; `current_player` is not the seat the phase would call; `cursor` points at an empty letter |
| `S-SEL` | §4.1 |
| `S-POINTS` | `last_score_move` null mismatch (§3.1); a seat's `last_score_move` > `move_offset` |
| `S-SETUP` | in `SB`/`L1`/`L2`, seats' buildings/lines inconsistent with `done`; a non-empty line before `L1` |
| `S-END` | `END` with a missing/invalid `end` block |
| `S-HISTORY` | an id is illegal or not in range |

Not checked (fixtures may be unreachable): whether a resolution node (`LN`, `PX`, `BD`, `VP`, `VD`) has any legal action, whether `moves_left` ≤ the forced count (R-060) or `picked_node` is on the owner's line — the engine never emits such a node (ACTIONS §3), but a fixture may load one and then has `legal_actions() == []`; the tier-fill order of buildings (R-010), whether lines respect R-038/R-039 placement, the relation between points and the number of occupied buildings, who holds which stones. A `"strict": true` flag turns the first of these into an error for editor-built games that start from a fresh setup.

## 7. Defaults ("fresh") and shorthand
`{"format": 1}` is the state right after R-014: round 0, phase `SB`, `done` 0, seat 0 to act, empty board with the four roundabout passengers (R-015). A shorthand scenario lists only differing fields: for example a round-1 selection fixture is `{"format":1,"round":1,"phase":"SEL","seats":[…],"waiting":{…}}` with the setup buildings, lines and waiting passengers supplied (`waiting` is required from round 1 so the four roundabout passengers of R-015 are never dropped silently). The loader resolves shorthand and `dump_scenario` emits it in full.

## 8. Named reusable scenarios

Files live in `tests/scenarios/`, named `<id>.json`. Each is described by intent and the rules it exercises. They use real ids from `board.yaml` chosen by the author; the table below states the shape, not ids.

| Id | Shape and intent | Rules |
|---|---|---|
| `fresh_4p` | the fresh scenario (§7) | R-014..R-019 |
| `setup_tier_rollover` | `SB`, tier A with exactly one empty site left, seat acts; second building must go to tier B | R-010, R-016, R-048 |
| `setup_second_marker` | `L2` with a one-street line and an unoccupied and an occupied candidate at each end | R-018, R-038 |
| `lines_blue_midline` | rulebook Blue example: Blue's end touches the middle of Red's line, no exception | R-039 counter-example |
| `lines_red_cd` | rulebook Red example: one end with an unoccupied candidate, other end all occupied | R-038 |
| `lines_shared_end` | another seat's end at the node, different last street; plus the same-arrival counter-example | R-039 |
| `lines_full_loop` | a line whose two ends sit at one node; both end choices legal | R-041 |
| `lines_partial_loop` | a loop in the middle of the path, ends elsewhere, other seat's end inside the loop | R-035, R-039 |
| `lines_power_m3` | markers on a, b, c, d of Lines, M = 3: powers 3, 2, 1, 0 | R-030, R-031 |
| `lines_power_5p` | five players, M = 3: powers 4, 3, 2, 1, 0, −1 | R-033 |
| `lines_dead_end` | one live end allowing 2 further placements with power 3: third lost | R-040 |
| `lines_maximal_choice` | power 2, one dead end, a choice of two streets where only one allows both placements | R-042 |
| `lines_no_markers` | seat with 25 Bus Line markers on the board | R-040 |
| `select_mandatory` | `SEL` pass 0 and 1: `DROP_OUT` illegal; same action twice allowed | R-023 |
| `select_full_action` | an action with all letters full, an ineffective action still choosable | R-025, R-026 |
| `select_low_markers` | a seat with 1 marker left skipped in pass 1 | R-027 |
| `select_all_out` | selection ending and Phase 2 start | R-028 |
| `buses_raise_m` | Buses before Passengers in the same round changes M | R-032, R-043 |
| `buses_five` | seat already with 5 buses resolves Buses | R-043, R-044 |
| `passengers_short_supply` | supply 1, marker power 3 | R-045 |
| `buildings_tier_advance` | power 3 across a tier boundary | R-047, R-048 |
| `sites_full_round_end` | last site filled mid-round; game ends at round end | R-049, R-066, R-067 |
| `clock_cycle` | advance from each of house/office/pub | R-051 |
| `clock_pause` | pause with stones left; stones and passenger re-entry | R-052, R-055 |
| `clock_last_stone` | one stone left, pause ends the game, later Vrrooomm! does not resolve | R-053, R-066 |
| `passengers_step` | buildings of each type and waiting passengers at a node; step-out then step-in count `min(...)` | R-054, R-055 |
| `vrrooomm_example2` | rulebook Vrrooomm! Example 2: 3 passengers, 3 eligible offices, 2 buses → 2 moves | R-057..R-060 |
| `vrrooomm_eligibility` | passenger off the line, inside a building, building off the line, wrong type | R-058, R-059 |
| `vrrooomm_same_node` | passenger and an eligible building at one node | R-061 |
| `vrrooomm_canonical` | several empty buildings of the clock's type at one node: `DELIVER_TO` offers only the lowest site id | R-063 |
| `vrrooomm_zero` | no eligible passenger or no building: no node | R-060 |
| `start_player_default` | no Starting Player marker: next seat | R-064 |
| `end_markers` | at round end, ≤ 1 seat with markers; also 0 | R-066 |
| `end_both` | sites full and markers at the same round end: `types` has both | R-066 |
| `tiebreak_stones` | equal final scores, different stones | R-069 |
| `tiebreak_move` | equal scores and stones, different `last_score_move` | R-069, R-070 |
| `tiebreak_shared` | tie surviving all tiebreaks (also all-zero points) | R-069, R-070 |
| `returns_win_loss` / `returns_margin` | terminal with scores 14, 8, 5 (3p), sole and shared winners | R-071, R-072 |

Rulebook examples (`lines_blue_midline`, `lines_red_cd`, `vrrooomm_example2`) are transcribed onto real board ids chosen to reproduce the geometry; each file's `description` says which rulebook example it mirrors and the mapping from the example's letters to ids.

## 9. Clarifications (Phase 1 unit 3)

1. Error codes for cases §6 does not name: an unknown `phase` string, `done` / `units_left` / `moves_left` out of range, a `pending` or `end` block that does not fit the phase → `S-PHASE`. A `waiting` key that is not an integer → `S-PASSENGERS`; an integer key outside `0..N-1` → `S-ID`; a `waiting` count < 1 → `S-PASSENGERS`. An unknown action name on the board or an unknown `expect` key → `S-SCHEMA`. An empty line when `round` ≥ 1 → `S-LINE`.
2. In `VD` the picked passenger still stands at `picked_node` and is counted in `waiting` until `DELIVER_TO` moves it.
3. `load_scenario` checks the schema of `expect` but does not evaluate it; `check_expect` (used by the harness) evaluates it after `history`.
4. `load_scenario(doc, game)` with a `game` whose parameters differ from `parameters` → `S-PLAYERS` (player count) or `S-SCHEMA` (returns).
5. Move numbers: the move number used for scoring (R-062, R-070) is `move_offset` + the number of actions applied since load. `dump_scenario` emits that value as `move_offset`. `State.move_number()` keeps OpenSpiel's meaning (actions applied to this state object).
6. `S-SETUP`: in `SB` the number of buildings equals `done`; in `L1`/`L2` there are 2n buildings and the seats' line lengths follow from `done` and turn order (`L1`: served seats have 1 street, others 0; `L2`: served seats 2, others 1). Buildings have no owner; the per-seat setup building count of the state API is derived from `done` and turn order (2 for every seat after `SB`).
7. PLAN §6.3 "markers only decrease" covers both a seat's Action markers in hand (`markers_left`, R-065) and its Bus Line markers left (25 − line length).
8. Shortfall (Phase 1 unit 7): on entering a marker, `units_left` is its full power (R-031..R-033) even when the supply, empty sites or Bus Line markers cannot cover it; units that become impossible are lost when reached (R-034), with no node.
9. A last-stone `END` (R-053) keeps the action board as it was when the game ended; a round-end `END` (R-065, R-066) has an empty board. `load(dump(s))` holds for both.
10. End data (Phase 1 unit 12): `end.types` is canonicalised to R-066's listing order (`sites_full`, `last_stone`, `markers`); the loader accepts any order and `dump_scenario` emits the canonical one. A round-end `END` has R-064 applied (`start_player` = the Starting Player owner, or start + 1), since R-066 ends the game after Starting Player resolves. For unreachable hand-built fixtures only: two seats with equal non-null `last_score_move` stay tied (shared, R-069), and a null `last_score_move` ranks after any non-null one.
