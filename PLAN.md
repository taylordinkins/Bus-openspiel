# Bus — OpenSpiel Implementation Plan

**Status:** Draft v1 · 2026-11-14 (scaffold and board data done; §4 defaults await your review)
**Repository:** `taylordinkins/Bus-openspiel` (public; default branch `master`)
**Rules source:** *Bus* Complete Edition rulebook (Jeroen Doumen & Joris Wiersinga, Splotter / Capstone Games 2024), kept locally at `third_party/rules/bus_rulebook.pdf` (git-ignored). Then §3–§4 of this file.
**Board:** the reworked map `assets/map/bus_map_reworked_v2.pdf` (unofficial, committed), digitized into `data/board.yaml`.
**Target framework:** OpenSpiel `open-spiel==2.0.2`, Python 3.12 (same pin as `kingisdead` and `lignum`)
**Prior art:** `lignum` — agents, hooks, skills, server and web shell are ported from it (D3). The earlier `BusRL` repo is not used (D4).

### Progress
- [x] Phase 0 — Scaffold, board data, rules spec, board look ✋ *you review spec, §4 defaults and the board preview*
- [x] Phase 1 — Engine: the full game, 3–5 players (2026-11-14; report `docs/phase-reports/phase-1.md`)
- [ ] Phase 2 — Server + polished board UI ✋ *playtest #1*
- [ ] Phase 3 — Bots (random, heuristic, MCTS) ✋ *playtest #2*
- [ ] Phase 4 — Trained agents *(starts with a mini-interview)*

---

## 1. Goal

A faithful, well-tested implementation of *Bus* as a **Python OpenSpiel game**, playable by humans in a **browser** with a board that is pleasant to look at and quick to use, against other humans and bots, and usable unchanged by OpenSpiel algorithms for RL work.

**Done for v1 (end of Phase 3) means:**
1. `pyspiel.load_game("python_bus", {"players": n})` works for n = 3, 4, 5 and passes `random_sim_test` with serialization.
2. Every rule in `docs/RULES_SPEC.md` has at least one test citing its rule ID; `data/board.yaml` matches the map.
3. You can play a full game in the browser on a board drawn after the reworked map, with click-to-act on streets, sites, stations and passengers, against any mix of humans and bots, and report a problem with one click.

## 2. Scope

| In v1 | Later | Out of scope |
|---|---|---|
| Complete Edition rules, 3–5 players | Trained agents (Phase 4) | 2-player or solo variants |
| Hotseat; random, heuristic, MCTS seats | | Networked multiplayer, accounts |
| Board UI after the reworked map, own icons, generic pieces | | Mobile-optimised layout |
| Report button, replay export/import, setup editor | | C++ port (only if Phase 4 measures a need) |

## 3. Decisions log

| # | Topic | Decision | Notes |
|---|---|---|---|
| D1 | Language and framework | Python game registered via `pyspiel.register_game`; pin `open-spiel==2.0.2`, Python 3.12 | Proven twice; `docs/OPENSPIEL_NOTES.md` copied from lignum |
| D2 | Phase flow (2026-11-14) | Four phases instead of lignum's eight: spec and scaffold together, the whole engine in one phase, server and the real board UI together (no text-board stage), then bots. Opus audits only at phase close and at spot checks (§8) | User: get the implementation out faster |
| D3 | Server and web | Port lignum's game-agnostic `server/` (app, sessions, bots, reports, schemas, adapter base) and `web/src/shell/` in Phase 2; the Bus board plugin replaces the text board | The shell lives in the private `lignum` repo (the public `lignum-openspiel` lacks `web/src/shell/`) |
| D4 | BusRL (2026-11-14) | Not used, not even its board file; the map was digitized again from the reworked map | User decision |
| D5 | Information and chance | `PERFECT_INFORMATION`, `chance_mode=DETERMINISTIC`: Bus has no randomness. The starting player is a lobby choice (seat 0 starts, ⚑) | Simplifies MCTS and RL; no resamplers |
| D6 | Player counts (2026-11-14) | 3, 4 and 5 players, as published | User decision |
| D7 | Returns | Parameter `returns`: `win_loss` (default) — sole winner +1, others −1/(n−1); a k-way tie that survives every tiebreak shares the win, (n−k)/(k·(n−1)) each; `margin` — own final score minus the mean of the others, divided by 10 and clipped to [−1, 1] (GENERAL_SUM) | As lignum D7; policies may train on `margin`, are always evaluated on `win_loss` |
| D8 | Third-party material (2026-11-14) | The rulebook stays local in git-ignored `third_party/rules/`. The reworked map is unofficial and is committed under `assets/map/` | User decision. Rulebook text is paraphrased in the spec, never pasted |
| D9 | Bots and UI shape the engine | `action_meta(state, id)` gives every legal action its kind and board targets (street, node, site, end, building type) so the UI can be click-to-act and bots can featurize; observation tensor designed for RL in Phase 1 | The user's priorities: good UI now, agents later |
| D10 | Board data | `data/board.yaml` is the only source of topology (36 nodes, 70 streets, 47 sites A–D, 2 stations, 4 roundabouts) with map-pixel coordinates the UI reuses; `bus/board.py` loads and validates it | §4 C1 |
| D11 | Persistence | Persist action histories (plus scenario JSON for editor-built states); never accept `serialize()` strings from clients | As lignum D18 |
| D12 | Spec lock | `docs/RULES_SPEC.md` Status DRAFT → APPROVED → AMENDING; a hook blocks edits while APPROVED; only you change the Status line | As lignum D16 |
| D13 | Tests ahead of code | `@pytest.mark.pending("R-0xx")` (non-strict xfail); `spec-test-writer` can't read `bus/` source (hook) | As lignum D17 |
| D14 | Model use | Sonnet workhorses, Haiku search and sims, Opus audits at phase close and spot checks (§8); main session defaults to Sonnet | As lignum D15 |
| D15 | Board art (2026-11-14) | The UI uses the user's art `assets/map/bus_map_reworked_v2.png` as the background for both the map and the placement section (action board, clock, stones, score track); only pieces are drawn on top, positioned from `data/board.yaml` (topology plus a `layout` block for action spaces, bus grid, clock, stone slots, score cells) | User decision; replaces the redrawn-board reading of §5.5 |
| D16 | Scenario format (2026-11-14) | SCENARIOS.md §9.1–9.10 settle error codes, `waiting` from round 1, `move_offset`, `units_left` = full power on entry, last-stone END keeps the board, canonical `end.types`, R-064 at round-end END | Phase 1 unit decisions (format only, no rules) |
| D17 | Spec amendment (2026-11-14) | R-049/R-066 "every site has a building"; R-046 punctuation | User-approved, Phase 1 unit 9 |
| D18 | Encoding (2026-11-14) | Absolute observation tensor with seat-to-act one-hot, counts scaled to [0, 1], points /10 capped (user); `game_result` None before terminal; current seat 0 at END | ACTIONS.md §7 |
| D19 | Bus grid (2026-11-14) | The printed 4×5 grid in the map's **Buses** box (`layout.bus_grid`) is read by **columns**: 5 columns = one per seat (P0…P4, left to right), 4 cells per column = the buses a seat can add (buses 2–5, R-043). The art and board.yaml coordinates stay unchanged. The side panel separately shows each seat's total (1–5) | User clarification during `/start-phase 2`; replaces the earlier "row per seat" reading. Columns fill top to bottom (bus 2 in the top cell); the starting bus (R-014) is not drawn on the map (user, same day) |
| D20 | Setup editor (2026-11-14) | Dropped from Phase 2. Phase 2 keeps scenario JSON load (paste, upload, named) and export in the lobby; the on-board editor is deferred, to be revisited after playtest #1 | User decision during `/start-phase 2` |
| D21 | UI/UX (2026-11-14) | `docs/UI_DESIGN.md` is the UI contract: BGA-style prompt bar, only legal targets lit, click the board itself, previews on hover, automatic steps animated and logged. Seat colours match the art's Lines stripes (P0 red, P1 green, P2 orange, P3 blue, P4 white); human seats always click, even with one legal action (bots auto-play); Lines shows legal streets only; desktop/laptop only | User choices during `/start-phase 2`; amends §5.3 (UI no longer auto-advances single choices) |

## 4. Rules clarifications

Where the rulebook is silent or ambiguous. Each becomes a numbered entry in `docs/RULES_SPEC.md`. ⚑ = my default where no source settles it — **veto any of them during the Phase 0 review.**

**C1 — The board** (`data/board.yaml`, overlay `docs/board-check.png`). The map's letters A–D replace the rulebook's numbered construction-site tiers: A is the lowest, filled first (setup builds on A), then B, C, D. Each site belongs to one intersection, the one passengers enter it from. The 4 double-ring intersections are the Central Park roundabouts; the 2 bus-icon intersections are the train stations (no sites). Two streets bend (3–6 and 28–35 are drawn via a corner); bends are drawing only. Every site assignment, including the two judgement calls at nodes 8 and 26, was confirmed by the user (2026-11-14).

**C2 — Components.** Per player: 20 Action markers, 25 Bus Line markers, 5 buses (1 placed at setup). 15 passengers in all (4 placed at setup). Time stones: 5, or 4 with 3 players. Buildings are effectively unlimited (90 discs, 47 sites). Score track has no cap (the map's 1–20 track is drawn as a lap counter in the UI).

**C3 — Setup**, in order: clock points at house; one passenger on each roundabout (they stand at the intersection and do not enter buildings until a Clock action resolves ⚑); each player places 2 buildings, one player at a time in turn order, each on any empty lowest-tier site with a type of their choice; each player places a first Bus Line marker on any street (occupied or not), in turn order; then in reverse turn order a second marker extending from either end of their first, under C5's placement rule. Seat 0 is the starting player.

**C4 — Action selection.** Starting with the start player and going in seat order, each player places one Action marker; then a second round, also mandatory; then rounds continue in which each player either places a marker or drops out for the rest of the round. A player with no markers left is skipped (and out). A multi-use action takes the earliest empty letter (a, b, …); an action whose spaces are full can't be chosen. Any action with a free space may be chosen even if it will have no effect (blocking is a real tactic) ⚑. Selection ends when every player has dropped out.

**C5 — Line expansion.** A line is an ordered path of streets with two ends; it never repeats a street of its own and may revisit intersections (loops). Extending means adding one street at one end. The engine stores the path, so loop ends are never ambiguous; in a full-line loop the player's choice of which end to extend from is the "from A to B / from B to A" choice in the rulebook. **Placement rule, per end ⚑ (from the rulebook's Red C/D example):** from a given end, if any of its candidate streets is unoccupied (no player's marker), the new marker must go on an unoccupied street; otherwise any candidate is allowed. **Exception:** if another player's line has an end at the same intersection and its last street is not your last street, you may extend onto that street even when unoccupied streets exist. The exception covers exactly that one street per use. An end with no candidate street is dead; if both are dead or you have no markers left, the remaining extensions are lost.

**C6 — Resolution and action power.** Actions resolve top to bottom: Lines, Buses, Passengers, Buildings, Clock, Vrrooomm!, Starting Player. Multi-use spaces resolve left to right as printed: for Lines and Buildings (letters printed f…a), the marker on the highest letter goes first; for Passengers and Vrrooomm! (a…f), a goes first. Dwindling power for the marker on letter index k (a = 0) is M − k for Lines, Passengers and Buildings (Lines +1 with 5 players); M is the current most buses any player has, read when the marker resolves (so a Buses action raises M for Passengers and Buildings in the same round). Power ≤ 0 does nothing. Vrrooomm! is not dwindling: each marker uses all its owner's buses. Every effect must be carried out as fully as possible.

**C7 — Buses.** Add one bus if you have one left; with 5 already, no effect.

**C8 — Passengers.** For each unit of power, place one passenger from the supply on either station (your choice each time); stop when the supply is empty.

**C9 — Buildings.** For each unit of power, choose an empty site of the lowest tier that still has empty sites, and a type (house, office, pub). When every site is full, remaining builds are lost, and the game ends at the end of the round.

**C10 — Clock.** Choose: advance (house → office → pub → house) or pause (take a time stone, −1 at game end). Taking the last stone ends the game immediately, before any further action resolves. Otherwise passengers then move: every passenger in a building steps out to its intersection; then at each intersection, waiting passengers enter empty buildings of the clock's current type there, one per building (passengers are identical, so no choice).

**C11 — Vrrooomm!** Markers resolve a → f. The owner moves up to (number of buses) passengers, one at a time, each from an intersection on their line (not inside a building) to an empty building of the clock's type whose intersection is on their line, +1 point each. A passenger already standing at the destination's intersection may be moved into it ⚑. The count is always min(buses, waiting passengers on the line, empty target buildings on the line), so only *which* passengers and buildings is a choice.

**C12 — Starting player and round end.** The Starting Player marker's owner starts next round; if nobody took it, the start passes to the next seat. Spent markers are gone. If at most one player has Action markers left, the game ends ⚑ (the rulebook says "only 1"; 0 is treated the same).

**C13 — Game end and winner.** Ends: all sites full (end of that round); last time stone taken (immediately); at most one player with markers (end of round). Final score = points − stones. Ties: most time stones; then whoever reached their points total first (the engine records the move number when each player's points last changed; earlier wins). A tie that survives (e.g. players still on 0 points) is a shared win (D7) ⚑.

## 5. Architecture

### 5.1 Repository layout

```
Bus-openspiel/
├─ PLAN.md · CLAUDE.md · README.md
├─ .claude/                   # settings, agents, skills, hooks, rules (ported from lignum)
├─ .github/workflows/ci.yml   # python job now; web + e2e jobs join in Phase 2
├─ third_party/rules/         # rulebook PDF, local only (git-ignored)
├─ assets/map/                # the reworked map (PDF + 1991 px PNG)
├─ data/board.yaml            # board topology + map coordinates (D10)
├─ docs/  (RULES_SPEC.md, ACTIONS.md, SCENARIOS.md, OPENSPIEL_NOTES.md, board-check.png, phase-reports/)
├─ bus/                       # board, constants, rules, actions, game, observer, scenario, bots/
├─ server/                    # Phase 2: ported FastAPI app + bus adapter
├─ web/                       # Phase 2: ported React/TS shell + games/bus/ board plugin
├─ tests/  (data/, rules/, scenarios/, regressions/, playthroughs/, support/)
├─ tools/  (render_board, play_text, replay, sim_stats, tournament)
└─ playtest/reports/
```

### 5.2 OpenSpiel game definition

`short_name="python_bus"`, `dynamics=SEQUENTIAL`, `chance_mode=DETERMINISTIC`, `information=PERFECT_INFORMATION`, `utility=ZERO_SUM` (`win_loss`) or `GENERAL_SUM` (`margin`), `reward_model=TERMINAL`, players 3–5. Parameters: `players` (4), `returns` ("win_loss"). Observation via `make_py_observer`: one tensor plane set per street (line occupancy per seat, own-line ends), per node (waiting passengers, line membership), per site (building type, occupied), plus clock, stones, markers left, buses, scores, action-board occupancy and the pending sub-decision. `max_game_length` bounded from markers (20 per player) and per-marker sub-decisions.

### 5.3 Round machine

```
SETUP: buildings (2 each, turn order) ─► first line marker (turn order) ─► second marker (reverse order)
ROUND: SELECT (mandatory ×2, then place-or-drop until all dropped)
       ─► RESOLVE Lines ─► Buses ─► Passengers ─► Buildings ─► Clock ─► Vrrooomm! ─► Starting Player
       ─► end checks ─► next ROUND or TERMINAL
```

Resolution sub-decisions are made by the marker's owner as they come up; steps with exactly one legal choice auto-advance for bots, while human seats still click them (D21).

### 5.4 Actions (sketch; fixed in `docs/ACTIONS.md` during Phase 0, frozen when Phase 1 closes)

| Kind | IDs (approx.) |
|---|---|
| `SELECT action a` (7) and `DROP_OUT` | 8 |
| `EXTEND(end, street)` — end 0/1 × 70 streets (also the setup line markers) | 140 |
| `BUILD(site, type)` — 47 × 3 (also setup buildings) | 141 |
| `PASSENGER_TO(station)` | 2 |
| `CLOCK_ADVANCE`, `CLOCK_PAUSE` | 2 |
| `PICK_PASSENGER(node)` then `DELIVER_TO(site)` | 36 + 47 |

About 380 decision ids, no chance ids.

### 5.5 Server and web

Ported (D3): the action history is the source of truth; Back, Rewind (dev), Report, replay export/import; the action-list fallback stays as a debug panel. The Bus plugin shows the user's map art as the background (D15) and draws pieces over it as SVG from `data/board.yaml`: generic pieces for line markers (coloured offset strokes so shared streets show every line), buses, passengers, time stones and the clock. Click targets: line ends then streets, sites then a type picker, stations, passengers then buildings. The action board, clock, stones, scores and markers-left sit beside the map as on the printed board.

### 5.6 Bots

Random; heuristic (greedy over a few hand features: deliverable passengers, buildings reachable on own line, line contestation); Python MCTS (`open_spiel.python.algorithms.mcts.MCTSBot`, never the C++ bot, see notes) with heuristic or random rollouts. Trained agents are Phase 4.

## 6. Verification strategy

Cheapest first:

1. **Board data tests** (`tests/data/`): counts, connectivity, site tiers; plus your review of `docs/board-check.png`.
2. **Rule tests** named `test_R###_…`, written from the spec before the code, on small scenario fixtures.
3. **Invariants after every action** in random simulations: passengers on board + supply = 15; at most one passenger per building; each line is a valid trail (no repeated own street, contiguous); line markers + supply = 25 per player; buses 1–5; markers only decrease; points never decrease; sites filled tier by tier; stones on clock + taken = 5 (or 4); non-terminal states have a legal action; returns sum to 0 for `win_loss`.
4. **OpenSpiel conformance**: `random_sim_test` with serialization, every player count.
5. **Golden playthroughs**: fixed-seed files; any diff explained in the commit.
6. **Independent audits**: `rules-auditor` (spec vs rulebook, code vs spec), `code-reviewer` (diff).
7. **Your playtests** at each ✋, with the Report button.

## 7. Phases

Each phase starts with `/start-phase N` (task list for your approval) and ends with `/close-phase` (evidence report, PR). ✋ = stop for you.

**Phase 0 — Scaffold, board data, rules spec, board look ✋**
- [x] 1. Scaffold: `uv` project, pins, ruff, pyright, pytest, CI, `CLAUDE.md`, `.claude/` ported from lignum (agents, skills, hooks, rules), `docs/OPENSPIEL_NOTES.md`, hook tests green (2026-11-14)
- [x] 2. Board digitized: `data/board.yaml`, `bus/board.py`, `tests/data/test_board.py`, overlay `docs/board-check.png` (2026-11-14)
- [x] 3. `docs/RULES_SPEC.md` from the rulebook and §4 — `implementer` (2026-11-14; R-001–R-073)
- [x] 4. `docs/ACTIONS.md` and `docs/SCENARIOS.md` — `implementer` (2026-11-14; 376 action ids)
- [x] 5. Spec audit vs rulebook (user runs it from the CLI before implementation), board data audit vs map — `rules-auditor`, then fixes (2026-11-14; spec FAIL→fixed in 4ed05b6, board PASS)
- [x] 6. Board preview: `tools/render_board.py` draws `docs/board-preview.svg` (map layout, icons, sample lines, buses, passengers, action board) so the look is settled before Phase 2 — `implementer` (2026-11-14)
- [x] 7. ✋ You review the spec, §4 defaults, §9 questions and the preview; you set the spec Status to APPROVED (2026-11-14; approved by the user)
- *Exit:* spec APPROVED; board data confirmed; preview look agreed.

**Phase 1 — Engine (the whole game, 3–5 players)**
- Constants, actions, setup, selection, all seven resolutions, end and scoring, both `returns`; observer with RL-ready tensor; `action_meta`; scenario loader; invariant checker; `tools/play_text.py`, `tools/replay.py`, `tools/sim_stats.py`; golden playthroughs for 3 seeds.
- [x] 1. Constants, action IDs, board fixes (R-001–R-008; ACTIONS.md §2, §4; backlog: Layout hashable, bounds, negative parser tests, supplies to constants) — `spec-test-writer` → `implementer` (2026-11-14)
- [x] 2. State model and game registration (R-003, R-009, R-014, R-073; §5.2 params) — `spec-test-writer` → `implementer` (2026-11-14; state-aware EXTEND wording moves to unit 5)
- [x] 3. Scenario loader and invariant checker (SCENARIOS.md, `S-*` codes, §6.3; backlog: `move_offset`, round-1 `waiting`, `S-STONES`) — `spec-test-writer` → `implementer` (2026-11-14; SCENARIOS.md §9 clarifications)
- [x] 4. Line geometry ⚑ (R-035–R-039, R-041) — `spec-test-writer` → `implementer` → `rules-auditor` (2026-11-14; audit PASS)
- [x] 5. Setup phases (R-015–R-019, R-021; plus state-aware EXTEND `action_to_string`) — `spec-test-writer` → `implementer` (2026-11-14)
- [x] 6. Action selection (R-020, R-022–R-028) — `spec-test-writer` → `implementer` (2026-11-14)
- [x] 7. Resolution framework and Buses ⚑ (R-029–R-034, R-043, R-044; owns `start_resolution` — replace unit 6's placeholder when only Buses/Starting Player hold markers) — `spec-test-writer` → `implementer` → `rules-auditor` (2026-11-14; audit PASS after R-027 hand-off fix)
- [x] 8. Lines resolution ⚑ (R-040, R-042; must include a state-level O-2 test: unit k+1's legal set reflects the street placed in unit k, per unit 4 audit) — `spec-test-writer` → `implementer` → `rules-auditor` (2026-11-14; audit PASS)
- [x] 9. Passengers and Buildings (R-045–R-049; R-049/R-066 wording amendment first, needs user OK) — `spec-test-writer` → `implementer` (2026-11-14; amendment applied)
- [x] 10. Clock and passenger movement ⚑ (R-050–R-055) — `spec-test-writer` → `implementer` → `rules-auditor` (2026-11-14; audit PASS)
- [x] 11. Vrrooomm! ⚑ (R-056–R-063; add a reachable Vrrooomm!-first fixture with moves > 0 — `resolve_vp` may be unreachable) — `spec-test-writer` → `implementer` → `rules-auditor` (2026-11-14; audit PASS)
- [x] 12. Round end, game end, returns ⚑ (R-064–R-072; fill the `end_round` stub: R-065 bookkeeping, R-066 checks incl. no seat with markers) — `spec-test-writer` → `implementer` → `rules-auditor` (2026-11-14; audit PASS)
- [x] 13. Observer, `action_meta`, `forced_action`, `max_game_length` (§5.2; ACTIONS.md §3, §5, §6; plus ACTIONS §3.12 `game_result(state)` and winners in the observer string; canonical `current` at END so `str(load(dump(s))) == str(s)`, per unit 12 audit) — `spec-test-writer` → `implementer` (2026-11-14; tensor 1178/1474/1770 floats, ~100 µs; observed max length 177/197/257 vs 702/936/1170)
- [x] 14. Conformance and random sims (`random_sim_test` 3/4/5, 10,000 games slow) — `implementer`, `sim-runner` (2026-11-14; 10,000 games, 0 violations, 6m40s; margin utility_sum fix)
- [x] 15. `tools/play_text.py`, `tools/replay.py` — `implementer` (2026-11-14)
- [x] 16. `tools/sim_stats.py`, golden playthroughs for 3 seeds — `implementer`, `sim-runner` (2026-11-14)
- *Exit:* rule tests green; `random_sim_test` passes for 3, 4, 5 players; 10,000 random games with zero invariant violations; `sim_stats` report (end-type mix, rounds, scores, markers used); `rules-auditor` + `code-reviewer` pass.

**Phase 2 — Server + polished board UI ✋ playtest #1**
- Port server and shell (D3); Bus adapter (views, labels, `action_meta`); SVG board plugin per §5.5 and the agreed preview; scenario JSON load/export (editor deferred, D20); random seats; Playwright test that finishes a game; CI web + e2e jobs.
- **Model note (user, 2026-11-15):** for the rest of Phase 2 (units 9–11) the `implementer` runs as **Opus 5.5, medium effort** (`model: opus`, `effort: medium` on the Agent call), overriding the Sonnet default in CLAUDE.md. The UI look from units 6–8b is approved; later units refine behaviour, not the visual direction.
- [x] 1. Port the server shell (lignum `server/`: app, sessions, random bots, reports, schemas, `adapters/base.py`; toy adapter and game-agnostic server tests; fastapi/uvicorn/httpx deps) — `implementer` (2026-11-14; 90 server tests)
- [x] 2. Bus adapter: `view`, labels, `prompt`, `targets` (UI_DESIGN §6; ACTIONS §5) — done: every legal action in random 3/4/5-player games has a label and a target; view matches fixtures — `spec-test-writer` → `implementer` (2026-11-14; contract UI_DESIGN §6.5)
- [x] 3. Bus adapter: per-action automatic-step `events`, board-geometry endpoint, strict scenario load (backlog: reject non-terminal no-legal-action states), random seats — done: all-random game to END via the API; events match state diffs — `implementer` (2026-11-14; 1560 tests; contract UI_DESIGN §6.6)
- [x] 4. Bus grid by seat column (D19) and seat palette from the art's Lines stripes (D21) in `board-palette.md` and `tools/render_board.py` — done: data tests green, preview regenerated and reviewed — `implementer` (2026-11-14; 1562 tests; `Layout.bus_cell`)
- [x] 5. Port the web shell (`web/`, `src/shell/`, OpenAPI export, generated types; Bus on the action-list fallback) — done: typecheck, test, build, check:api green — `implementer` (2026-11-14; 53 vitest, 1562 pytest)
- [x] 6. Board renderer: art, pieces (UI_DESIGN §3), zoom/pan, hover info — done: render tests + screenshot review — `implementer` (2026-11-14; 67 vitest, 1562 pytest; P4 white pieces not yet seen on screen; view nests at `game.view.view`, UI_DESIGN §6.5 to be corrected; hit areas below the 24 px §8 target, tune in unit 8)
- [x] 7. Panels: prompt bar, seat panels, hotseat turn frame, log with hover highlight, END results — done: render tests for SB, SEL, each resolution phase, END — `implementer` (2026-11-14; 95 vitest, 1562 pytest; side column via new plugin hooks `renderSide`/`renderLog`; UI_DESIGN §6.5 nesting corrected)
- [x] 8. Interaction layer (UI_DESIGN §4): glow, hover ghosts, type popover, VP→VD, bar buttons, last-stone confirm, Back — done: every legal action at every phase code reachable by clicking — `spec-test-writer` → `implementer` (2026-11-14; 400 vitest, 1569 pytest; contract UI_DESIGN §6.7; fixed a mount-effect race that wiped an early popover click)
- [x] 8b. Visual theme (2026-11-14; Opus 5.5 medium; 400 vitest; tokens in web/src/theme/, UI_DESIGN §3a; weaknesses logged below in Backlog) (user request 2026-11-14): restyle the whole app, shell included (lobby, game screen, tabs, dialogs, curtain, action list, panels, log, buttons), to match the board's palette (`docs/board-palette.md`), lettering and style, so it is not the plain Lignum look. Direction (user, 2026-11-14): follow the map (cream paper, its inks, condensed bold headings); the quality bar is a publishable, presentable app, not basic wrapper styling — considered type scale, spacing, depth, hover/focus/disabled states, empty and loading states, polished dialogs and buttons — done: theme tokens in one CSS file, UI_DESIGN typography/theme section updated, screenshots reviewed at 1440×900 and 1280×720 for lobby, SEL, resolution, END, hotseat curtain; vitest/e2e selectors unchanged — `implementer`
- [x] 9. Animation and pacing: automatic steps from `events`, +1 float, speed and bot-pace settings — done: vitest + screenshot review — `implementer` (2026-11-15; Opus 5.5 medium; 430 vitest, 1574 pytest; server `GET /api/games/{id}?at=N` frames for bot pacing; UI_DESIGN §6.8)
- [x] 10. Scenarios in the lobby: paste, upload, named; export (editor deferred, D20) — done: API tests + vitest — `implementer` (2026-11-15; Opus 5.5 medium; 441 vitest, 1607 pytest; `GET /api/scenarios`, `data/scenarios.yaml`; UI_DESIGN §6.9)
- [x] 11. Playwright (smoke, layout, a game finished by clicking, screenshot baselines) and CI web + e2e jobs — done: `npm run e2e` green — `implementer`, `sim-runner` (2026-11-15; Opus 5.5 medium; 14 e2e + 6 opt-in screenshot tests, 445 vitest, 1627 pytest; screenshots run via `npm run e2e:shots`, advisory in CI)
- [ ] 12. ✋ Playtest #1: one hotseat game and one against random bots, with reports
- *Exit:* you play one hotseat game and one against random bots, and file reports.

**Phase 3 — Bots ✋ playtest #2**
- Heuristic and Python MCTS seats; `tools/tournament.py` with Wilson CIs; server seats with think-time limits.
- *Exit:* MCTS beats random and the heuristic beats random over a few hundred 4-player games; you play against them.

**Phase 4 — Trained agents** *(mini-interview: algorithm, compute budget, evaluation, C++ port only on measured need)*

## 8. Working agreement for agent sessions

| Role | Agent | Model | Used for |
|---|---|---|---|
| Orchestrator | main session | Sonnet (project default) | Runs the skills, delegates, talks to you |
| Workhorse | `implementer` | Sonnet | One task-list unit |
| Workhorse | `spec-test-writer` | Sonnet | Tests from the spec before code; regression tests |
| Cheap | `Explore` | Haiku | Codebase search |
| Cheap | `sim-runner` | Haiku | Full suite, slow sims, tool scripts |
| QA | `rules-auditor` | Opus, high effort | Spec vs rulebook; code and tests vs spec |
| QA | `code-reviewer` | Opus, high effort | Correctness review of the phase diff |

Opus spot audits only for units touching line-expansion validity, resolution order and action power, Vrrooomm! scoring, or game end; everything else waits for `/close-phase`.

**The loop:** `/start-phase N` → `/next-task` per unit (tests from spec → implement → suite → optional spot audit → commit, tick) → `/close-phase` (full run, both reviewers in parallel, one fix round, phase report, PR) → `/triage-report <file>` for playtest reports.

**Rules for every session:** no rule behaviour without a spec entry; ambiguity → §9 and ask. Tests cite rule IDs and are never edited to pass. Commit at each green step on the phase branch; `master` changes only by PR. Keep this file current.

## 9. Open questions

- ~~Q1 (board)~~ resolved 2026-11-14: the B/C pair below node 8 belongs to node 8 and the A/C pair below node 26 to node 26 (user confirmed).
- ~~Q2~~ resolved 2026-11-14: C5's per-end reading and C4's any-free-action confirmed (user confirmed).
- ~~Q3~~ resolved 2026-11-14: move-number tiebreak and shared surviving ties confirmed, including O-4 (user confirmed).
- ~~Q-A~~ resolved 2026-11-14: 5-player Lines +1 is added before the power ≤ 0 check (R-033).
- ~~Q-B~~ resolved 2026-11-14: Lines markers must be resolved maximally; only (end, street) choices that keep the maximum remaining placements reachable are legal (R-040, R-042).
- ~~O-1/O-2/O-3 (spec)~~ resolved 2026-11-14: building supply unlimited per type; multi-street Lines evaluated one street at a time; markers exhausted in mandatory passes → skipped (user confirmed, see RULES_SPEC.md).

### Backlog
Should-fix and minor findings, tagged with the phase that found them. `/close-phase` appends here.

- [Phase 2 unit 8b, implementer, should-fix] Theme weaknesses: hotseat "P3's turn" toast never shows (curtain remounts the turn frame); side column scrolls at END at 1280×720 (log below the fold); L1 glow low-contrast when every street is legal (add an ink keyline); long CK/SEL prompts wrap to two lines; "⚑" and "→" fall back to a system font; lobby map crop clips its top labels; game id truncated in the status line; under-900 px layouts unreviewed.
- [x] (done Phase 2 unit 4) [Phase 2, from Phase 0 preview review] Buses cell of the action board: the printed 4×5 grid is **one column per seat** (5 columns × 4 cells), not one row per seat — see D19. The preview currently fills it in reading order across seats; fix in Phase 2 unit 4. No 5th row is needed and the art is not changed.
- [Phase 0, code-reviewer, should-fix] `bus/board.py` `Layout` holds plain dicts, so the frozen cached `Board` is unhashable and mutable (`hash(load_board())` raises). Store tuples/`MappingProxyType` or exclude `layout` from hash/compare before Phase 1 caches on the board.
- [Phase 0, code-reviewer, should-fix] `tests/data/test_board.py`: frame/letter/count tests only re-check data `load_board()` already validates; add negative tests feeding bad raw dicts to the parser (out-of-frame point, letters a–e, 19 bus cells, missing clock segment) expecting `ValueError`.
- [Phase 0, code-reviewer, minor] `bus/board.py` bounds check accepts 1991 (`<=` vs `<` FRAME_SIZE); tests repeat the literal 1991. `_parse_layout` accepts unknown action keys, non-positive clock hand/radius, 3-element or float points.
- [Phase 0, code-reviewer, minor] `tools/render_board.py`: clock order re-listed instead of `CLOCK_SEGMENTS`; supplies 25/20 are literals (move to `bus/constants.py` in Phase 1); scores >20 clamped silently (preview only); `BG_HREF` is docs/-relative, so a non-`--embed` render elsewhere has a broken background.
- [Phase 0, code-reviewer, minor] `tests/data/test_board.py:62` names node pairs (3,6) and (28,35) — the only ids outside board.yaml; acceptable for a data test.
- [x] *(done Phase 1 unit 3: `move_offset`)* [Phase 0, rules-auditor, should-fix] ACTIONS.md:26 / SCENARIOS.md:53 (R-062, R-070): move numbers are history length, but scenarios can set `last_score_move` before load; define a scenario move-number origin (`move_offset`, or default to the max `last_score_move`) so the tiebreak stays correct for editor games and fixtures.
- [x] *(done Phase 1 unit 9)* [Phase 0, rules-auditor, should-fix, needs AMENDING] RULES_SPEC.md R-049, R-066: "every site is occupied" means every site has a building, but "occupied" means passenger-inside elsewhere (R-062, SCENARIOS `occupied`); reword to "every site has a building" at the next amendment.
- [x] *(done Phase 1 unit 3)* [Phase 0, rules-auditor, minor] SCENARIOS.md: `waiting` defaults to `{}` outside setup, so the round-1 shorthand drops the 4 roundabout passengers (R-015) — document or require it; S-STONES should reject `stones_on_clock` 0 unless phase `END`.
- [x] *(done Phase 1 unit 9)* [Phase 0, rules-auditor, minor, needs AMENDING] RULES_SPEC.md R-046: missing full stop after "(R-058)".
- [x] *(done Phase 1 unit 14: explicit `EXPECTED_UNREACHABLE = {resolve_vd}`)* [Phase 1 unit 11, rules-auditor, minor] `tests/scenarios/resolve_vp.json` / `resolve_vd.json` are unreachable (forced count 0 for the owner; VD has no legal action). `tests/test_invariants.py:106-111` skips such fixtures as "phase not implemented yet", which now hides them. Replace the generic skip with an explicit expected-unreachable list (or make the fixtures reachable) once all phases play.
- [Phase 1 unit 12, rules-auditor, minor] END fixtures (`tests/scenarios/end_*.json`, `end/*.json`) claim `end.types` their state doesn't support (e.g. `sites_full` with empty sites; `end_last_stone` with an empty board). Either validate end types against the state in the loader (new code) or list it under SCENARIOS §6 "Not checked" and fix the fixtures.
- [Phase 1 unit 12, rules-auditor, minor] tests/rules/test_game_end.py: stale module docstring (lines 3, 20-21), wrong docstring on the R-072 clip case (764-771, exact −1.0 boundary untested), dead `apply_action` at 797-805.
- [Phase 1, code-reviewer, should-fix] `state.clone()` deep-copies the immutable Core (~94 µs, the main MCTS cost); give `Core`/`Seat` `__deepcopy__`/`__copy__` returning self (~33 µs). Before Phase 3.
- [Phase 1, code-reviewer, should-fix] R-042 lookahead recomputed several times per step (legal_actions, apply validation, next unit, `_enter`); memoise `_ln_legal` on the hashable core and hoist other seats' ends/occupied set out of `_best_continuation`. Before Phase 3.
- [Phase 1, code-reviewer, minor] `bus/observer.py:65` `_nodes_of` unbounded `@cache` → `lru_cache(maxsize=…)`.
- [x] *(done Phase 2 unit 3: `require_legal`)* [Phase 1, code-reviewer, minor] Loader accepts non-terminal resolution nodes with no legal action (SCENARIOS §6 allows for fixtures); reject in the server/editor path (or under `strict`) before Phase 2 accepts scenarios.
- [Phase 1, code-reviewer, minor] `tools/play_text.py:149` doesn't catch `ScenarioError` (traceback on `--players 7`); `bus/rules.py:186` apply at END says "not implemented" instead of "game is over"; `tools/sim_stats.py:31-32` docstring return order.
- [Phase 1, code-reviewer, minor] `tests/test_conformance.py:66-72` aliasing check can't fail (advance only the original); `tests/support/invariants.py:115` scoring must strictly advance `last_score_move` (`<=`).
- [Phase 1, rules-auditor, minor] Docstrings cite wrong rule IDs (`bus/rules.py:475,486` R-046→R-047/R-048; `:558` R-058→R-059); stale module docstring `rules.py:3-5`, `:17`.
- [Phase 1, rules-auditor, minor] No test tagged R-003, R-009, R-011, R-013 (all exercised under other IDs); no direct test that `returns` never changes legal actions (R-002).
- [Phase 1, orchestrator, minor] `tools/replay.py` rejects golden playthrough files (`tests/playthroughs/*.json`, S-SCHEMA); accept them (parameters + history) or document the two formats.
- [Phase 1, orchestrator, minor] `play_text` prints the state as one ~1,000-character line; a multi-line human view would help terminal play.
- [Phase 0, both reviewers] `layout.bus_grid` validation requires exactly 20 cells; that is correct (5 seat columns × 4 added buses, D19) — keep it, and add the column-per-seat accessor in Phase 2 unit 4.
- [x] *(done Phase 2: curtain kept by user decision, UI_DESIGN §7 rewritten; it now waits for the mover's animation)* [Phase 2 unit 9, implementer, should-fix] Hotseat curtain contradicts UI_DESIGN §7 ("hotseat has no curtain"): the shell still shows a hand-off curtain right after each move, hiding the mover's animation. Decide with the user which is right (change the doc or the shell), then fix.
- [Phase 2 unit 9, implementer, minor] vitest prints React `act()` warnings (`shell.test.tsx` 12, `pacing.test.tsx` 2, `useGame.test.tsx` auto-advance tests, ~170 in total); source of most not traced and not checked against pre-unit-9. Tests pass; wrap the state updates in `act`/`waitFor` or silence deliberately.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Line-expansion rules (loops, exception, per-end rule) are subtle | Dedicated rule tests with drawn scenarios; Opus spot audit; UI shows legal streets only |
| Sub-decision chains (Vrrooomm!, Buildings) bloat search trees | Canonical orderings; forced single-choice steps auto-advance |
| Board digitization error | Data tests, overlay review, auditor check against the map |
| Python speed for MCTS/RL | Fine for play; `search_clone` pattern from lignum; C++ only on measured need |
