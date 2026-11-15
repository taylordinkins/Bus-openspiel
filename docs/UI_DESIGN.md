# Bus — UI/UX design (Phase 2)

**Status:** AGREED · 2026-11-14 · user choices in §9 (D21)
**Scope:** the web board plugin (`web/src/games/bus/`) and the adapter data it needs. Rules behaviour is never decided here: the engine's legal actions and `action_meta` (ACTIONS.md §5) are the only source of what may be clicked.

## 1. Principles (Board Game Arena style)

1. **One obvious prompt.** A status bar at the top always says who acts and what to do, in one sentence with the seat's colour: *"P2 — Lines: choose a street to extend your line (2 of 4)"*. Buttons for non-board choices sit at the right of the same bar.
2. **Only legal targets light up.** Legal pieces/places glow and pulse gently and show a pointer cursor; everything else is inert. No click is ever rejected by the server in normal play.
3. **Click the thing itself.** Streets, sites, stations, passengers, the clock and the action spaces on the art are the controls. The action list exists only as a hidden debug panel.
4. **Fewest clicks.** One click when a click determines one action id; a second step (popover, or a second target) only when one place maps to several ids.
5. **Preview before commit.** Hover shows a ghost of the result (translucent line marker, building, passenger path, clock hand). Confirmation dialogs only for game-ending choices.
6. **Nothing changes silently.** Automatic steps (Buses, passenger step-out/in, skipped seats, power-0 markers, round clean-up, start player) are animated in order and written to the log.
7. **Undo is cheap.** Bus has no hidden information or chance, so Back is always safe; it undoes your last decision (in hotseat, back to the previous human decision).

## 2. Screen layout (desktop, ≥ 1280 × 720)

```
┌──────────────────────────────────────────────────────────────┬───────────────────┐
│ STATUS BAR  ● P2 — Lines: choose a street (2 of 4)   [Back]  │                   │
├──────────────────────────────────────────────────────────────┤  PLAYER PANELS    │
│                                                              │  ● P0  12 pts ⚑   │
│               MAP ART (fit to height, zoomable)              │    ▮▮▮▯▯ buses    │
│        streets · sites · stations · passengers · lines       │    action 9 · line│
│                                                              │    markers 14     │
│ ─────────────────── placement section (on the art) ───────── │  ● P1 …           │
│  1 Lines │ 2 Buses (grid) │ 3 Passengers │ stones            │                   │
│  4 Buildings │ 5 Clock │ 6 Vrrooomm! │ 7 Start player        │  ROUND 3 · pass 4 │
│                                                              ├───────────────────┤
│                                                              │  LOG (hover =     │
│                                                              │  highlight)       │
└──────────────────────────────────────────────────────────────┴───────────────────┘
```

- **Map:** the user's art is the background (D15) scaled to the viewport height, with zoom (wheel, pinch, +/− buttons, "fit") up to 2.5× and pan by drag. The placement section stays part of the art: action spaces, bus grid, clock and stones are clicked where they are printed.
- **Player panels:** seat colour band, name/kind (human, random bot), score, ⚑ start player, buses as 5 icons (owned filled), Action markers left, Bus Line markers left, "dropped out" state for the current round. The acting seat's panel is raised and outlined. Hovering a panel highlights that seat's line on the map.
- **Log:** one line per decision plus indented lines for automatic steps; hovering an entry highlights the pieces it touched. Round headers separate rounds.
- **Turn frame:** in hotseat, the whole board gets a thin frame in the acting seat's colour, and a "P3's turn" toast shows when the human seat changes.

## 3. Pieces and look

Kept from the agreed preview (`docs/board-preview.svg`), drawn as SVG over the art from `data/board.yaml` coordinates:

| Piece | Look |
|---|---|
| Line marker | Rounded stroke in the seat colour with a dark outline, offset per seat on shared streets so every line shows; line ends get a small round cap ("handle") |
| Building | Disc on the site with our own house/office/pub icon; occupied = a passenger figure inside |
| Waiting passenger | Small figure; ≥ 4 at one node shows one figure with a count badge |
| Bus | Bus glyph in seat colour; on the map's Buses grid per D19 (column per seat, top to bottom, starting bus not drawn) |
| Action marker | Seat-coloured token on the printed letter space |
| Clock hand / stones | Hand pointing at the current type; stones on the printed slots, taken stones move to the taker's panel |
| Score | Seat-coloured disc on the art's score track (`layout.score_track`), stacked when seats tie; the number also in the panel |

Motion: placements ease in over ~250 ms; passengers move along the street path; a "+1" floats from a delivery to the scorer's panel. A settings toggle sets animation speed (normal, fast, off) and bot pacing.

Seat colours match the art's Lines stripes, in stripe order: P0 red, P1 green, P2 orange, P3 blue, P4 white (§9 Q1). Hues are taken from the art; white pieces and red pieces get a dark outline so they read against the cream paper and the art's red frames, and seat identity never relies on colour alone (§8).

Typography: one bold condensed sans for headings to match the art's lettering, a plain sans for text; the board palette tokens in `docs/board-palette.md`. Light theme only (the art is cream); dark mode is out of scope.

## 4. Interaction per decision point

`action_meta` gives each legal id its targets; the UI groups ids by target. "Glow" = legal target highlight.

| Phase | Prompt (status bar) | Glow | Click flow |
|---|---|---|---|
| `SB` setup buildings | "Place a building on an A site (1 of 2)" | empty sites of the open tier | site → popover with 3 icons (house/office/pub) at the site → done |
| `L1` first line marker | "Place your first Bus Line marker on any street" | every street (hover shows ghost in seat colour) | street → done |
| `L2` second marker | "Extend your line from either end" | legal streets at both ends | street → done |
| `SEL` selection | "Place an Action marker (pass 3, 9 left)" + **[Drop out]** from pass 3 | each action box with a free space; the letter the marker would take (`space`) | action box → done; Drop out is a button (and the only option when the board is full) |
| `LN` Lines | "Lines: extend your line (2 of 4)" | own line-end handles + legal streets | street → done; if one street is legal from two ends (loop, R-041) → choose the end by clicking a handle |
| `PX` Passengers | "Passengers: add a passenger to a station (1 of 2) · 9 in supply" | both stations | station → done |
| `BD` Buildings | "Buildings: build on a B site (1 of 3)" | empty sites of the lowest open tier | site → type popover → done |
| `CK` Clock | "Clock: advance to Pub or take a time stone" + **[Advance → Pub]** **[Take stone (2 left)]** | the clock (next segment) and the stones | click either; the last stone opens a confirm: "Taking the last stone ends the game now" |
| `VP` pick | "Vrrooomm!: pick a passenger on your line (1 of 3) · deliver to Offices" | waiting passengers on own line; own line drawn thicker | passenger node → goes to VD |
| `VD` deliver | "Deliver the passenger to an Office on your line" | legal sites, with a dashed path from the picked node | site → +1 animation; back to VP until done |
| `END` | results panel | — | final scores, winners, tiebreak explanation (R-069/R-070), end reason (R-066), [Export] [New game] |

**Single legal action:** for a human seat it is not auto-played: the one option glows and the player clicks it (§9 Q2). Bots still play it at once.

Only legal streets are highlighted in `LN`; streets excluded by maximal resolution (R-040/R-042) are simply not lit (§9 Q3).

## 5. Hover information

- Node: waiting passengers, buildings and occupants, which lines pass.
- Site: tier, type, occupied.
- Action box: the markers on it, in resolution order, and the power each will have (M − k, R-031) when computable from the view.
- Seat panel: highlights the seat's line, buses and markers.

## 6. Adapter view contract (`server/adapters/bus.py`)

The adapter's view for Bus (at `GameView.view.view`, see §6.5) is a dict with four keys. Seats are 0-based everywhere.

### 6.1 `state` — the position as scenario JSON
A dict in the format of `docs/SCENARIOS.md` (`format: 1`, `parameters`, `round`, `phase`, `current_player`, `start_player`, `clock`, `stones_on_clock`, `seats`, `buildings`, `waiting`, `supply`, `action_board`, `pending`, `end` at END, `move_offset`), with no `history`/`expect`. **Contract:** loading `view.state` with the scenario loader gives a state whose observer string equals the viewed state's (the round trip of D11 and unit 10). Canonical forms: `seats` as a list; `buildings` sorted by site; `waiting` with string node keys and counts ≥ 1; `action_board` keyed by the action names `lines`, `buses`, `passengers`, `buildings`, `clock`, `vrrooomm`, `starting_player` (as in `layout.actions` of `data/board.yaml`) with every list at full length (6 or 1) and `null` for free spaces; `move_offset` = the scenario move number of the viewed position (history length plus the original offset).

### 6.2 `derived` — facts the UI would otherwise recompute
- `seats[i]`: `line_markers_left` (25 − line length), `line_nodes` (ordered node ids of the line; `[]` when no line), `ends` (`[node, node]`, or `null` with no line), `dropped` (bool, this round's SEL).
- `street_lines`: map street id (string) → list of seats whose line uses it, ascending seat order (drives the per-seat offsets on shared streets).
- `open_tier`: letter of the lowest tier with an empty site, or `null` when all sites are built (R-010).
- `resolving`: `null` outside resolution, else `{action, letter, owner, power}` — `action` a name as above, `letter` 0-based, `power` the marker's power under R-031/R-033 (`null` for actions without dwindling power).
- `result`: `null` before END; at END `game_result(state)` (`end_types`, `winners`, `final_scores`, `returns`).

### 6.3 `prompt` — what the status bar says
`{phase, actor, units_left, power, context}`: `phase` the code; `actor` = current seat (`null` at END); `units_left` from `pending` (`units_left`, or `moves_left` in VP/VD, else `null`); `power` as in `resolving`; `context` phase-specific: SB/L1/L2 `{done}`; SEL `{pass, mandatory (pass < 2), markers_left}`; PX `{supply}`; BD `{open_tier}`; CK `{clock, next_clock, stones_on_clock}`; VP/VD `{clock, moves_left, picked_node}` (picked_node `null` in VP); END `{}`.

### 6.4 `targets` — legal ids grouped by what is clicked
Built only from the legal actions and `action_meta`:
- `streets`: street id (string) → list of `extend` ids on it (two ids only for the R-041 loop case, distinguished by `end`).
- `sites`: site id (string) → list of `build` ids (one per building type) or one `deliver_to` id.
- `stations`: node id (string) → `passenger_to` id.
- `nodes`: node id (string) → `pick_passenger` id.
- `actions`: action name → `select` id.
- `clock`: `{advance: id|null, pause: id|null}`.
- `buttons`: list of `{id, kind}` for ids with no board target (`drop_out`).
**Contract:** every legal id appears exactly once across the groups, and no illegal id appears. At END every group is empty.

`GameView.legal_actions[i]` carries `id`, `label` (ACTIONS §4 string, seats as `P0`…), `kind` and `params` (the rest of `action_meta`, ACTIONS §5). History entries use the label computed in the state the action was played in.

### 6.5 Clarifications (unit 2 test hand-off)
1. `LegalAction.params` is a dict (the server schema changes from `list[int]`). The adapter's `action_meta` returns `{"label", "kind", "params"}` with the ACTIONS §5 keys other than `id`, `kind`, `player`, `label` inside `params`.
2. The four keys sit one level down, in `GameView.view.view` (`GameView.view` itself is the server's per-seat envelope: `seat`, `current_player`, `is_terminal`, `legal_actions`, `returns`, `winners`, `view`). The web plugin reads them through `busInner()` in `model.ts`, which also accepts the keys directly in `view` for hand-made fixtures.
3. Single-id groups (`stations`, `nodes`, `actions`) map to an int; `streets` and `sites` map to lists; `clock.advance`/`clock.pause` are int or null.
4. `targets.actions` and `resolving.action` use the lowercase board names (`lines` … `starting_player`).
5. `context` for LN is `{}`.
6. SEL `markers_left` is the actor's Action markers in hand (int).
7. `power` is the nominal power of the marker (R-031/R-033; M read now, R-032), not the units still to resolve; hand-built fixtures may have `units_left` > `power`.
8. `derived.seats[i].dropped` is false outside SEL.
9. `move_offset` follows SCENARIOS §2 (the loaded offset plus the history length since load).
10. VD `picked_node` is a node id; `state.end` exists only at END as `{"types": [...]}`.
11. `state.current_player` is absent at END (SCENARIOS §4: no actor), so the END round trip holds; `prompt.actor` is null there.
12. Labels follow ACTIONS §4 verbatim; "Starting Player" is an action name, so the seat-name check is "no `Player <digit>`", not "no `Player`".

### 6.6 Events, board geometry, scenarios, random seats (unit 3)
- **Events.** Each `HistoryEntry` has `events: list[Event]` (`type` plus type-specific fields): the automatic steps the engine ran as a result of that action, computed from the states before and after it (so replay, bots and Back give the same events). The latest action's events are `history[-1].events`; after bot turns, take every entry past the last one seen. Types: `scored {seat, points, node, site}` (a delivery); `passengers_moved {exited: [{site, node}], entered: [{site, node}]}` (Clock step: all occupied buildings release, then clock-type buildings take one; not emitted for the last-stone end); `selection_ended {}`; `bus_added {seat, buses}` and `marker_skipped {action, letter, owner}` (markers resolved without a decision, in resolution order; a marker that ends early with units unused is not reported); `round_ended {round, start_player}`; `round_started {round}` (also the first SEL after setup); `game_ended {end_types}`. Applying them to the previous `state` reproduces the next state's seats' buses and points, occupied buildings, waiting passengers (with the action's own effect), round and END.
- **Geometry.** `GET /api/boards/{game}` returns `{game, geometry}`; for Bus `geometry` holds `map {width, height}`, `nodes`, `streets` (with `via` bends), `sites` (tier, node, x, y) and `layout` (`actions` letter positions, `bus_grid`, `clock`, `stones`, `score_track`), all read from `data/board.yaml`. 404 for a game without geometry.
- **Scenarios.** The server create/import path loads scenarios with `require_legal`: a non-terminal position with no legal action is rejected (HTTP 422 on create, 4xx on import; `S-PHASE` message). `bus.scenario.load_scenario` keeps accepting such fixtures by default.
- **Random seats.** `seats` entries are `human` or `random`; bots play inline or on the worker as in the shell; a game of only random seats runs to END at creation.

### 6.7 Interaction contract additions (unit 8; user decision 2026-11-14)

The UI posts only ids that are in `legal_actions`, one id per click path, through `BoardUi.onAct(id)` (called exactly once, with a number). It never counts passes, sums stones or infers legality; every flag below comes from the adapter. **Implemented (unit 8).** The adapter emits the fields marked **new** (`street_ends` is always present, `{}` outside line phases; `last_stone` in every CK view; `mandatory` is computed as "Drop out is not legal", which equals `pass < 2` in every reachable position). `tools/gen_ui_fixtures.py` records `fixtures/interaction/*.json` from scenarios; only `LN.loop.json` (a synthetic two-street double-end case) and `VP.lone.json` (VP.multi trimmed to one pick) are still hand-edited. The 24 px hit-area test skips views whose only legal target is a prompt-bar button (board-full SEL Drop out) and asserts instead that it is an enabled real `<button>`. Implementation notes: LN handles are drawn and legal only when some street has two ids (loop case), so ordinary LN views light streets only; legal hit shapes are `ceil(24 * map.width / 640) + 2` map units wide, and a click or hover on overlapping legal sites, stations or handles goes to the nearest centre.

**a. LN loop (R-041).** Every `extend` legal action has `params.end` (0/1; already in the fixtures) and `params.street`. **New:** `targets.street_ends`: street id (string) → list of `{id, end, node}`, in the same order as `targets.streets[street]`, one entry per `extend` id on that street (`node` = `params.from_node`). `targets.streets` keeps its §6.4 shape. A street with one id is posted by clicking the street (fewest clicks, principle 4). A street with two ids (the loop case) is ambiguous: clicking it posts nothing and arms it, and the street's handles (`[data-testid="handle-<end>"]`, drawn on the line ends in LN, `data-end`, `data-node`) are marked `data-pending="true"`; clicking one of those handles posts that end's id. Clicking a handle first arms that end instead (streets legal from it keep `data-legal`, the others lose it) and the next street click posts. Either order posts the same id. In LN, handles whose end has at least one legal id carry `data-legal="true"` and `data-action-ids`.

**b. CK last stone (R-053).** **New:** `prompt.context.last_stone: boolean`, true exactly when the legal `clock_pause` would take the last stone and end the game (equals `params.ends_game` of that action); false otherwise; present in every CK view. The UI does not compute it from `stones_on_clock`. When true, clicking the stones target or the "Take stone" bar button opens a confirm (`role="alertdialog"`, text "Taking the last stone ends the game now", buttons **Confirm** and **Cancel**); only Confirm posts the pause id, Cancel closes it and posts nothing. When false the pause id is posted by the first click. Advance never confirms.

**c. SEL Drop out.** `targets.buttons` contains `{id: 7, kind: "drop_out"}` exactly when `drop_out` is a legal action, and no entry otherwise. `prompt.context.mandatory` is true exactly while Drop out is rule-forbidden (the two mandatory passes, R-023), false once it is permitted, and is a pure display flag (it decides the prompt wording "(mandatory pass)"); the UI shows the button solely from `targets.buttons`. When the board is full and Drop out is the only legal action it is still shown and clicked (never auto-posted). Fixtures: `mandatory === false` iff a `drop_out` legal action exists.

**d. VP → VD.** A click on a legal passenger node posts the `pick_passenger` id immediately: no popover, no confirm, no client-side "picked" state; the board keeps showing the VP view until the server's next view arrives. In that VD view `prompt.context.picked_node` (node id; also `state.pending.picked_node`) is the only source for the picked passenger: the board draws a marker `[data-testid="picked-node"]` (`data-node`) there, and hovering a legal site draws a dashed ghost path from it. Clicking a legal site posts its `deliver_to` id at once (one id per site, canonical site R-063).

**e. DOM contract of the interaction layer** (what the tests query; all inside the output of `renderBoard(game, ui)`):
- Targets are the existing hit shapes `#street-<id>`, `#site-<id>`, `#node-<id>`, plus `[data-testid="target-action-<name>"]` (name = lowercase board name, `starting_player`, `vrrooomm`; present for all 7 actions), `target-clock-advance`, `target-clock-pause` (the stones) and `handle-<end>`. Each is a `circle`, `rect` or `path`.
- A target is *legal* when it carries `data-legal="true"`, class `glow`, and `data-action-ids="<id>[,<id>]"` listing the ids it can post. An illegal target has none of the three and a click on it does nothing. Legal sets are exactly the keys/values of `view.targets`.
- When `ui.canAct` is false or `ui.onAct` is absent (bot seat, request in flight, not the viewed seat) nothing is legal: no `data-legal`, no glow, bar action buttons absent or disabled, clicks post nothing.
- A site with several `build` ids opens the type popover on click: `[data-testid="type-popover"]` (`role="menu"`) with one `role="menuitem"` per id named by its building type (house, office, pub) and `data-action-id`; choosing posts that id; Escape or a click on another target closes it without posting. A site with a single id posts immediately. `deliver_to` sites never open a popover.
- Hover ghosts: at most one `[data-testid="ghost"]` at a time, only for a legal target, removed on mouse leave. `data-kind`: `street` (`data-street`, `data-seat` = actor), `building` (`data-site`, `data-seat`), `passenger` (`data-node`), `marker` (`data-action`, `data-letter` = letter of `params.space`, `data-seat`), `clock-hand` (`data-clock` = `next_clock`), `path` (VD: `data-from`, `data-to`, `stroke-dasharray` set).
- Bar buttons live inside `[data-testid="bus-prompt"]` and are real `<button>`s: **Drop out** (only with a `drop_out` button target), **Advance → Pub** (name matches /advance/i), **Take stone (n left)** (name matches /take stone/i), and **Back**. Each posts the same id as its board target. `BoardUi` gains `onBack?: () => void | Promise<unknown>` and `canBack?: boolean`; **Back** is rendered when `onBack` is given, disabled unless `canBack`, calls `onBack` only. `GameScreen` supplies `onBack` = `goBack` and `canBack` = `history_length > 0 && !terminal && !busy`.
- Hit size (§8): at fit zoom the reference board is 640 px square (a 1280 × 720 window less bar and chrome), so one map unit is `640 / map.width` px. Every *legal* target's hit shape is at least 24 px in its smallest dimension (circle: 2r; rect: min(width, height); path: stroke-width). Overlap between neighbouring legal hit shapes is allowed; the nearer centre wins.

## 7. Hotseat, bots, undo

- Hotseat: perfect information, so no curtain; the turn frame and toast make the change of seat obvious.
- Bots: moves play at a readable pace (default 0.8 s, setting 0–2 s), each with its animation; a "skip to my turn" button fast-forwards.
- Back: undoes to the previous human decision; disabled at the start and at END.

## 8. Quality bar and checks

- Every legal action reachable by clicking (vitest over recorded legal sets for every phase code).
- Hit areas ≥ 24 px on screen at fit zoom; streets get a wide invisible hit stroke.
- Seat identity never by colour alone: seat label on hover, panel order, and P-number on action markers.
- Playwright screenshots of fixed states (setup, mid-selection, each resolution phase, END) are sent to the user for review at the end of each visual unit; approved shots become visual regression baselines.

## 9. User choices (2026-11-14)

- Q1 Seat colours: **match the art's Lines stripes** (red, green, orange, blue, white for P0–P4).
- Q2 Single legal action: **always click** for human seats; bots auto-play.
- Q3 Lines: **legal streets only**, no blocked-street explanation.
- Q4 Screens: **desktop/laptop** (≥ 1280 × 720, mouse).
