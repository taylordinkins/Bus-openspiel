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

Typography: one bold condensed sans for headings to match the art's lettering, a plain sans for text; the board palette tokens in `docs/board-palette.md`. Light theme only (the art is cream); dark mode is out of scope. The full theme is §3a.

### 3a. Theme (unit 8b)

The whole app (shell included) follows the map: cream paper, its inks and reds, condensed bold lettering, the map's framed panels and letter tiles. Nothing is carried over from Lignum's look.

- **Tokens.** `web/src/theme/tokens.css` is the single source of colour, type, spacing, radius, depth and motion (CSS custom properties on `:root`). Palette values are those of `docs/board-palette.md` (`--paper`, `--ink`, `--red`, `--red-dark`, `--stone`, `--river*`, `--district-*`, `--seat-0..4` with `--seat-ink-0..4`), plus derived paper shades (`--paper-raised` cards, `--paper-sunk` wells, `--paper-deep` page, `--paper-edge` hairlines), text inks (`--ink-soft` ≈ 11:1, `--ink-muted` ≈ 6:1 on paper, both AA), the logo colours, `--glow` (legal targets) and `--focus` (dark river blue). The board SVG cannot read custom properties in presentation attributes, so `web/src/games/bus/model.ts` mirrors the same values (`PAPER`, `INK`, `RED`, `GLOW`, seat colours); components carry no other raw colours.
- **Fonts.** Display: *Barlow Condensed* 600/700/800 (headings, phase titles, buttons, chips, scores, tiles, board labels); text: *Barlow* 400/500/600 (+400 italic for log events). Both are self-hosted from `@fontsource/*` (latin subset, `font-display: swap`, bundled by Vite, so builds and e2e are offline-safe), with fallbacks `"Roboto Condensed", "Arial Narrow", Arial` and `"Segoe UI", "Helvetica Neue", Arial`.
- **Type scale.** 11 / 12 / 13 / 15 (body) / 17 / 20 / 26 / 36 / 64 px (`--fs-2xs` … `--fs-hero`); uppercase with `--tracking-caps` only for short display labels (eyebrows, field labels, phase titles, tiles).
- **Space and shape.** 4 px grid (`--sp-1` … `--sp-10`); radii 3/5/8/12 px (5 px is the map's letter tile); 2 px ink rules.
- **Depth.** Paper lying on paper: soft warm shadows (`--shadow-1..3`), `--inset` for wells and fields. The map's framed panel (red frame with a thin cream rule inside) is used for the primary surfaces: the new-game form, results, dialogs, the curtain card, the type popover.
- **Components** (`web/src/theme/base.css`, `web/src/shell/layout.css`, `web/src/games/bus/bus.css`):
  - Buttons: secondary by default (paper, 2 px ink border, a 2 px ink drop that presses in on `:active`); `.btn-primary` is the map's red letter tile (red, cream text, inner cream rule) for the move-making choices (Drop out, Advance, Take stone, Confirm, Create game, Click when ready, Submit report); `.btn-ghost` for toolbars; `.btn-sm`/`.btn-lg`/`.btn-icon`. Disabled = sunk paper, muted ink, no shadow, `not-allowed` cursor.
  - Focus: one `:focus-visible` ring everywhere, 3 px `--focus` with 2 px offset; fields show a blue border and halo.
  - Chips: ink tiles with a thin cream inner rule (prompt bar units/power/context), after the map's letter tiles. Seat badges: seat colour, seat ink, 2 px ink border, P-number always printed.
  - Section headings in the lobby use the map's numbered headers (black or red number tile + condensed title).
  - Dialogs (`.dlg` in `.dlg-backdrop`): framed panel over a warm, slightly blurred backdrop; eyebrow, condensed title, actions bottom-right with the primary on the right. Errors are `.alert` (red rule, "!" badge); in-game errors float as a toast at the bottom centre. Loading states show a red/cream spinner with the text.
  - Tooltips: ink card with paper text; board tooltips put the first line (the place) in `--glow` and flip away from the right and bottom edges.
- **Game screen layout.** Desktop is exactly one viewport: board column left, side column right (`--side-w` = clamp(21rem, 100vw − 100vh − 40px, 25rem)). The prompt bar spans the board column with a seat-colour band at its left edge; the board is a square that hugs the map, centred below it (≥ 640 px at 1280 × 720), with the zoom rail floating in the margin to its left (on the board's corner when there is no margin) and the raw-view tab at the bottom left. In hotseat the turn frame hugs the board. The side column holds the wordmark, Lobby, an icon toolbar (Back, Export, Import, Report, each with a tooltip and an accessible name), the game line, the dev Rewind row, then (at END) the results, the seat panels, the round strip, the collapsed action list (the debug panel of §1.3, toggled by "Action list (n)") and the log card, which takes the remaining height and scrolls.
- **Seat panels.** Paper cards with a keylined seat-colour band, condensed score numerals, front-view bus icons after the map's Buses grid (owned filled ink, missing dashed sepia), a wax-seal stone icon, tags (START, TO ACT, DROPPED OUT) as small caps tiles; the acting panel is raised with a 2 px ink border and ink drop.
- **Lobby.** Hero with the BUS wordmark (red/yellow/blue letters with an ink keyline over two route stripes, after the map's logo), eyebrow, tagline; the framed new-game form (a Start from control, Fresh setup or Scenario (6.9); players, returns, seat cards with seat tiles, a hotseat/bots hint, seed, a large red Create game); beside it a tilted print of the map, "Continue a game" (import) and "At the table" tips.
- **Curtain.** Full-screen paper with diagonal red hatching, a thick seat-colour band round the screen edged in ink, and a framed card with a large seat tile, "Pass to Pn" and a red "Click when ready".
- **States.** Hover: paper darkens (buttons), cards lift (seat panels), log lines and table rows take `--glow-soft`. Board glow pulses gently and stops under `prefers-reduced-motion`, which also zeroes UI transitions.

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
- **Events.** Each `HistoryEntry` has `events: list[Event]` (`type` plus type-specific fields): the automatic steps the engine ran as a result of that action, computed from the states before and after it (so replay, bots and Back give the same events). The latest action's events are `history[-1].events`; after bot turns, take every entry past the last one seen. Types: `scored {seat, points, node, site}` (a delivery); `clock_advanced {from, to}` (the automatic Clock step of a round with no Clock marker, D22/R-050: the hand moves one space, building type names; emitted at the Clock slot, after the Buildings and before the Vrrooomm! markers in resolution order; a Clock marker's own advance is the action itself, not an event); `passengers_moved {exited: [{site, node}], entered: [{site, node}]}` (Clock step, after a CK decision or right after `clock_advanced`: all occupied buildings release, then clock-type buildings take one; only when some building released or took a passenger; not emitted for the last-stone end); `selection_ended {}`; `bus_added {seat, buses}` and `marker_skipped {action, letter, owner}` (markers resolved without a decision, in resolution order; a marker that ends early with units unused is not reported); `round_ended {round, start_player}` (`start_player` is omitted when the round end also ends the game, R-066: there is no next round); `round_started {round}` (also the first SEL after setup); `game_ended {end_types}`. **Last-stone end (R-053, R-067):** a CK pause that takes the last stone ends the game at once, so its only event is `game_ended {end_types: ["last_stone"]}`: the markers still on the board never resolve (no `marker_skipped`, `bus_added` or `clock_advanced`), no `passengers_moved`, and no `round_ended` (Starting Player never resolves; `start_player` is unchanged). Applying them to the previous `state` reproduces the next state's seats' buses and points, occupied buildings, waiting passengers (with the action's own effect), the clock (with a `clock_advance` action's own effect), round and END.
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

### 6.8 Playback, motion, settings (unit 9)
- **Frames.** `GET /api/games/{id}?at=N` (optional `seat`) renders the state after the first `N` history entries: same `GameView` shape, `history_length = N`, `history` = the first `N` entries, legal actions of that state; `bot_thinking` is true when a bot was to move there, `bot_error` null. `N` out of `0..history_length` is 409; `N = history_length` is the live view. Read-only.
- **Playback** (`shell/playback.ts`). The shell renders a *shown* view that may lag the latest server view. New entries are split into steps: each bot entry is one step, runs of human entries merge into one, chance entries join the step before. Human steps show at once; each bot step waits until `pace` seconds after the previous shown change, then shows the frame at its end (the last step is the latest view, no request). While shown is behind latest, `canAct` is false, the action list is hidden, auto-advance waits, and the status line shows the bot with **Skip to my turn**; Skip jumps to the latest view and keeps jumping until a human seat is to move or the game ends. A view that no longer extends the shown history (Back, rewind, another game) shows at once and cancels playback; a failed frame request jumps to the latest view. Pace 0 shows the latest view at once.
- **Motion.** Each shown change that extends the previous one carries `BoardUi.motion = {seq, from, to, events, speed, scale}` (null with animation off): `events` are those of entries `[from, to)`. The board eases in pieces that are new since the previous rendered view (`data-anim="in"`: line streets, buildings, occupants, waiting groups, buses, action markers, score discs, clock hand) over `--t-place × scale`; `scored` walks a passenger from `node` along the scorer's line (`derived.seats[s].line_nodes`) into `site` and floats a seat-coloured "+1" from the site to that seat's panel score (`[data-testid=score-float]`, body portal); `passengers_moved` walks exits (site → node), then entries (node → site); `clock_advanced` sweeps the clock hand clockwise from `from` to `to` (the `.clock-hand` group gets `data-anim="turn"` and `data-from`, over `--t-move × 0.6 × scale`; the change's passenger walks and deliveries start after it), and the log shows "Clock advances House → Office (no Clock marker)" with no seat. A Clock marker's own advance is not an event and keeps the plain ease-in. The motion layer takes no pointer events; hit targets and selectors are unchanged.
- **Hand-off after motion.** A plugin may report a change's animation length (`GamePlugin.motionMs(motion)`; Bus: `motionDuration(events, scale)` + 60 ms, an upper bound of the board's walks, floats and ease-ins). While that time runs after a shown change (animation on, bot pace > 0), a hotseat change of viewed seat is held: the board stays on the mover's view, `canAct` is false and auto-advance waits; then the viewed seat changes and the curtain covers the board (§7).
- **Settings** (`shell/settings.ts`, toolbar **Settings** dialog). `{animation: "normal" | "fast" | "off", botPace: 0..2 s (0.1 steps)}` in `localStorage["bus.settings.v1"]` (read/write in try/catch); defaults normal (off under `prefers-reduced-motion`) and 0.8 s. Scale: normal 1, fast 0.4. Vitest runs with animation off and pace 0 (`setupTests.ts`) unless a test sets them.

### 6.9 Scenarios in the lobby (unit 10; D20, editor deferred)
- **Named list.** `GET /api/scenarios?game=bus` returns `ScenarioSummary[]` `{id, game, title, description, players, returns, round, phase}`; `GET /api/scenarios/{id}?game=bus` returns the same plus `scenario` (the JSON to post back). 404 for an unknown game or id. The list is the curated manifest `data/scenarios.yaml` (ids, titles, files under `data/scenarios/`, byte-identical copies of the `tests/scenarios/` fixtures kept in sync by `tests/server/test_scenario_files.py`); `description` is the file's own, `expect` is dropped, `round`/`phase` are those of the loaded position. Every entry loads with `require_legal` (checked at first use and by `tests/server/test_scenarios_api.py`).
- **Parameters from the scenario.** On create/import with a scenario, `num_players` and `returns` come from the scenario (`parameters`) when the client leaves them unset; an explicitly set value that differs is 422 (`<param> <given> does not match scenario (<value>)`). Any loader error is 422 `invalid scenario: S-…: …`.
- **Lobby.** The new-game form has a **Start from** segmented control: *Fresh setup* (players, returns, seats, seed as before) or *Scenario*. Scenario mode shows a *Named scenario* picker (list fetched on first switch), an *Upload .json* button and the *Scenario JSON* textarea; picking or uploading fills the textarea, which may then be edited. A status line under it (`shell/startDoc.ts`, shape only) reads the document: a **scenario** (has `format`; players/returns from `parameters`, default 4 / win_loss) or a **replay export** (has `seats` and `history`; its seat types are adopted when it loads). Players/returns are not editable there; seat cards follow the document's count. *Create game* posts a scenario to `POST /api/games` (`{game, seats, scenario, seed?}`, no `params`) and a replay to `POST /api/games/import` with the export's fields and the lobby's `seats` (no seed). Unparsable JSON, an empty textarea and every server rejection show in the form's `.alert`; nothing is posted for local errors.
- **Export round trip.** The toolbar Export file (`GameExport`, with `scenario` for scenario games) pasted or uploaded in Scenario mode, or loaded through *Continue a game*, rebuilds an equal `view.state` and history; `view.state` itself pasted as a scenario also gives an equal state (6.1). `{animation: "normal" | "fast" | "off", botPace: 0..2 s (0.1 steps)}` in `localStorage["bus.settings.v1"]` (read/write in try/catch); defaults normal (off under `prefers-reduced-motion`) and 0.8 s. Scale: normal 1, fast 0.4. Vitest runs with animation off and pace 0 (`setupTests.ts`) unless a test sets them.

### 6.10 Bot seats and think time (Phase 3 unit 6)
- **Seat picker.** Each seat card (`Seat P0`...) is one select: Human, Random bot, Heuristic bot, and MCTS at three strengths, Quick (50 sims), Normal (200) and Strong (1000), posting `human`, `random`, `heuristic`, `mcts:50`, `mcts:200`, `mcts:1000`. Defaults are unchanged (P0 human, the rest random). A seat type adopted from a loaded replay (e.g. `mcts:7-heuristic`) shows as an extra option.
- **Think time.** A *Think time (seconds per move)* number field (default 5, 0.1..60) appears under the seats only while some visible seat is MCTS; Create is disabled while it is out of range. It posts as `bot_think_time_s` on create and import (omitted when no seat is MCTS, so the server default 5 applies). The server hands the cap to MCTS seats (`with_time_limit`); an MCTS failure or illegal result falls back to the heuristic bot for that move instead of the bot-error state. `GameView.bot_think_time_s` and `GameExport.bot_think_time_s` (exported only when not the default) carry it back.
- **Labels.** Seat panels read "P2 · Heuristic" style names from `seat_types` (`seatTypeName`); the "bot is thinking" prompt/toolbar text stays up for as long as a slow MCTS move takes, since `bot_thinking` is true until the worker applies the move.

## 7. Hotseat, bots, undo

- Hotseat (more than one human seat): when the human seat to move changes, a hand-off curtain (§3a Curtain, `shell/Curtain.tsx`) covers the board, actions and log with "Pass to Pn" until that seat clicks "Click when ready" (user decision 2026-11-15, although Bus has no hidden information). The curtain waits for the change that hands over to finish animating (§6.8 motion; length from the plugin's `motionMs`), so the mover sees their move play out; meanwhile the viewed seat stays the mover's and no input or auto-advance is accepted. With animation off it shows at once. No curtain with one human seat or at END; the turn frame and toast still mark the change of seat.
- Bots: moves play at a readable pace (default 0.8 s, setting 0–2 s), each with its animation; a "skip to my turn" button fast-forwards (contract §6.8).
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
