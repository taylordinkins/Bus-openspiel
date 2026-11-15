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

## 6. Data the adapter adds (Phase 2 units 2–3)

- `view.prompt`: `{phase, actor, unit, units_total, context}` (e.g. clock type, picked node, supply, pass index) so the status bar never parses strings.
- `view.targets`: legal ids grouped by target (`street`, `site`, `station`, `node`, `action`, `clock`, `button`) built from `action_meta`.
- `events` per applied action: the automatic steps that followed (bus added, passengers stepped out/in, seats skipped, markers spent at power ≤ 0, round end, start player), computed by the adapter from the state before and after, for animation and the log.

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
