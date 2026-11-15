# Bus — Rules Specification

**Status:** APPROVED

Contract between the rulebook and the engine. Every rule behaviour in `bus/` must trace to an entry here (CLAUDE.md). Only the user changes the Status line (D12). Entries marked ⚑ are defaults the user may veto during the Phase 0 review.

## Sources and priority

1. The *Bus* Complete Edition rulebook (`third_party/rules/bus_rulebook.pdf`), Capstone Games / Splotter 2024. The PDF is two large sheets, cited here as "RB p.1" (cover sheet: contents, Assembly, Setting Up, First Buildings, First Bus Lines, Play, Phase 1: Select Actions, Phase 2 intro, Line Expansion power text) and "RB p.2" (second sheet: Line Expansion rules and examples, Buses, Passengers, Buildings, Clock, Vrrooomm!, Starting Player, cleanup, End of the Game). The section name is given where it helps.
2. PLAN.md §3 decisions (D-numbers) and §4 clarifications (C-numbers). A ⚑ clarification is a default where no source settles it.
3. Where the rulebook and PLAN §4 differ, the rulebook wins unless a D-number or a user answer in PLAN §9 overrides it. None of the differences found are of that kind; the open ones are in "Open questions" at the end. D22 (2026-11-15) replaces the ⚑ defaults of C3 (starting passengers), C10 (the Clock step only on a marker) and C11 (same-node Vrrooomm!) where the rulebook is silent; it overrides no rulebook text.

The rulebook is paraphrased here, never pasted.

## Conventions

- **Rule IDs** are `R-001` style, sequential, permanent. A retired rule keeps its ID and is marked RETIRED.
- **Seats** are numbered from 0 everywhere (engine, tests, logs, examples) and labelled `P0`…`P4`. Never "Player 1" for seat 0.
- **Board vocabulary** (R-009 to R-013): node = intersection; street = edge between two nodes; site = construction site, with a tier A–D; station = train-station node; roundabout = Central Park roundabout node. Topology lives only in `data/board.yaml` (loaded by `bus/board.py`). This spec never states a node, street or site id as a rule; ids in examples are labelled examples, and most examples use small abstract fragments with letters (nodes `u`, `v`, `w`, streets `s1`, `s2`, …).
- **Action letters** are a (first/lowest) to f. Internally a letter has index k, a = 0 … f = 5.
- "Building" means a house, office or pub disc on a site. "Waiting passenger" means a passenger standing at a node, not in a building.
- "M" is the bus maximum (R-032).

---

## 1. Parameters

### R-001 — Player count
`players` ∈ {3,4,5}, default 4. Counts differ only in the number of time stones (R-006) and the Lines action power (R-033). All other rules are identical. Two-player and solo variants are out of scope.
Source: D6; C2; RB p.1 (Setting Up, 3-player note; Line Expansion, 5-player note).

### R-002 — Returns parameter
`returns` ∈ {`win_loss`, `margin`}, default `win_loss`. It only affects the returns vector at game end (R-071, R-072). It never changes legal actions or any other state.
Source: D7.

### R-003 — Seats and turn order
Seats are `P0`…`P(n-1)`. Turn order starts at the current start player and goes through ascending seat numbers, wrapping from the last seat to P0 (this is the rulebook's "clockwise"). **Reverse turn order** is the same cycle backwards, so it begins with the last player in turn order and ends with the start player.
Example (n=4, start player P2): turn order P2, P3, P0, P1; reverse turn order P1, P0, P3, P2.
Source: RB p.1 (Setting Up: "turn order is clockwise, starting with this player"; First Bus Lines); C3.

---

## 2. Components

### R-004 — Per-player components
Each seat has: 20 Action markers, 25 Bus Line markers, 5 buses and one score disc. Markers and buses are never shared between seats. A seat's Action markers and Bus Line markers, once used, are never recovered (R-024, R-065). Of the 5 buses, 1 is placed at setup (R-014); the rest are added one at a time by the Buses action (R-043).
Source: RB p.1 (Contents, Setting Up); C2.

### R-005 — Passengers
There are 15 passengers in all, all identical. At setup 4 are placed on the board (R-015) and 11 are in the supply. A passenger is always in exactly one place: the supply, a waiting position at a node, or a building. Passengers never leave the game. The only way the supply shrinks is the Passengers action (R-045). Invariant: passengers on board + supply = 15.
Source: RB p.1 (Contents, Setting Up); C2, C8.

### R-006 — Time stones
There are 5 time stones, or 4 when `players` = 3. At setup all of them are on the clock. A stone, once taken, stays with its taker for the rest of the game.
Source: RB p.1 (Setting Up), RB p.2 (Clock); C2.

### R-007 — Buildings
A building is of one of three types: house, office or pub. The box holds 90 building discs and the board has 47 sites, so building supply is treated as unlimited (never a constraint). The per-type split of the discs is not stated (see O-1).
Source: RB p.1 (Contents, First Buildings); C2.

### R-008 — Score and the score track
A seat's points start at 0. Points increase only through Vrrooomm! (R-062); they never decrease. Time stones count −1 each at game end (R-068) but never move the score disc and are not part of "points". The score track has no cap.
Source: RB p.2 (Vrrooomm!, End of the Game); C2.

---

## 3. Board

### R-009 — Board data and vocabulary
The board is the reworked map as digitized in `data/board.yaml`: nodes (intersections), undirected streets between two nodes, sites (construction sites, each with a tier and a node), stations and roundabouts. All topology comes from that file; the engine never hard-codes ids. Where this spec and the data disagree, that is a bug to report.
Source: D10; C1.

### R-010 — Site tiers and the lowest open tier
Sites have tiers A, B, C, D (the rulebook's numbered tiers 1, 2, 3, 4 become letters; A is the lowest). The **lowest open tier** is the first tier in order A, B, C, D that still has at least one empty site. Every Buildings-action build and every setup building must go on an empty site of the lowest open tier (R-016, R-047).
Example (illustrative): tier A has one empty site and tier B has several. The lowest open tier is A. Once that site is filled, the lowest open tier is B.
Source: RB p.1 (First Buildings), RB p.2 (Buildings); C1.

### R-011 — A site belongs to one node
Every site belongs to exactly one node, the intersection passengers enter it from and step out to. A building lives at its site's node. "A building is on a player's line" means its site's node is on that line (R-035).
Source: C1; RB p.2 (Vrrooomm!: building "on your bus line").

### R-012 — Stations and roundabouts
Stations (2) are the nodes where the Passengers action places passengers (R-045). Stations have no sites. Roundabouts (4) are the nodes that receive one passenger each at setup (R-015). Other than these roles a station or roundabout is an ordinary node: lines pass through it and passengers wait there.
Source: C1; RB p.1 (Setting Up); RB p.2 (Passengers).

### R-013 — Streets and occupancy
A street joins two distinct nodes and has no direction. A street is **occupied** if at least one player has a Bus Line marker on it, and **unoccupied** otherwise. Several players may have markers on one street; one player may never have two markers on the same street (R-037). Bends in a street's drawing have no rules effect.
Source: RB p.2 (Line Expansion rule 1b, rule 3); C1.

---

## 4. Setup

### R-014 — Initial state
At the start of setup: the clock points at house; every seat's points are 0; every seat has exactly 1 bus; all time stones are on the clock (R-006); no sites are filled; no lines exist; every seat has all 20 Action markers and 25 Bus Line markers; the action board is empty.
Source: RB p.1 (Assembly, Setting Up); C2, C3.

### R-015 — Starting passengers
Place one passenger on each of the 4 roundabout nodes (the rest, 11, stay in the supply). These passengers stand at the intersection and do not enter buildings during setup. They can first step in at the Clock step of round 1 (R-050, R-055), which runs whether or not anyone selected Clock. Until then they are waiting passengers.
Example: a roundabout node gets a house and an office during first buildings. Nobody selects Clock in round 1, so at the round-1 Clock step the clock advances house → office and the passenger there steps into the office (R-055); it is then not eligible for Vrrooomm! that round (R-058). If instead the owner of a Clock marker pauses in round 1, the clock stays at house and the passenger steps into the house.
Source: RB p.1 (Setting Up: one passenger on each roundabout); D22 (the rulebook is silent on entering buildings at setup; replaces C3's ⚑ default).

### R-016 — First buildings
After setup of the board, in turn order from the start player, each player in turn places 2 buildings: each on any empty site of the lowest open tier (R-010), of any type of that player's choice. Both buildings are placed by one player before the next player begins. Because the lowest open tier is re-evaluated for each building, the first building may fill the last empty tier-A site, so the second must then go on tier B.
Example (4 players, 12 tier-A sites): P0 puts two on A, P1 two on A, P2 two on A, P3 two on A; 8 tier-A sites filled, 4 left.
Source: RB p.1 (First Buildings); C3.

### R-017 — First Bus Line marker
After first buildings, one time around in turn order from the start player, each player places one Bus Line marker on any street. The street may already be occupied by other players' markers. This marker starts that player's line (R-035).
Source: RB p.1 (First Bus Lines); C3.

### R-018 — Second Bus Line marker
Then one time around in **reverse** turn order (R-003), each player places a second Bus Line marker extending their line from either end of the first marker, under the placement rule for extensions (R-036 to R-040). The candidate streets are those incident to one of the first marker's two end nodes (excluding the street already placed, which the player may not repeat). The player picks both the end and the street.
Example: P0 starts on street `s1` between nodes `u` and `v`. P0's second marker may go from `u` via any street at `u` other than `s1`, or from `v` via any street at `v` other than `s1`, subject to R-038 and R-039. After placement the line has two streets and the new end is where the second street ends.
Source: RB p.1 (First Bus Lines); C3, C5.

### R-019 — Starting player
Seat 0 always starts the first round (D5). There is no start-player parameter; the lobby decides who sits in seat 0. The start player's seat determines the turn order for the setup placements (R-016 to R-018) and for round 1 (R-022 onwards).
Source: D5; C3; RB p.1 (Setting Up: last person to ride a bus starts, or choose randomly). The rulebook's random or "last rider" choice is replaced by the fixed seat-0 start.

---

## 5. Round structure

### R-020 — Rounds have two phases
A round has Phase 1 (select actions, R-022 to R-028) then Phase 2 (resolve actions, R-029 onward). Then round end (R-064, R-065) and the end-of-game checks (R-066).
Source: RB p.1 (Play, Phase 1, Phase 2); §5.3.

### R-021 — Setup precedes round 1
Setup (R-014 to R-019) completes before the first round's Phase 1. There is no Phase 2 for setup.
Source: RB p.1 (Setting Up through First Bus Lines).

---

## 6. Action selection (Phase 1)

### R-022 — The action board
There are 7 actions, listed in resolution order (R-029): Lines (Line Expansion), Buses, Passengers, Buildings, Clock, Vrrooomm!, Starting Player. Buses, Clock and Starting Player each have **1** space. Lines, Passengers, Buildings and Vrrooomm! each have **6** spaces, lettered a to f. The board is empty at the start of every round.
Source: RB p.1 (Phase 1: the action board shows single-space Buses, Clock and Starting Player, and six lettered spaces for the other four actions, displayed from F down to A); C4, C6.

### R-023 — First two selections are mandatory
Starting with the start player and going in turn order, each player places one Action marker on an action. Then a second pass in turn order, in which each player places another Action marker. These two passes are mandatory (no dropping out); a player may choose the same action in both.
Source: RB p.1 (Phase 1); C4.

### R-024 — Later passes and dropping out
After the two mandatory passes, passes continue in turn order. On each pass, every player who has not dropped out either places one more Action marker or **drops out** for the rest of the round (dropping out is permanent for the round; a dropped-out player is skipped in later passes). A placed marker is spent: it can never be recovered, and it is returned to the box at round end (R-065).
Example (3 players, start P0): P0 a, P1 b, P2 c; second pass P0 a, P1 d, P2 e; third pass P0 drops, P1 places, P2 drops; fourth pass P1 drops. Selection is over.
Source: RB p.1 (Phase 1: "either dropping out ... or placing another"; spent markers are never recovered); C4.

### R-025 — Choosing a space on a multi-use action
Placing a marker on an action with lettered spaces puts it on the earliest empty letter (a, then b, …). The player chooses only the action, not the letter. An action whose spaces are all occupied cannot be chosen.
Example: Lines has a and b occupied; the next marker placed on Lines goes on c.
Source: RB p.1 (Phase 1: "place your Action marker on the empty space earliest in the alphabet"); C4.

### R-026 — Any action with a free space may be chosen
A player may place a marker on any action that still has a free space, even when that action will have no effect for them (a blocking tactic): for example Buses with 5 buses already, Passengers when the supply is empty. Confirmed by the user (Q2, 2026-11-14).
Source: C4; RB p.1 (Phase 1) is silent on ineffective placements.

### R-027 — Players with no markers left
A player with no Action markers left cannot place and is skipped, which counts as being out for the round and for the rest of the game. A player who still has markers but whose every action space is full must drop out. Passes continue for the others.
Example: P2 has 1 marker left at the start of a round. In the first mandatory pass P2 places it; in the second mandatory pass P2 is skipped.
Source: RB p.1 (Phase 1: "you will take no more actions for the rest of the game"); C4. The mandatory-pass skip for a player holding fewer than 2 markers is C4, not the rulebook.

### R-028 — End of selection
Phase 1 ends once every player has dropped out or is skipped. Phase 2 then begins.
Source: RB p.1 (Phase 1: "once everyone has dropped out"); C4.

---

## 7. Resolution order and action power (Phase 2)

### R-029 — Resolution order between actions
Actions resolve top to bottom: Lines, Buses, Passengers, Buildings, Clock, Vrrooomm!, Starting Player. Every marker on an action resolves before the next action starts. An action with no markers is skipped, with one exception: the Clock step (R-050) runs every round at the Clock action's place in this order (after Buildings, before Vrrooomm!), with or without a Clock marker.
Example: round 1 has markers only on Lines, Passengers and Vrrooomm!. Order: Lines, Passengers, the Clock step (automatic advance, house → office), Vrrooomm!; then, with no Starting Player marker, the start passes on (R-064).
Source: RB p.1 (Phase 2: "from top to bottom ... starting with Line Expansion and ending with Starting Player"); C6; D22 (the Clock step every round, at the Clock slot: user confirmed 2026-11-15).

### R-030 — Order within a multi-use action
Markers on one action resolve left to right as the spaces are printed. For Lines and Buildings the highest letter resolves first (f, e, … a), so the player who selected last goes first. For Passengers and Vrrooomm! letter a resolves first (a, b, … f), so the first to select goes first. Each marker resolves completely, owned by the player who placed it, before the next.
Example: Lines has markers on a (P1), b (P2), c (P0). Resolution order: c (P0), b (P2), a (P1).
Source: RB p.1 (Phase 2); C6.

### R-031 — Dwindling power
Lines, Passengers and Buildings have **dwindling action power**: the marker on the letter with index k (a = 0) has power M − k (R-032), where printed penalties show −1 for b, −2 for c, … −5 for f. A marker with power ≤ 0 does nothing but is still spent. A marker with power p ≥ 1 performs its effect p times, each time as the rule of that action says. The Lines action has +1 power at 5 players (R-033).
Example: M = 3. Markers on a, b, c, d of Lines have power 3, 2, 1 and 0 (the d marker does nothing).
Source: RB p.1 (Phase 2: dwindling action power, example with M = 3); C6.

### R-032 — The bus maximum M
M is the largest number of buses any single player currently has (1 to 5). It is read at the moment each marker resolves, not at the start of Phase 2. Because Buses resolves before Passengers and Buildings (R-029), a Buses action in the same round can raise M for them. Lines resolves before Buses, so Lines never sees the same round's new bus.
Example: all players have 2 buses; P1 resolves Buses (3 buses, M becomes 3). Passengers markers resolving afterwards use M = 3; Lines markers resolved earlier used M = 2.
Source: RB p.1 (Phase 2, "M is the bus maximum"); RB p.2 (Buses: new bus raises M immediately); C6.

### R-033 — Five-player Lines power
When `players` = 5, a Lines marker's power is (M − k) + 1. This does not apply to Passengers, Buildings, Buses, Clock, Vrrooomm! or Starting Player. The +1 is added first, then the "power ≤ 0 does nothing" check applies to the result (user decision Q-A, 2026-11-14 ⚑-resolved).
Example (5 players, M = 3): Lines markers on a, b, c, d, e, f have power 4, 3, 2, 1, 0, −1 (so e and f do nothing). With M = 3, letter d has power 1 and places one street.
Source: RB p.2 (Line Expansion: "5-player game: action power + 1, but only for Line Expansion"); C6. The +1 is read as applying to every letter's power.

### R-034 — Every effect is carried out as fully as possible
When a marker resolves, its owner must carry out the full effect and may not choose to do less. If part of an effect becomes impossible (no supply, no legal target) it is lost; the rest still happens. Vrrooomm! is not dwindling: each marker uses all its owner's buses (R-057).
Source: RB p.1 (Phase 2: "you must resolve it as fully as possible"); C6.

---

## 8. Lines (Line Expansion)

### R-035 — A line is an ordered path of streets
A player's **line** is an ordered path of streets with two ends: path index 0 is end 0, the last is end 1. Consecutive streets share a node. A line may pass through a node more than once (loops). A line never has two markers on the same street (R-037). A single-street line has two ends, one at each of its nodes. A node is **on** a player's line if it is an endpoint of one of the line's streets. The engine stores the path, so ends are never ambiguous even in loops.
Source: RB p.2 (Line Expansion rules 1 and 2); C5.

### R-036 — Extending
To extend a line, add one Bus Line marker on a street adjacent to one of its ends, so the path grows by one street at that end. The player chooses the end and the street. After the placement that end is the far node of the new street.
Source: RB p.2 (Line Expansion); C5.

### R-037 — Candidate streets and the no-overlap, no-branch rule
For an end at node `v`, the **candidate streets** are streets incident to `v` that the player does not already have in their line. The line must always remain a single non-branching path, so a street can only be added at an end; it may not be a street the player already has.
Example (illustrative): the player's line has an end at `v`; streets `s1`, `s2`, `s3` meet at `v`; `s1` is already in the line. The candidates are `s2` and `s3`.
Source: RB p.2 (Line Expansion rule 1a, 1b); C5.

### R-038 — Placement rule, per end
For a chosen end, if any of that end's candidate streets (R-037) is unoccupied (R-013), the new marker must go on an unoccupied candidate street. If every candidate street at that end is occupied, any candidate may be chosen. The rule is evaluated per end, not across both ends; evaluation is sequential, so a marker placed earlier counts as occupying its street when a later extension is evaluated. The exception in R-039 widens the set of allowed streets.
Example (illustrative, from the rulebook's Red example): at end node `v`, candidates `s1`, `s2` where `s1` is yellow-occupied and `s2` is unoccupied. Red extending from `v` must choose `s2`. At the other end `w`, all candidates are occupied, so Red may choose any of them.
Source: RB p.2 (Line Expansion rule 3 and the Red C/D example); C5 (per-end reading confirmed by the user, Q2, 2026-11-14). See O-2 for the sequential-evaluation reading.

### R-039 — The shared-end exception
The player may extend from an end at node `v` onto street `y` even when an unoccupied candidate street exists at that end if: (a) another player's line has an end at `v`; (b) `y` is that other line's last street at that end; (c) `y` is not the street the player's own end arrives on (that is, "coming from a different direction"); and (d) `y` is a candidate (R-037). The exception allows exactly that one street per use, and it is an extra option: the unoccupied candidates remain legal.
Example (illustrative): Blue's end at `v` arrives via `b`. Yellow also has an end at `v`, arriving via `y`, `y` ≠ `b`. Unoccupied streets `s1` and `s2` also touch `v`. Blue may go on `s1`, `s2` or `y`, but not on another occupied street `z` that is not the last street of some other player's end at `v`.
Counter-example: if Yellow's last street at `v` is also `b` (the same street Blue arrives on), Yellow's end gives Blue no exception, because Blue is not "coming from a different direction".
Counter-example (middle of a line): if the player's end touches the *middle* of another player's line, that is, the intersection is not one of that line's ends, the exception does not apply (rulebook p.2: Blue extending from A, which lies mid-way along Red's line).
An end of another player's line that sits inside that player's own loop (a partial-line loop) is still an end; its last street is available to others under the exception.
After the exception is used, the new end is evaluated afresh: normally no exception applies at the next street unless another player's end is there.
Summary: a new marker may share a street with another line only when (i) all candidates at that end are occupied (R-038); (ii) the shared-end exception applies (R-039); or (iii) it is the first setup marker (R-017).
The exception also applies to the second setup marker (R-018).
Source: RB p.1 (First Bus Lines, asterisk), RB p.2 (Line Expansion, asterisk); C5. The formal test (b)–(c) is C5's wording of the rulebook's "same intersection, different direction".

### R-040 — Dead ends and maximal resolution
An end with no candidate street is dead. If both ends are dead, or the player has no Bus Line markers left in their supply, any remaining extensions of that Lines marker are lost. Extensions are lost only if no sequence of legal placements can place them: the Lines marker must be resolved maximally (user decision Q-B, 2026-11-14). See R-042 for how this restricts each choice.
Example: a marker with power 3 and a line whose only live end allows 2 further placements. Two extensions happen, the third is lost.
Source: RB p.2 (Line Expansion rule 3 and Phase 2 "as fully as possible"); C5.

### R-041 — Loops and ends
If a line forms a loop (a new street ends at a node already on the same line), both ends still exist as end 0 and end 1 of the stored path. For a line whose two ends sit at the same node (a full-line loop), the player's choice of which end to extend from is a real choice that determines which end the new street attaches to and which one stays; for a line whose ends are at different nodes the choice is determined by the node.
Example: the path `s1, s2, s3` closes into a loop at node `u` (end 0 and end 1 are both at `u`). Extending from end 0 puts the new street at the start of the path; extending from end 1 appends it. Later extensions evaluate R-038 and R-039 per end from the path's current ends.
Source: RB p.2 (Line Expansion rule 2 with the full-line and partial-line loop examples); C5.

### R-042 — Resolving a Lines marker
When a Lines marker resolves it has power p (R-031, R-033). For each of the p units in turn, the owner (if a legal extension exists, R-040) chooses an end and a candidate street under R-038/R-039, and places one Bus Line marker from their supply; the new marker counts as occupying its street for later units. Maximal resolution (user decision Q-B, 2026-11-14): when choosing each street, only (end, street) choices from which the maximum achievable number of remaining placements can still be reached are legal, where "remaining placements" is bounded by the marker's remaining power and the Bus Line markers left in supply, under R-037 to R-039. The lookahead depth is at most the remaining power (at most 6). A player's supply of Bus Line markers is 25 minus the markers already on the board; no marker is ever returned to supply.
Example (M = 3, marker on a, 4 players): power 3, so three streets are added one at a time. Marker on c: power 1, one street.
Example (maximal resolution): power 2, one end dead. At the live end, street s1 would leave both ends dead after one street, while s2 allows a second street. Only s2 is legal, because it is the only choice that places both streets.
Source: RB p.2 (Line Expansion: "extend your bus line a number of streets equal to its action power"); C5, C6.

---

## 9. Buses

### R-043 — Buses action
When a Buses marker resolves, its owner adds 1 bus to their Buses area if they have a bus left (that is, they have fewer than 5 on the board). With 5 buses already the action has no effect. The Buses action has no dwindling power. The new bus counts immediately for M (R-032).
Example: P0 has 2 buses and resolves Buses: P0 has 3.
Source: RB p.2 (Buses); C7.

### R-044 — Bus count bounds
A player always has 1 to 5 buses. Buses are never removed.
Source: RB p.1 (Contents, Setting Up), RB p.2 (Buses); C2, C7.

---

## 10. Passengers

### R-045 — Passengers action
When a Passengers marker resolves with power p (R-031), its owner places up to p passengers from the supply, one at a time; each goes onto either station, chosen by the owner for each passenger. The station does not need to be on the owner's line. When the supply is empty, the remaining units are lost.
Example: M = 3, supply has 1 passenger, marker on a (power 3): one passenger is placed, two units are lost.
Source: RB p.2 (Passengers); C8.

### R-046 — Passengers placed are waiting passengers
A passenger placed by the Passengers action is a waiting passenger at the chosen station. It becomes available to Vrrooomm! (R-058). A passenger at a station never steps into a building by itself (stations have no sites, R-012); Vrrooomm! may carry it away (R-058, R-059).
Source: RB p.2 (Passengers; Vrrooomm! Example 2); C8.

---

## 11. Buildings

### R-047 — Buildings action
When a Buildings marker resolves with power p (R-031), its owner builds up to p buildings, one at a time. Each build chooses an empty site of the lowest open tier (R-010) and a type (house, office or pub). The site need not be on any player's line.
Example: tier A has exactly one empty site; P0's marker has power 3. First build: that A site. Second and third: sites on tier B, because tier A is now full.
Source: RB p.2 (Buildings: "lowest-number construction site", "doesn't have to be on anyone's bus line"); C9.

### R-048 — Tier advance within one action
The lowest open tier is recomputed before each individual build, including within a single marker's power.
Source: RB p.2 (Buildings: "once the 1s are full, you must build on 2"); C9.

### R-049 — Sites full
Builds with no empty site left are lost. A state where every site has a building marks the game to end at the end of the current round (R-066). The remainder of the round still resolves.
Example: Buildings fills the last site during Phase 2. The Clock step still runs (R-050, with or without a Clock marker), Vrrooomm! and Starting Player still resolve, then the game ends (R-066).
Source: RB p.2 (Buildings: "the game ends at the end of this round"; End of the Game); C9, C13; D22.

---

## 12. Clock and passenger movement

### R-050 — The Clock step every round: advance or pause
Every round, at the Clock action's place in the resolution order (after Buildings, before Vrrooomm!, R-029), the **Clock step** runs once. First the clock advances (R-051) or pauses (R-052). Then passengers move (R-054, R-055), unless the game ended on the pause (R-053). The clock cycles house → office → pub → house. If a Clock marker was placed this round, its owner chooses advance or pause; the marker replaces the automatic advance, so the clock changes at most once per round. If no Clock marker was placed, the clock advances automatically (no one chooses, no stone is taken). This holds in every round, including round 1 and a round in which every site gets a building (R-049). Buildings built earlier in the same round take part in the step.
Example: round 1, nobody selects Clock. At the Clock step the clock advances house → office and passengers move; the Vrrooomm! markers then use office.
Example: P1's Clock marker resolves and P1 chooses advance. The clock moves one space (not two), and P1 takes no stone.
Source: RB p.2 (Clock: the marker's owner pauses or otherwise advances; "either way" passengers move); D22 (the step runs every round, at the Clock slot, and advances automatically without a marker: the rulebook is silent on the no-marker case; replaces C10; slot confirmed by the user 2026-11-15).

### R-051 — Advance
Advance moves the clock hand to the next type in the cycle. No time stone is taken. It happens automatically when there is no Clock marker, or as the Clock marker owner's choice (R-050).
Example: clock at office; advance → pub; clock at pub; advance → house.
Source: RB p.2 (Clock); C10; D22.

### R-052 — Pause
Only the owner of a Clock marker can pause; it is the only way to keep the clock where it is (R-050). Pause leaves the clock hand unchanged, and the owner takes 1 time stone from the clock; each stone the player holds is −1 point at the end of the game (R-068). Pause is legal only while at least one stone remains on the clock; since the game ends when the last stone is taken (R-053), a stone is always available when a Clock marker resolves.
Example: 5 stones on the clock (4 or 5 players); after one pause the pausing player holds 1 and 4 remain. In a 3-player game the clock starts with 4 and 3 remain.
Source: RB p.2 (Clock: "If you pause time, take a Time stone"); C10, C2; D22.

### R-053 — Last stone ends the game immediately
If a pause takes the last stone from the clock, the game ends immediately: no passenger movement, and no further action or marker resolves. The pausing player keeps that stone, and it counts −1 for them in the final score.
Example: 1 stone left; P1's Clock marker resolves and P1 pauses. The game ends. A Vrrooomm! marker that was to resolve later this round does not resolve.
Source: RB p.2 (Clock: "ends the game immediately"; End of the Game); C10, C13; D22.

### R-054 — Passengers step out of buildings
After the clock advances or pauses (when the game continues), every passenger standing in a building steps out to a waiting position at that building's node (R-011). This happens for all buildings, whatever their type, in every round's Clock step (R-050).
Source: RB p.2 (Clock: step 1: move passengers on building discs to the intersection); C10; D22.

### R-055 — Passengers step into buildings of the clock's type
Then at each node, waiting passengers enter empty buildings of the clock's current type at that node, one passenger per building, until either runs out. At that point all buildings are empty (R-054), so the number entering at a node is min(waiting passengers there, buildings of the clock's type there). Passengers are identical, so there is no choice; which of several same-type buildings at one node is filled is not rule-relevant (the engine fills in ascending site id for determinism). Passengers who stay out wait at their node. After this step no node has both a waiting passenger and an empty building of the clock's type, and every passenger in a building is in one of the clock's type. Vrrooomm! moves only fill buildings, so both stay true for the rest of the round (R-058, R-059). A state that breaks either of these while Vrrooomm! resolves is not a legal game state (SCENARIOS rejects it).
Example: clock = pub; node `u` has 3 waiting passengers and 2 pubs and 1 house. Two passengers enter the pubs; one waits at `u`. After a pause, passengers who were in pubs step out and re-enter pubs at the same node.
Example: clock = house; node `v` has 1 house and 2 waiting passengers. One enters the house and stays there until the next Clock step: it cannot ride this round (R-058). The other waits at `v` and can be carried by Vrrooomm! to an empty house elsewhere on the owner's line (R-059).
Source: RB p.2 (Clock: step 2; "or same destination, if you paused time"; "only 1 passenger per building"); C10; D22 (scenario rejection: user decision 2026-11-15).

---

## 13. Vrrooomm!

### R-056 — Vrrooomm! marker resolves
Markers resolve a to f (R-030). A marker does not dwindle: the owner uses **all** their buses in this resolution (R-057). The destination type is the clock's current type, as set by this round's Clock step (R-050), which always runs before Vrrooomm!.
Source: RB p.2 (Vrrooomm!); C6, C11; D22.

### R-057 — Passengers moved: up to one per bus
The owner moves passengers one at a time, up to as many passengers as the owner has buses. Each move takes one eligible passenger (R-058) and delivers it to one eligible building (R-059). The owner chooses which passenger and which building. Each move scores 1 point (R-062). The buses are not physically moved on the map; each bus is "used" once per resolution.
Source: RB p.2 (Vrrooomm!: "Only 1 passenger per bus"); C11.

### R-058 — Which passengers may be carried
A passenger may be moved only if: (a) it is waiting at a node, not in a building; and (b) its node is on the owner's line (R-035). Stations and roundabouts count like any other node. After the Clock step every passenger in a building is in one of the clock's type (R-055), so (a) means that a passenger occupying a building of the clock's type is not eligible to be carried. Only passengers outside such a building are eligible: for example, one at a node whose buildings are of other types, or one at a node whose building of the clock's type is occupied by another passenger. So buildings can block transport: a passenger that stepped in at the Clock step neither rides nor frees its building.
Example: line covers nodes `u`, `v`, `w`. A waiting passenger at `v` qualifies; one at `x` (not on the line) does not; one inside a building at `u` does not.
Example (clock = office): on P0's line, node `u` has one office and two passengers were waiting there at the Clock step: one stepped into the office, the other still waits. Node `v` has only a house and one waiting passenger. Node `w` has an empty office. Both waiting passengers (at `u` and `v`) are eligible; the one in the office at `u` is not. P0 has 2 buses, but only one office is empty, so P0 moves one of them to `w` (R-060).
Source: RB p.2 (Vrrooomm!: listed conditions; Example 1: passengers who stepped into offices stay, the others "need to ride"); C11; D22 (user clarification, 2026-11-15).

### R-059 — Which buildings may receive
The destination must be an empty building of the clock's type whose node is on the owner's line. After the Clock step no such building shares a node with a waiting passenger (R-055), and states that break this are rejected on load, so a passenger is always carried to an empty building of the clock's type elsewhere on the owner's line, never at its own node (the retired R-061 allowed this).
Example: clock = office. A passenger waits at station node `u` on P1's line; an empty office is at node `v` on P1's line; P1 moves the passenger into the office, +1 point.
Example (blocking): clock = house. P2's line covers `u` and `v`. The only house on the line is at `u`, and a passenger stepped into it at the Clock step. A passenger waits at `v`, which has only a pub. P2's Vrrooomm! moves no one: the passenger at `v` is eligible but has no empty house to go to, and the one in the house at `u` is not eligible (R-058, R-060).
Source: RB p.2 (Vrrooomm! conditions; Example 2); C11; D22.

### R-060 — The move count is forced
The number of moves is exactly min(owner's buses, eligible passengers on the owner's line (R-058), empty buildings of the clock's type on the owner's line (R-059)). The owner must move that many and cannot choose fewer, so the only choice is which passengers and which buildings. Every eligible passenger can be carried to every such building (they never share a node, R-059), so this count can always be reached. Ineligible markers (count 0) do nothing.
Example (from the rulebook's second Vrrooomm! example, paraphrased): a seat with 3 passengers and 3 eligible offices on the line but 2 buses moves 2 passengers and scores 2.
Example (playtest game 2b0b32, 3 players, round 1; ids from data/board.yaml, for illustration): setup built offices at nodes n10, n14 and n20 and houses at n13, n18 and n21. P0's Passengers put 2 passengers at station n5; nobody selected Clock. At the Clock step the clock advances house → office. No passenger stands at n10, n14 or n20, so nobody steps in, and Vrrooomm! runs with the clock reading office. P0 (1 bus; line n5–n10–n13): eligible passengers at n5 and n13 (the n13 passenger's building is a house, so it is outside any office); one empty office (n10); P0 moves 1. P1 (1 bus; line n13–n17–n20–n25): eligible passengers at n17 (and at n13 if P0 took the n5 one); empty office at n20; P1 moves 1. P2 (2 buses; line n5–n14–n18): eligible passengers at n5 and n18; one empty office (n14); P2 moves 1. In the reported game the clock stayed at house in round 1, and P0 carried a passenger into the house at n13 while another passenger waited there; under D22 that cannot happen.
Source: RB p.2 (Vrrooomm!: "you must try to use each of your buses"; Example 2); C11; D22.

### R-061 — RETIRED (was: Passenger already at the destination's node)
Retired 2026-11-15 by D22. This entry (a ⚑ default from C11) let Vrrooomm! carry a waiting passenger into an empty building of the clock's type at its own node. Now that the Clock step runs every round (R-050), such a passenger has already stepped in (R-055), and loaded states where it has not are rejected, so the case cannot arise; a passenger is carried only to an empty building of the clock's type elsewhere on the owner's line (R-058, R-059). The ID is kept and is never reused.
Source: D22.

### R-062 — Scoring and effects of a move
Each move places the passenger into the chosen building (building becomes occupied) and gives the owner +1 point. The owner's points can only go up (R-008). The engine also records, for each seat, the move number (the index in the game's decision sequence) at which its points last changed (used by R-070).
Source: RB p.2 (Vrrooomm!: "earn 1 point"); C11, C13.

### R-063 — Destination sites at one node are interchangeable
Empty buildings of the same type at the same node are equivalent for every later rule. The engine may canonicalise the choice of destination site among them (ACTIONS.md decides).
Source: derived from R-055, R-058, R-059; C11.

---

## 14. Starting player and round end

### R-064 — Starting Player action
When a Starting Player marker resolves, its owner becomes the start player for the next round (they take the Start bus). If no marker is on Starting Player, the start passes to the next seat after the current start player (ascending seat number, wrapping).
Example (4 players): P2 started this round, no one took the action: P3 starts next round. If P1 took it: P1 starts next round.
Source: RB p.2 (Starting Player: "pass the Start bus clockwise"); C12, R-003.

### R-065 — Spent markers are removed
After Phase 2 every Action marker placed this round is removed from the game; markers are never recovered. A seat's markers-left count only decreases (from 20).
Source: RB p.2 (End of Phase 2: "Return all spent Action markers to the box"); C4, C12.

---

## 15. Game end, scoring, tiebreaks, returns

### R-066 — The three end conditions
The game ends when any of these conditions holds, checked as stated:
1. **Sites full:** every site has a building. The game ends at the end of that round, after Starting Player resolves and spent markers are removed (R-049, R-065).
2. **Last time stone:** the last stone is taken from the clock. The game ends immediately (R-053).
3. **Markers:** at the end of a round (after R-065), at most one player still has any Action markers ⚑. The rulebook says "only 1 player"; 0 is treated the same.
Sites-full and markers can both hold at the same round end. The recorded end type is the set (list) of all conditions that held; the time-stone end is immediate, so it is always alone in the set.
Source: RB p.2 (Phase 2 cleanup; End of the Game); C12 (⚑ the 0 case), C13.

### R-067 — Timing of the end checks
Conditions 1 and 3 are checked only at round end, after Starting Player resolves and markers are removed; condition 2 can occur mid-round. When a game ends at round end, no further round starts and no additional marker resolves.
Source: RB p.2 (End of the Game); C13.

### R-068 — Final score
Final score = points − time stones held. Stones never move the score disc.
Example: P1 has 12 points and took 2 stones: final score 10.
Source: RB p.2 (Clock, End of the Game); C13.

### R-069 — Winner and tiebreaks
The player with the highest final score wins. If tied, among the tied players the one with the most time stones wins. If still tied, the tied player who reached that points total first wins (R-070). A tie that survives all of these is a shared win.
Example: P0 and P2 both have final score 9; P0 took 2 stones and P2 took 0. P0 wins at the first tiebreak.
Source: RB p.2 (End of the Game); C13. The shared win for surviving ties is confirmed by the user (Q3, 2026-11-14).

### R-070 — "Reached the points total first"
Among players still tied after the stones tiebreak (so they also have equal points), the one whose points last changed at the earliest move number (R-062) wins. A player whose points never changed has no move number and is treated as having reached 0 points at the same instant as every other zero-point player, so a tie among zero-point players is shared (R-069). The rulebook's stack order on the score track is read as this move-number comparison.
Example: P1 and P3 tie on points and stones with 8 points; P1's last scoring move was move 340, P3's was move 412. P1 wins.
Source: RB p.2 (End of the Game: "whose score marker is lowest in its stack"); C13. The move-number reading is confirmed by the user (Q3, 2026-11-14).

### R-071 — Returns: `win_loss`
With n players and k players tied for the win after all tiebreaks (R-069), each winner receives (n − k) / (k · (n − 1)) and every other seat receives −1/(n − 1). A sole winner (k = 1) receives +1. Returns sum to 0 (zero-sum).
Example (n = 4): sole winner +1, others −1/3 each. Two winners: (4 − 2) / (2 · 3) = +1/3 each, the other two −1/3 each (sum: 2/3 − 2/3 = 0).
Source: D7.

### R-072 — Returns: `margin`
Each seat's return = (own final score − mean of the other seats' final scores) / 10, clipped to [−1, 1]. The game is general-sum under this setting.
Example (n = 3): final scores 14, 8, 5; seat 0: (14 − 6.5)/10 = 0.75.
Source: D7.

---

## 16. Information and chance

### R-073 — Perfect information, no chance
Every state element is public (board, lines, buildings, passengers, clock, stones, markers on the action board, markers left, scores). There are no chance nodes: the game is deterministic given the players' decisions. Seat 0 always starts the first round (R-019).
Source: D5; §5.2.

---

## Open questions

None open.

### Resolved questions
- ~~O-1~~ (R-007) resolved 2026-11-14: building supply is unlimited for every type, as C2 (user confirmed).
- ~~O-2~~ (R-038, R-042) resolved 2026-11-14: a multi-extension Lines marker is evaluated one street at a time; a street added earlier in the same resolution counts as occupied (user confirmed).
- ~~O-3~~ (R-027) resolved 2026-11-14: a player who runs out of markers during the mandatory passes is skipped and out of the round, as C4 (user confirmed).
- ~~Q2~~ (R-026, R-038, R-039) confirmed 2026-11-14: C5's per-end placement rule and C4's any-free-space selection (user confirmed).
- ~~Q3~~ (R-069, R-070) confirmed 2026-11-14: move-number tiebreak and shared surviving ties (user confirmed).
- ~~O-4~~ (R-070) resolved 2026-11-14: zero-point ties are shared, same decision as Q3 (user confirmed).
- ~~Q-A~~ (R-033) resolved 2026-11-14: with 5 players the Lines +1 is added first, then the power ≤ 0 check (user decision).
- ~~Q-B~~ (R-034, R-040, R-042) resolved 2026-11-14: Lines markers are resolved maximally; only choices that keep the maximum achievable number of remaining placements reachable are legal (user decision).

### Amendments
- 2026-11-14 (Phase 1 unit 9, user-approved): R-049 and R-066 say "every site has a building" instead of "every site is occupied" ("occupied" means a passenger inside, R-062); R-046 punctuation. Wording only, no change in meaning.
- 2026-11-15 (D22, from playtest reports 2b0b32 and a22926; user rulings): the Clock step runs every round, including round 1 and the sites-full round, at the Clock slot (after Buildings, before Vrrooomm!), and advances automatically without a Clock marker; a Clock marker's owner pauses (takes a stone) or advances (no stone) instead (R-029, R-049 example, R-050 to R-055). Starting passengers first step in at the round-1 Clock step; ⚑ removed (R-015). Vrrooomm! carries only passengers outside a building of the clock's type, to an empty building of that type elsewhere on the line; loaded states that break the post-Clock-step invariant are rejected (R-055, R-056 to R-060). R-061 retired. Sources note updated (item 3).
