# Rules audit: opus-vs-bots-g1 notes

Audit of the rules claims in `docs/playtests/opus-vs-bots-g1.md` against `docs/RULES_SPEC.md` (APPROVED), the engine (`bus/rules.py`, `bus/lines.py`) and the replay (`runs/agent_play/opus_g1/game.json`, 249 moves, re-run through `tools.agent_play.build_state`). Move numbers below are history indices.

**Spec vs engine:** no disagreement found. Every claim checked in the replay matches the spec: resolution order and dwindling power (R-029 to R-033), the shared-end exception (R-039), the "all candidates occupied, so any candidate" case (R-038), the pause (R-052 to R-055), Vrrooomm! counts (R-060), the markers end (R-066) and tiebreaks (R-068, R-069).

Verdicts: CORRECT, INCORRECT, IMPRECISE (true here, but a condition is missing), NOT A RULE (strategy; any rules facts it relies on are correct).

## 1. Heuristic takeaways (§7)

| # | Claim (short) | Verdict | R-IDs | Correction |
|---|---|---|---|---|
| 1 | The clock auto-advances house→office→pub; before Clock, pre-build next type at nodes with no waiting passenger | IMPRECISE | R-010, R-029, R-047, R-050–R-055 | It auto-advances only if no Clock marker was placed. If someone holds the Clock marker, they pick pause or advance. Builds must go on the lowest open tier, so you can't freely pick "private nodes". A new building at node v stays empty only if v has fewer passengers *in total* (waiting plus inside any building) than buildings of the new type: everyone at v steps out and then steps into new-type buildings at v. Passengers step out at their own node, not "next to" your buildings. |
| 2 | Deliveries = min(buses, eligible passengers, empty targets on line) | IMPRECISE | R-058–R-060, R-047 | The formula is exact (R-060), but it is evaluated when each marker resolves, after earlier Vrrooomm! markers have taken passengers and filled buildings. "Eligible" means waiting at a node on your line. "Targets" means empty buildings of the clock's type after this round's Clock step. A private target needs a lowest-open-tier site on your line, at a node that won't fill it at the Clock step (see 1). |
| 3 | Vrrooomm! slot priority: only seats that can reach your passengers or targets matter | IMPRECISE | R-025, R-029, R-030 | Letters follow selection order (earliest empty letter, a resolves first). "Can reach" must be judged *after this round's Lines*, which resolve before Vrrooomm!. In R5, P3's Lines b reached n4 via s11 (move 124), and P3's Vrrooomm! b then took P0's n4 passenger (move 132). |
| 4 | Beware the straddling double Vrrooomm! | NOT A RULE | R-030, R-056, R-057 | The rules basis is right: one seat may hold several Vrrooomm! markers, and each one uses all of its buses. |
| 5 | A late Lines letter resolves first; block the opponent's only unoccupied exit; don't leave your end at the contested node (shared-end exception) | IMPRECISE | R-030, R-031, R-033, R-037–R-039, R-042 | The order (f..a) and power (M−k, +1 at 5 players) are right, and so is the warning about the shared-end exception: at move 99, P3 followed onto P0's own s12 at n4 because P0's end sat there. But occupying the *only* unoccupied candidate at an end leaves every candidate occupied, and R-038 then lets the opponent take **any** candidate, including your street. At move 221, P2 at n4 took P0's mid-line s12 this way. A block works only if at least one *other* unoccupied candidate remains at that end. It must hold at every node the opponent can reach within its power, from either end. |
| 6 | Lines power = M_now − k; a Lines c at M=2 does nothing; same for Buildings/Passengers | IMPRECISE | R-031–R-033, R-043, R-045, R-047 | Lines is right for 3–4 players; at 5 players add +1. Passengers and Buildings resolve *after* Buses, so a Buses marker this round can raise M for them. Units are also lost when the passenger supply is empty or no sites are left. |
| 7 | Pausing (−1 stone) puts every passenger back into its building, so all opponent Vrrooomm! markers are wasted; the stone is the first tiebreak | INCORRECT (mechanism) | R-050, R-052–R-055, R-058, R-060, R-068, R-069 | Pausing only re-runs step-out/step-in at the same type. Passengers that were in buildings re-enter. **Waiting passengers stay waiting and remain eligible.** After the R3 pause (move 84), 2 passengers still waited at n5 on P3's line. Opponents scored 0 only because no empty pub was on any of their lines. The stone costs 1 final point, and the tiebreak only helps when final scores are equal. Only the holder of the single Clock space can pause. A pause that takes the last stone ends the game immediately (R-053). |
| 8 | Deliver into shared targets first, keep private ones | NOT A RULE | R-054, R-055, R-057 | Rules fact behind it: a delivered passenger stays at that node. At every later Clock step it re-enters a building of the clock's type there, so a target you fill now is usually full again the next time its type comes round, unless someone carries the passenger off. |
| 9 | Don't feed shared stations | NOT A RULE | R-045, R-046, R-058 | Basis is right: the Passengers owner picks either station for each passenger. Anyone whose line touches it can carry those passengers. |
| 10 | The game ends at round end when ≤1 seat has markers; `am` is already net of this round's placements | CORRECT | R-024, R-027, R-065–R-067 | `am` drops when a marker is placed (`bus/rules.py:345`), and the end check is `bus/rules.py:733`. The "0 seats" case counts too (⚑). A seat at 0 markers is skipped for the rest of the game. |
| 11 | A looped/short line has 1–2 exits; one Lines unit can deny a whole power-3 extension | IMPRECISE | R-037, R-038, R-040, R-041 | Other players' markers never remove candidates (only your own streets do, R-037), so you can't make an end dead or cancel an extension. P3 still placed all 3 streets in R8 (moves 215–217). You can only *redirect* it, and only while an unoccupied candidate remains (move 216: legal set {s7}). |
| 12 | Buses early pay off only with enough targets | NOT A RULE | R-032, R-043, R-060 | Basis is right: a bus raises your delivery cap, not your deliveries. It raises M (and so opponents' Lines/Passengers/Buildings power) only if your new count exceeds the current maximum. |
| 13 | Forced builds: pick the type furthest away, or build off all lines | NOT A RULE | R-034, R-047, R-048, R-050 | You must build every unit, on the lowest open tier, re-evaluated each build. "Off all lines" is only possible if such a site exists in that tier. Type arithmetic: see corrected rule 13. |

## 2. Other rules claims in the notes

| Where | Claim | Verdict | R-IDs | Note |
|---|---|---|---|---|
| §1 | Round 1's Clock step auto-advances house→office, so the first scoring type is office | IMPRECISE | R-015, R-050, R-052 | Unless a Clock marker owner pauses (then house). |
| §1 | Station n5: "Passengers action refills it" | IMPRECISE | R-045 | Only if the owner sends passengers there and supply remains. |
| §1 | Each bus = +1 delivery per Vrrooomm!, raises M for everyone | IMPRECISE | R-032, R-060 | +1 to the cap only. M rises only if you become the new maximum. Lines this round never sees it. |
| §2 setup | P1's 2 offices at n13 swallow its starting passenger in round 1 | CORRECT | R-015, R-055 | Given the automatic advance (move 34 onward). |
| §2 R1 | P1's Vrrooomm! a scored 0, no empty office on its line | CORRECT | R-055, R-060 | n18's starting passenger had stepped into P3's office there. |
| §2 R3, §4 R3 | The pause made all 4 opponent markers score 0 "by putting every passenger back into the pub" | Outcome CORRECT, cause INCORRECT | R-054, R-055, R-060 | See takeaway 7 (n5 still had 2 waiting passengers). |
| §2 R4 | P3's Lines ran n5-n4-n10-n5 over P0's area | CORRECT | R-039 | Move 99 was the shared-end exception onto P0's last street s12 (P0's end was at n4). This is the trap takeaway 5 warns about. |
| §2 R5 | Lines c had power 1 | CORRECT | R-031, R-032 | M=3 during Lines. P2's Buses resolved after Lines. |
| §2 R6/R8 | c resolved before a/b, with powers 2 vs 4 and 2 vs 3 | CORRECT | R-030, R-031 | M=4. |
| §4 R6 | P2 had no unoccupied-rule escape and no shared-end exception onto s29 | CORRECT in principle | R-038, R-039 | Reasoning matches the rules (not re-derived as a counterfactual). |
| §4 R8 | s8 and s9 forced P3 onto s7 | CORRECT | R-038 | Move 216: only s7 legal. It worked because s7 was unoccupied (takeaway 5). |
| §4 R8 | Buildings b resolved before P1's a | CORRECT | R-030 | Highest letter first. |
| §5 | P3's line looped, both ends at n4, one exit | CORRECT | R-037, R-041 | Move 215: only s6 legal, from either end. |
| §5 | P2 dropping with 3 markers in R8 kept the game alive | CORRECT | R-066 | P1 (1) and P2 (3) both held markers. Had P2 held 0, only P1 would have, and the game would have ended. |
| §6 | 9/7/8/6 points, P0 has 1 stone, final 8/7/8/6, P0 wins on stones | CORRECT | R-068, R-069 | Engine `winners` = (0,). End type `('markers',)`: all seats at 0 (⚑ case). |

## 3. Corrected heuristic rules (condition → action)

Notation. T is the clock type when Buildings resolves. T′ is the type for this round's Vrrooomm!: T+1 if no Clock marker is placed or its owner advances, T if the owner pauses. pass(v) is all passengers at node v, waiting plus inside buildings. M is the maximum bus count of any seat.

1. **Pre-build:** if your Buildings marker has power ≥1 and T′ is known (nobody holds Clock, or you do), consider each empty site of the lowest open tier (re-evaluated per build) whose node v is on your line after this round's Lines. Build type T′ there when pass(v) < (T′-buildings at v, counting the new one); only then is it a target this round. The passengers that become eligible are the waiting ones at line nodes u where pass(u) exceeds the T′-buildings at u (at a station, all of them).
2. **Delivery estimate:** for each Vrrooomm! marker, in letter order, compute min(buses, waiting passengers on the line, empty T′-buildings on the line) on the state after the Clock step. Apply each earlier marker's moves before computing the next one (`bus.rules.vrrooomm_moves`).
3. **Slot priority:** a seat counts as a threat if its line can touch your passengers or targets after this round's Lines. Assume a Lines marker at letter k adds up to M−k streets (+1 at 5 players) from either end, legal under R-038/R-039.
5. **Block with an earlier-resolving Lines marker:** to stop opponent O's end at v from using street s, all four must hold when O evaluates v:
   - s is occupied;
   - at least one other candidate at v is unoccupied;
   - no line (yours or a third player's) has an end at v whose last street is s, unless O's end arrives on s;
   - the block holds at every node O reaches within its power, from either end (O must resolve maximally, R-042).

   Never finish your own Lines with your end on a node O's end can reach.
6. **Power check before picking:** Lines power is M−k (+1 at 5 players), using the M at selection time. Passengers/Buildings power is M′−k, where M′ is M raised by any Buses marker this round whose owner then exceeds M. Skip a pick whose power is ≤0, or whose supply (passengers, sites) is empty.
7. **Pause:** only possible if you hold the Clock marker. At Clock resolution, compute every seat's Vrrooomm! count (rule 2) both ways: advance (T+1) and pause (T). Remember that waiting passengers stay eligible after a pause. Pause when the drop in opponents' deliveries minus your own lost deliveries is worth more than 1 final point. If stones_on_clock == 1, pause only if the immediate end (R-053, with standings as they are now) wins for you.
10. **Markers end:** during selection, project each seat's `am` after this round. If at most one seat will have am > 0, or every site has a building, this is the last round. Keep ≥1 marker unless you want the game to end.
11. **Short/looped lines:** count O's candidates at each end (incident streets minus O's own). You can't remove them. Only redirect, using rule 5.
13. **Forced extra builds:** build them on lowest-open-tier sites not on any line, if any exist. Otherwise prefer type T: under automatic advance it is used two rounds from now (T+1 this round, T+2 next, then T). Avoid T′ on any opponent's line node where pass(v) < T′-buildings.

(4, 8, 9, 12 are strategy and need no rules correction; 10 is already correct.)

## 4. Open questions

None. Every rule the takeaways depend on is settled by the spec (R-038/R-039 per-end evaluation, the R-042 maximal resolution, the R-066 zero-marker ⚑ case).
