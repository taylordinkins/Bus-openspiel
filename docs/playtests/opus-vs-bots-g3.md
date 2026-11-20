# Opus vs bots, game 3 (seed 13)

Game file: `runs/agent_play/opus_g3/game.json`. Opus plays P2.
P0 = `gumbel:64-sup-rank`, P1 = `net:ppo-long`, P3 = `gumbel:64-sup-leader`.

## 1. Setup and plan

- Setup buildings before me: offices n10 (P0), n13 (P0); house n18, pub n21 (P1). After me P3 added office n13 and pub n21.
- Mine: office at n14 (site 1) and pub at n18 (site 7). Idea: round 1 auto-advances to office (R-050/R-051), so the n18 roundabout passenger (house there, no office) waits and can ride to an empty office at n14; in round 2 (pub) it steps out at n14 (R-054) and can ride back to my n18 pub.
- Lines: started s33 (n18-n14); P3 then took s41+s31 and P1/P0 piled onto the n13-n18-n21 corridor. Second marker s41 to n21 so I had two passenger sources (n18, n21) instead of one contested one.
- Plan: be first on Vrrooomm! whenever my line holds a passenger that others can also reach; buy buses; get to a station (n5 via s14) before others; avoid feeding stations others touch.
- Observation at setup: all four lines ended up stacked on n13-n18-n21. The 4 roundabout passengers and the 3 setup offices are all in that corridor, so early points are a pure Vrrooomm! letter race.

## 2. Turn log

- R1 (office): took Vrrooomm! a first (P0 already had Lines a, P1 Buses). Then Buildings a (power 2 after P1's bus): pub n26, house n20 (future targets off the crowded corridor). Delivered n18 -> n14 office (+1). P0 delivered n21 -> n13. P1 wasted Lines b (power 0). Score P0 1, P1 0, P2 1, P3 0.
- R2 (pub): Vrrooomm! b (behind P1's a) plus Clock to stop a P0/P3 pause that would have put my n14 passenger back into its office. Advanced. P1 (2 buses) moved both n13 passengers to pubs; I moved n14 -> n21 pub (+1). P0 Lines d power -1 (wasted); P1 Lines c power 0 (wasted). Score 1/2/2/0.
- R3 (house): no house target on my line, so took Buses (first to 2 buses was P1/P0; I caught up). Mandatory 2nd pick Clock (every other pick fed opponents: forced builds at n17 on P0/P3 lines, passengers at stations P0/P3 could reach). Advanced (pause would only deny trailing P3 1 at my cost of 1). P0 Lines a took s13+s14 to n5 and onto MY n14 end. P3 n17 -> n20 house (+1). P3's second Vrrooomm! wasted. Score 1/2/2/1.
- R4 (office): Vrrooomm! b behind P0's a, plus Clock. 4 passengers (n18, n20, n21x2), 4 offices (n10, n13x2, n14). P0 (a) filled both n13 offices; I got n21 -> n14 (+1); P3 (c) found nothing (n13 full, n10 not on its line); P0's d had nothing. Score 3/2/3/1.
- R5 (pub): P1 opened with Vrrooomm! a. I took Vrrooomm! b, then Clock (a pause to stay at office would have zeroed P1's 2 and my 1). P0 Lines c power 0 (wasted). P1's Lines a went n21-n22-n14 onto my n14 passenger, but P1 used its 2 buses on the n13 pair; I moved n14 -> n21 pub (+1). Score 3/4/4/1.
- R6 (house): only empty houses after the Clock step would have been none (n18, n20 refill). Built my own: Buildings a (power 3; 2 forced on n17, made them pubs) + house at n27 (B), Lines c (power 1, resolved first) s48 to n27, Vrrooomm! d. n21 -> n27 house (+1); P3's a/c and P1's b scored 0 (no house on their lines). Score 3/4/5/1.
- R7 (office would be next): P0 had Vrrooomm! a and then Buses (4 buses); its line held 3 of the 4 waiting passengers (n18, n20, n21) and 4 offices. I took Clock, Lines b (n27-n26-n33 using the R-039 exception at P3's n26 end), Buildings a (power 4: office at n33, 3 houses off all lines). PAUSED (stone 1): stays house, every house refilled, nobody scored. Denied P0 3. Score 3/4/5(st1)/1.
- R8 (office): Vrrooomm! b behind P1's a; Lines a (power 4, resolved last) n33-n32-n25 and n14-n15-n5 (both stations); Passengers a (power 4) 4 to n25. P0 Lines c reached n33 (my office) via n26. P1 (3 buses after Buses) moved n21 -> MY n14 office and n18 -> n13, so I had one office (n33): n25 -> n33 (+1). P0 n20 -> n10. Score 4/6/6(st1)/1.
- R9 (pub, last round: P0/P3/me down to 0-1 markers, mandatory pass): start P3 picked Vrrooomm! a with its last marker, P1 b; I was 4th to pick, only Vrrooomm! c could score. P3 (3 buses) took n33 -> n21, n13 -> n18, n25 -> n26; P1 took n10 -> n21. All 4 pubs on my line were gone: 0. Game ended on markers.

## 3. Delivery attempts (mine)

| Rnd | Setup | Payoff | Why |
|---|---|---|---|
| R1 | Built office n14 + line n14-n18; Vrrooomm! a | +1 (n18 -> n14) | Only I had n14; first letter meant P0 could not take the n18 passenger first. |
| R2 | Pub at n18 from setup; Vrrooomm! b + Clock | +1 (n14 -> n21) | P1's a used 2 of 3 corridor pubs; my passenger at n14 was private until P1 reached n14 later. |
| R3 | none | 0 | No empty house on my line; bought a bus instead. |
| R4 | Vrrooomm! b | +1 (n21 -> n14) | P0 (a) filled the two n13 offices, not n14. |
| R5 | Vrrooomm! b + Clock | +1 (n14 -> n21) | 3 corridor pubs, P1 took 2. |
| R6 | Built house n27 + Lines c to it + Vrrooomm! d | +1 (n21 -> n27) | Private target built in the same round (Lines/Buildings resolve before Clock and Vrrooomm!, R-029); letter order irrelevant because no one else had a house. |
| R7 | (paused instead) | 0 | See section 4. |
| R8 | Lines a to n25, 4 passengers there, offices n14 + n33 | +1 (planned 2) | P1 put its passenger into my n14 office first; I had 2 buses but one target. |
| R9 | Vrrooomm! c | 0 | Picked 4th in the last round; P3 and P1 consumed all 4 pubs on my line. |

Total 6 points from 7 attempts. Every delivery came from being an earlier letter or from a private target.

## 4. Delivery preventions (mine)

- R1, R4, R5: earlier Vrrooomm! letter on a passenger that P0/P1/P3 could also reach. Worked (each time I scored the contested one).
- R2, R4, R5: took Clock to remove the pause option from opponents. No one could test it; cost 1 marker each time.
- R7: paused with the Clock (R-052). P0 had Vrrooomm! a with 4 buses and 3 eligible passengers / 4 offices on its line; at house every house refilled on the pause (R-054/R-055) and no empty house sat on any line. Worked: P0 0 instead of 3, P3 0, me -1 (stone). In hindsight it did not change the winner: final gap to P1 was 2 (7 vs 6-1); without the pause I would have had 6 or 7 points but P0 would have reached 6 and had a 4-bus pub round in R8.
- R8: put the 4 Passengers on n25 (my line + P3, who had no Vrrooomm! that round), not n5 (on P0's and P1's lines). Worked that round, but 2 of them were still at n25 in R9 and P3 carried one for a point.
- R6: forced n17 builds made pubs (two rounds away) rather than offices/houses on P0/P3's lines. Worked for R6-R8 (no opponent gained a target there), but in R9 (pub) P3 left the n17 pubs empty and took n18/n21/n26, the pubs I shared, which is what zeroed my R9.

## 5. Opponents' deliveries and preventions

- P1 (ppo-long, sampled): 7 points, winner. 2 in R2, 2 in R5, 2 in R8, 1 in R9. Its strength was always holding Vrrooomm! a with 2-3 buses on the shared corridor (n13 offices <-> n18/n21 pubs), picking first by being start player or picking Vrrooomm! first. R8: delivered into MY n14 office (deliberate or not, it cost me 1). Wasted Lines b/c at power 0 three times (R1, R2, R3).
- P0 (gumbel sup-rank): 4 points. Strong R3 Lines (n10-n5-n14 onto my end) and R4 (2 passengers into n13, leaving me n14). Wasted Lines d (power -1) in R2 and Lines c (power 0) in R5; took Buses in R9 with its last marker (no effect on score: game ended). Never picked Clock. Its R7 Vrrooomm! a with 4 buses was neutralised by my pause.
- P3 (gumbel sup-leader): 4 points, 3 of them in the last round. Double Vrrooomm! in R3 (one wasted) and R6 (both 0). Last round: with its last marker it opened Vrrooomm! a and took the 3 pubs that also served me and P1 — whether aimed at the leaders or just greedy, it decided P1 vs me.
- No bot ever paused. No bot blocked my Lines deliberately; P0's R8 extension to n33 and P1's R5 extension to n14 both put them on my targets.
- Bots took Starting Player often (P3 R1/R2/R5, P0 R3/R9, P1 R4/R6/R8); the start seat effectively decides who gets Vrrooomm! a.

## 6. Final result

Game over at the end of R9 (markers: only P1 had any, R-066). Points P0 4, P1 7, P2 6, P3 4; P2 held 1 stone. Final P0 4, **P1 7 (winner)**, P2 5, P3 4. Replay: `runs/agent_play/opus_g3/game.replay.json`.

## 7. Heuristic takeaways

1. **Vrrooomm! letter = ownership of shared passengers.** If a passenger on your line is also on an opponent's line, take Vrrooomm! before that opponent, ahead of every other pick. Letters go a first, earliest empty letter (R-025, R-030). *Confirmed* (R1, R2, R4, R5 all scored this way; R9 failed when I picked 4th).
2. **Private target in the same round.** Lines and Buildings resolve before the Clock step and Vrrooomm! (R-029), so a Lines unit onto a node + a building of next round's type T' on that node creates a target nobody else can use that round. Check that the node has no passenger that would step into it (R-055). *Confirmed* (R6: +1 while 3 opponent markers scored 0).
3. **Forced builds: type that is furthest in the cycle, or off all lines.** Under automatic advance, a building of type T (current) is next used 3 rounds later (R-050, R-051). Builds must go on the lowest open tier (R-010, R-047, R-048). *Confirmed as harmless in R6/R7*, but see 9.
4. **Pause when one opponent's single Vrrooomm! marker would take 3+ and you get ≤ 1.** Pause keeps the type; every passenger in a building re-enters at its node (R-052, R-054, R-055); waiting passengers stay eligible. Count empty T-buildings on each line after the pause. *Confirmed* (R7: P0 denied 3 for my 1 stone), but the stone counted in the final margin (R-068) and the beneficiary was the 3rd seat (P1). Speculative refinement: pause only if the seat you deny is your closest rival, or if the denied count ≥ 3 and the game has ≥ 3 rounds left.
5. **Hold the Clock as a cheap 2nd mandatory pick** when no other pick has positive value; it removes the opponents' pause option (R-050, R-052) at the cost of one marker. *Confirmed* as safe; value not demonstrated (bots never paused).
6. **Do not feed stations on an earlier Vrrooomm! opponent's line.** The Passengers owner chooses the station per passenger (R-045); any line through it may carry them (R-058). Put them on the station only you and Vrrooomm!-less opponents touch, and expect leftovers to feed others next round. *Confirmed* (R8 safe; R9 P3 used one leftover).
7. **Your delivered passenger becomes everyone's.** It waits at the destination node after the next Clock step (R-054). If that node is on an opponent's line, they can carry it next round. Prefer destinations only you touch. *Confirmed* (R9: P3 carried my n33 passenger away).
8. **Opponents' targets are not reserved.** An opponent may deliver into "your" building if it is on their line too (R-059); P1 filled my n14 in R8. Count only targets no earlier-letter opponent can reach. *Confirmed*.
9. **Last-round planning starts two rounds out.** When you will have 1 marker in the final round and pick late (not start player), you get one late Vrrooomm!. Either keep 2 markers, or take Starting Player the round before (R-064) so you pick first in the last round. *Speculative* (R9 lost me 1-2 points; I spent 3 markers in R8 on Lines/Passengers/Vrrooomm!).
10. **Lines power vs letter:** Lines a in a round where someone holds 4 buses is worth 4 streets (R-031, R-032), enough to reach a station from 2 nodes away and a second goal from the other end. Lines c at M=2 is wasted. Bots wasted 5 Lines markers at power ≤ 0. *Confirmed*.
11. **Shared-end exception as a shortcut:** when an opponent's end sits at your end's node, their last street is open to you even with free streets around (R-039). R7: s59 to n33 after P3 ended on n26. *Confirmed*.
12. **Start-player seat = first Vrrooomm! choice.** The start player picks first and usually takes Vrrooomm! a (R-023, R-025). Taking Starting Player is worth a marker when next round has a big contested delivery. *Speculative* (bots took it often; P1 won mostly via a-slots).
13. **Crowded corridor at setup = letter race.** If all lines start on the roundabout corridor, the early game is decided by letters, not routes; build your second setup building as a private target elsewhere and branch out by R3. *Speculative*.

Tool friction: none blocking. Clock choice is offered as ids 291/292 without stating the resulting type in the label; the summary's Vrrooomm! estimate only covers "advance" and ignores this round's Lines/Buildings, so pause decisions had to be computed by hand.
