# Opus vs bots, game 1

## 1. Setup
- Seats: P0 = Opus (agent), P1 = net:ppo-long, P2 = gumbel:64-sup-leader, P3 = gumbel:64-sup-rank. Seed 11, 4 players, game file `runs/agent_play/opus_g1/game.json`.
- Opening plan: round 1's Clock step auto-advances house -> office, so the first scoring type is office. Build both setup buildings as offices on tier-A sites next to station n5 (n10, n14), then start a line in the n13-n10-n5-n14-n18 corridor: it touches two roundabouts with starting passengers (n13, n18), the station n5 (Passengers action refills it) and both offices. Take an early Vrrooomm! slot in round 1 to score before others empty shared nodes. Then: Buses early (each bus = +1 delivery per Vrrooomm!, and raises M for everyone), build buildings of the type the clock will show next on my own line, and use Clock (pause/advance) when it turns a forced 0-delivery round for opponents into a delivery round for me.

## 2. Turn log
- Setup: built offices n10, n14 (A sites by station n5). P1 built 2 offices at n13 (roundabout: its starting passenger will be swallowed by the round-1 office clock). P2 house n18 + pub n21, P3 office n18 + pub n21. Lines: me s30 (n13-n17) then s23 to n10; P1 n18-n21-n17; P2 and P3 both n13-n18-n21.
- R1 (start P0): Buses (first pick), Vrrooomm b, Passengers a (2 to n5). Clock auto -> office. P1's Vrrooomm a scored 0 (no empty office on its line). I delivered n17 -> n13 office (deliberately the shared office, denying P3). P2 delivered n21 -> n26 office (P1 had built it on P2's line). Score P0 1, P1 0, P2 1, P3 0.
- R2 (start P3): Buildings a, Vrrooomm b. Built 2 pubs on the last tier-A sites (n17, on my line). Clock auto -> pub; n13's 2 passengers stepped out; my 2 buses carried both into my n17 pubs. P1 n18->n21 pub, P2 n26->n21 pub. P3 Lines 2 streets, P2 Buses. Score 3/1/2/0.
- R3 (start P3): Buildings a, Lines b (s12 to n4), Clock. Built houses n4 (on line) and n12. P1 took Vrrooomm a+b, P2 c, P3 d; I had no Vrrooomm slot that could beat P1 to my n17 passengers, so I PAUSED at pub (stone 1): all 4 opponent Vrrooomm markers scored 0. P3 Lines ran n22-n14-n5 (reaching my station passengers). Score 3/1/2/0, me 1 stone.
- R4 (start P2): Vrrooomm b, Clock (mandatory 2nd pick). P3 Lines a+b (3 streets) ran n5-n4-n10-n5: now overlaps my n4 house and n10 office. Clock advance -> house: P2 n21->n18, me n17->n4 house (before P3's d), P1 n17->n20. Score 4/2/3/0.
- R5 (start P1): Vrrooomm c, Lines c (power 1: s3 to n1, heading for n5). P3 held Vrrooomm b AND e (2 buses): b took n21 and MY n4 passenger, e took both n5 station passengers -> P3 +4; my c found nothing left. P1 Lines ran to n25, delivered n20->n26. Score 4/3/3/4.
- R6 (start P3): Vrrooomm b, Lines c (power 2, resolved before P2's power-4 a). Pub round: I took s29+s21 (n17-n12-n8) so P2 could not route n13-n12-n17 onto my pubs. P2 (Vrrooomm a, 4 buses) took both n13 passengers to n21 pubs; I carried n10->n17; P1 filled my other n17 pub with n26's passenger. Score 5/4/5/4.
- R7 (start P3): Vrrooomm b, Buildings a (power 4: house at n8 on my end + 3 houses off every line). House round: P3 (a) filled my n4 house as expected; my b delivered both n17 passengers to n8 and n12 houses. P2 n21->n20. Score 7/4/6/5 (me 1 stone).
- R8 (start P1): Vrrooomm c, Lines c, Buildings b, Vrrooomm d (all 4 remaining markers). Lines c resolved first: s8+s9 (n8-n3-n9) blocked P3's power-3 Lines route n4-n3-n8 to my n8 passenger; P3 stopped at n7 exactly as predicted. Buildings b resolved before P1's a: office on n9's B site. Office round: P1 n20->n10, P3 n4->n13, me n8->n9 office and n12->n13 office. I emptied my markers believing P1 was also at 0 (it had 1 left), so the game did NOT end. Score 9/5/6/6.
- R9 (start P2): I had no markers. P2 (3 markers: Vrrooomm, SP, Buses) carried n14, n13 -> n21 pubs; P1 (last marker, Vrrooomm) carried n10 and n13 passengers into MY n17 pubs. Game ended on markers. Points 9/7/8/6.

## 3. Delivery attempts (mine)
- R1: line n10-n13-n17, 1 bus, Vrrooomm b behind P1's a (P1 had no empty office). Office clock left n17's passenger waiting and one of n13's two offices empty. Paid off (+1). I chose the shared n13 office over my private n10 office to deny P3.
- R2: Buildings a: pubs on both remaining tier-A sites (n17, on my line), knowing the auto clock goes office -> pub and n13's two office passengers would step out next to my line. Vrrooomm b (after P1, who could not reach n13). Paid off (+2, both buses used).
- R3: houses at n4 (on line) and n12 for the house round, but P1 had Vrrooomm a+b ahead of me, so I paused instead (see §4). No delivery.
- R4: Vrrooomm b, n17 -> my n4 house just before P3 (Vrrooomm d), whose Lines had just run through n4. Paid off (+1). P1 took the other n17 passenger to n20.
- R5: Vrrooomm c for my n4 passenger to n10/n13 offices. FAILED: P3 had Vrrooomm b and e, swept 4 passengers incl. mine. My line had 1 eligible passenger and shared targets; Lines c (power 1) could not reach station n5 in time.
- R6: pub round, Vrrooomm b; passengers n10, n13 x2 eligible, my n17 pubs x2. Partial (+1): P2 (a, 4 buses) took both n13 passengers, P1 used my 2nd pub.
- R7: house round, Vrrooomm b, built a house at n8 (my line end) to have 2 private houses (n8, n12) after P3 took n4. Paid off (+2).
- R8: office round, Vrrooomm c+d; line extended to n9 + office built there so that, after P1 and P3 filled two of the shared offices, I still had 2 (n9 private, n13). Paid off (+2).


## 4. Delivery preventions (mine)
- R1: delivered into the shared n13 office instead of my private n10 office; P3's only empty office. Worked: P3 scored 0 (P2 still scored via an n26 office P1 had built on P2's line).
- R3: Clock pause (stone -1). P1 had Vrrooomm a+b, P2 c, P3 d; with the auto advance to house P1 would score 2 (n18/n20 houses) and I had no Vrrooomm. Pausing at pub put every passenger back into the pub at its node: 4 opponent Vrrooomm markers scored 0. Worked; the stone later won me the tiebreak (8-8 with P2).
- R6: Lines c (resolving before P2's power-4 Lines a) on s29 then s21 so my line has no end at n12; P2 could not get onto s29 to reach my n17 pubs (no unoccupied-rule escape, no shared-end exception). Worked: P2's line went elsewhere (it still took the n13 passengers to n21).
- R8: Lines c (before P3's power-3 b) on s8 then s9: P3's only route from its looped end n4 was n3; with s8 mine mid-line and s9 taken, P3 was forced onto s7 and reached only n7, one street short of my n8 passenger. Worked.
- R8 Buildings b over P1's a: resolving first let me claim the n9 B site (office) before P1's 4 builds.


## 5. Opponents' deliveries and preventions
- P1 (PPO): built 2 offices at roundabout n13 in setup, so its own round-1 office clock swallowed n13's passenger; then took Vrrooomm a in R1 with no empty office on its line (wasted). R3: took Vrrooomm a+b AND Lines c (power 0) - Lines c was wasted, and the Vrrooomm pair was neutralised by my pause. R8: built 4 buildings (offices on n33/n0/n22, all on others' lines). It did repeatedly use MY n17 pubs (R6, R9), which looked opportunistic. Several power-0 / no-target markers overall.
- P2 (gumbel leader): bought buses early (4 by R5, 5 at end) and used the big bus count in pub rounds to sweep n21 pubs (2 per round). Its lines grew long (16 streets) toward stations. Sometimes dropped without a Vrrooomm (R5, R8) while holding 4 buses: in R8 it dropped with 3 markers, which is what kept the game alive into R9 for it. Never tried to block me.
- P3 (gumbel rank): aggressive Lines (power 2-3 every round) into my area (n4, n10, n5 station), then Vrrooomm with two markers straddling mine (R5 b+e: +4). That double-Vrrooomm sweep was the best bot play of the game; it looked deliberate. It spent all 20 markers by R8 and its line looped (both ends at n4), leaving it one exit that I could block.
- Nobody but me used the Clock all game; no bot paused.


## 6. Final result
Game ended after R9 on markers. Points P0 9 (1 stone) / P1 7 / P2 8 / P3 6; final scores 8 / 7 / 8 / 6. **P0 wins** the 8-8 tie with P2 on the time-stone tiebreak (R-069). Replay: `runs/agent_play/opus_g1/game.replay.json`.


## 7. Heuristic takeaways
1. **Predict next round's type and pre-build it** (confirmed). The clock auto-advances house->office->pub. Before Clock, build on your line buildings of the NEXT type at nodes where no passenger is waiting, ideally private nodes; passengers currently inside this round's type will step out next to them.
2. **Count targets, not just passengers** (confirmed). Deliveries = min(buses, eligible passengers, empty target buildings on line). When an opponent with an earlier Vrrooomm slot shares your targets, add a private target (Buildings on a site at your line end) so their fills can't starve you (R7, R8).
3. **Vrrooomm slot priority = earliest seat that can touch your passengers or your targets** (confirmed). Only pick an early slot over another action when an opponent who shares your passengers/targets could go before you. Seats that cannot reach them don't matter (R1: P1's a was harmless).
4. **Beware the straddling double Vrrooomm** (confirmed, by P3 in R5). An opponent holding slots both before and after yours can take your shared passengers first and sweep leftovers after. If you share a pool with such a seat, either go first or make your passengers/targets private.
5. **Lines resolve f..a: a late Lines letter resolves first** (confirmed). If M - k >= 1, a later Lines letter lets you occupy a street before an earlier-letter opponent extends. Use it to block: occupy the opponent's only unoccupied exit toward your passengers/buildings, and continue one more street so your end does NOT sit at the contested node (otherwise the shared-end exception lets them follow onto your street) (R6, R8).
6. **Lines power check before picking** (confirmed). Lines resolves before Buses, so power = M_now - k. A Lines c with M=2 does nothing; bots wasted these. Same for Buildings/Passengers with dwindling power.
7. **Pause as denial when you have no Vrrooomm and opponents have several** (confirmed). If the coming type gives opponents >= 2 deliveries and you none, pausing (-1 stone) puts every passenger back into its building: all their Vrrooomm markers are wasted, your setup carries over a round, and the stone is the first tiebreak.
8. **Deliver into shared targets first, keep private ones** (confirmed R1). When choosing a destination, fill the empty building that an opponent who resolves after you also needs; keep your private targets for later rounds.
9. **Don't feed shared stations** (confirmed). Passengers to a station that an opponent's line touches (and whose Vrrooomm precedes yours) is a gift: P3 reached n5 and harvested the 2 I placed in R1.
10. **Count markers exactly before burning your last ones** (confirmed, by mistake). The game ends at round end when <= 1 seat has markers. The `am` field is already net of this round's placements. I emptied mine in R8 thinking only P2 would remain; P1 had 1 left, so a R9 happened in which P1/P2 scored 4 and I could not act (P1 used MY n17 pubs). Keep one marker unless the end is certain.
11. **A looped/short line has few exits: block them** (confirmed R8). An opponent whose ends sit together (loop) or at a dead corner has 1-2 candidate streets; one Lines unit there can deny a whole power-3 extension.
12. **Buses early pays for itself only with enough targets** (speculative). P2's 4-5 buses mostly delivered 2/round because targets, not buses, were the bottleneck; my 2 buses were fully used in most rounds. A heuristic should buy buses when (eligible passengers and private targets) regularly exceed bus count.
13. **Forced builds: choose types for the round that is furthest away** (speculative). When Buildings power exceeds your useful builds, put the extra buildings off all lines or use the type two rounds out, so opponents don't get immediate targets.
