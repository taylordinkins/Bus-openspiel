"""Board data checks: counts and structure of data/board.yaml (PLAN.md §4 C1)."""

from collections import Counter, deque

from bus.board import load_board


def test_counts() -> None:
    b = load_board()
    assert len(b.nodes) == 36
    assert len(b.streets) == 70
    assert len(b.sites) == 47
    assert Counter(s.tier for s in b.sites) == {"A": 12, "B": 11, "C": 9, "D": 15}


def test_two_stations_and_four_roundabouts() -> None:
    b = load_board()
    assert len(b.stations) == 2
    assert len(b.roundabouts) == 4
    assert not set(b.stations) & set(b.roundabouts)


def test_stations_have_no_sites_and_roundabouts_have_two_a_sites() -> None:
    b = load_board()
    for n in b.stations:
        assert b.node_sites[n] == ()
    for n in b.roundabouts:
        assert [b.sites[s].tier for s in b.node_sites[n]] == ["A", "A"]


def test_graph_is_connected() -> None:
    b = load_board()
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for sid in b.incident[node]:
            nxt = b.streets[sid].other(node)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    assert len(seen) == len(b.nodes)


def test_every_node_has_a_street() -> None:
    b = load_board()
    assert all(b.incident[n.id] for n in b.nodes)


def test_at_most_two_sites_per_intersection() -> None:
    b = load_board()
    assert max(len(s) for s in b.node_sites) == 2


def test_handshake() -> None:
    b = load_board()
    assert sum(len(i) for i in b.incident) == 2 * len(b.streets)


def test_only_bend_streets_carry_via() -> None:
    b = load_board()
    with_via = {(min(s.a, s.b), max(s.a, s.b)) for s in b.streets if s.via}
    assert with_via == {(3, 6), (28, 35)}


def test_coordinates_inside_frame() -> None:
    b = load_board()
    pts = [(n.x, n.y) for n in b.nodes] + [(s.x, s.y) for s in b.sites]
    pts += [p for s in b.streets for p in s.via]
    assert all(0 <= x <= 1991 and 0 <= y <= 1991 for x, y in pts)


def test_layout_actions_have_letters() -> None:
    lay = load_board().layout
    for key in ("lines", "passengers", "buildings", "vrrooomm"):
        assert sorted(lay.actions[key]) == list("abcdef")
    for key in ("buses", "clock", "starting_player"):
        assert sorted(lay.actions[key]) == ["a"]


def test_layout_counts_and_frame() -> None:
    lay = load_board().layout
    assert len(lay.bus_grid) == 20 and len(lay.stones) == 5 and len(lay.score_track) == 20
    assert set(lay.clock_angles) == {"house", "office", "pub"}
    pts = [p for v in lay.actions.values() for p in v.values()]
    pts += [*lay.bus_grid, *lay.stones, *lay.score_track, (lay.clock_x, lay.clock_y)]
    assert all(0 <= x <= 1991 and 0 <= y <= 1991 for x, y in pts)


def test_score_track_runs_downward() -> None:
    ys = [y for _, y in load_board().layout.score_track]
    assert ys == sorted(ys)
