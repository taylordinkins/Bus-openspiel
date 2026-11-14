# Bus — OpenSpiel

*Bus* (Splotter Spellen; Complete Edition rules) as a Python [OpenSpiel](https://github.com/google-deepmind/open_spiel)
game, with a browser UI for human play against other humans and bots, and a clean base for RL agents.
Played on an unofficial reworked map (`assets/map/`). Status and roadmap: [PLAN.md](PLAN.md).

## Setup

```bash
uv sync
uv run pytest -q            # fast suite
uv run pytest -q -m slow    # hook tests and long simulations
```

Python 3.12, `open-spiel==2.0.2` (wheels for macOS arm64, Linux, Windows).

The rulebook is not in the repository. To work on rules, put your copy at
`third_party/rules/bus_rulebook.pdf` (the directory is git-ignored).

## Layout

- `data/board.yaml` — the map: 36 intersections, 70 streets, 47 construction sites (tiers A–D),
  2 train stations, 4 Central Park roundabouts, with map coordinates for the UI.
  `docs/board-check.png` overlays it on the map for checking.
- `bus/` — engine (Phase 1). `server/`, `web/` — Phase 2.
- `.claude/` — agents, skills and hooks for Claude Code sessions (see `CLAUDE.md`).

*Bus* is designed by Jeroen Doumen and Joris Wiersinga. This is an unofficial fan implementation.
