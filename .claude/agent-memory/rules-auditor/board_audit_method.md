---
name: board-audit-method
description: How to audit data/board.yaml against the map without PIL, plus the visual traps (labels on row lines look like streets) and the 2026-11-14 baseline result
metadata:
  type: reference
---

**Tooling:** PIL is not installed (not in the system python, not in uv). Use `pdftoppm` on `assets/map/bus_map_reworked_v2.pdf`:
- Zoomed crop: `pdftoppm -r R -png -x X*R/100 -y Y*R/100 -W .. -H .. pdf out` (board.yaml coordinates = 100 dpi pixels).
- Pixel checks: `pdftoppm -r 100` gives a P6 PPM; parse it in pure python (header split on 3 newlines). Node centres are cream (254,245,212) with a red ring about 7–8 px away; that confirms every node within 1 px.

**Visual traps:**
- Site-letter pairs (AA, AC) sit exactly on the row lines between intersections (13–14, 16–17, 17–18, 19–20, 20–21, 27–28), so at full-map zoom they look like streets. None of these are streets; zoom in to confirm.
- Station icons are off-centre from where the streets meet (node 5's icon is centred at x 1348.5 but the streets meet at about x 1344). Data uses the icon centre, which is fine.

**Baseline (2026-11-14, Phase 0 unit 5):** all 70 streets, 47 site tiers/owners, 2 stations, 4 roundabouts and 36 node coords matched the map. The only nit was the 28–35 bend `via` (1387,1337) vs the drawn corner at about (1376,1336). If board.yaml changes later, diff it against this rather than redoing the whole audit.

Related: [[audit-patterns]].
