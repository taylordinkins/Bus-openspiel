# Board palette tokens

Defined at the top of `tools/render_board.py`; Phase 2 reuses them. Sampled from `assets/map/bus_map_reworked_v2.png`.

| Token | Hex | Use |
|---|---|---|
| PAPER | #FFF4D6 | board background |
| INK | #1A1A1A | icons, text |
| RED / RED_DARK | #B00A0A / #7E0606 | frames, streets, site slots, action spaces |
| DISTRICT_OUTER / MID / CORE | #E3D5F0 / #FDE3D6 / #CDE39A | lavender, peach, green zones |
| RIVER / RIVER_EDGE | #8FE3FF / #14B4F0 | river |
| STONE | #8C1D1D | time stones |
| P0..P4 | #FE0000, #007F00, #FE6500, #0065FE, #FEFEFE | seats: red, green, orange, blue, white (D21) |
| SEAT_TEXT | #FFFFFF, #FFFFFF, #1A1A1A, #FFFFFF, #1A1A1A | label colour on each seat colour |

Seat colours are the five stripes of the Lines cell in the art, sampled from the PNG's pixel rows at x=180 (stripe centres y≈1570, 1588, 1607, 1625, 1645; first red, green, orange, blue, white). Pieces in these colours carry a dark outline (`INK`) so white and red read against the cream paper and the red frames; identity should not rely on colour alone (UI_DESIGN §8).

The Buses grid (D19) is 5 seat columns by 4 rows (buses 2-5, bus 2 on top); `Layout.bus_cell(seat, bus)` gives the cell centre. The starting bus is not drawn.
