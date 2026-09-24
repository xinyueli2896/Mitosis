"""Shared style of the model-figure panels. Palette (2026-09-24, second
set): Phantom / Breeze / Morning Mist / White Russian / Air-Kiss /
Coconut Macaroon. The two lightest swatches tell the streams apart
(Morning Mist for x, White Russian for y, their wedges in Breeze and
Air-Kiss), Coconut Macaroon fills every component, Air-Kiss the two
cross-attention boxes and gates, Breeze the translucent expert band,
and Phantom is the ink: outlines, arrows, lines and type. Changing a
value here restyles all three panels."""

BLUE, BLUE_L = 'E1EAEC', 'C3DDE4'        # stream x: Morning Mist fill, Breeze wedge
RED, RED_L = 'EEDFD9', 'F0D3C8'          # stream y: White Russian fill, Air-Kiss wedge
FILL = {'x': BLUE, 'y': RED}
LINE = {'x': BLUE_L, 'y': RED_L}
COMP = 'D7C7BD'                          # model components: Coconut Macaroon
POOL = 'C3DDE4'                          # the expert pool band: Breeze, drawn translucent
GREEN, GREEN_L = 'F0D3C8', 'D7C7BD'      # x-stream cross path: Air-Kiss box, Coconut Macaroon lines
YELLOW, YELLOW_L = 'F0D3C8', 'D7C7BD'    # y-stream cross path: Air-Kiss box, Coconut Macaroon lines
CROSS = {'x': (GREEN, GREEN_L), 'y': (YELLOW, YELLOW_L)}
GREY_L = '6F7C80'                        # self-attention lines: Phantom
INK = '6F7C80'                           # outlines, arrows and type: Phantom
LW = 1.5                                 # outline weight of every box
AW = 1.1                                 # arrow weight
FS = 10.5                                # label size in boxes
BH = 0.4                                 # box height
