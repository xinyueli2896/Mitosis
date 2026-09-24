"""Shared style of the model-figure panels. Palette (2026-09-24): the
six-swatch set Vanilla Cream / Blush Petal / Rosewood / Sage Leaf /
Misty Sky / Midnight Lagoon -- Misty Sky for stream x, Sage Leaf for
stream y, Vanilla Cream for every model component, Blush Petal for the
two cross-attention paths with their lines in the reading stream's
colour, Rosewood for the self-attention lines, Midnight Lagoon for
outlines, arrows and type. Changing a value here restyles all three
panels."""

BLUE, BLUE_L = 'A9B7C6', 'A9B7C6'        # stream x: Misty Sky (fill and wedge)
RED, RED_L = 'A8B58A', 'A8B58A'          # stream y: Sage Leaf (fill and wedge)
FILL = {'x': BLUE, 'y': RED}
LINE = {'x': BLUE_L, 'y': RED_L}
COMP = 'FFF7E6'                          # model components: Vanilla Cream
POOL = 'A9B7C6'                          # the expert pool band: Misty Sky, drawn translucent
GREEN, GREEN_L = 'F7C8D3', 'A9B7C6'      # x-stream cross path: Blush Petal box, lines in the reading stream's Misty Sky
YELLOW, YELLOW_L = 'F7C8D3', 'A8B58A'    # y-stream cross path: Blush Petal box, lines in the reading stream's Sage Leaf
CROSS = {'x': (GREEN, GREEN_L), 'y': (YELLOW, YELLOW_L)}
GREY_L = 'B46A72'                        # self-attention lines: Rosewood
INK = '2D3A47'                           # outlines, arrows and type: Midnight Lagoon
LW = 1.5                                 # outline weight of every box
AW = 1.1                                 # arrow weight
FS = 10.5                                # label size in boxes
BH = 0.4                                 # box height
