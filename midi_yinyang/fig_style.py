"""Shared style of the model-figure panels, taken from the hand sketch of
2026-09-23 (second version): light blue for stream x, salmon for stream
y, grey for every model component, green for the x-stream cross path
(Cross Attn box, its gate, its routed lines) and yellow/orange for the
y-stream cross path, dark outlines at one heavier weight, serif type.
Changing a value here restyles all three panels."""

BLUE, BLUE_L = 'DCE6F2', 'B7C9DE'        # stream x fill; its light line
RED, RED_L = 'F4A79A', 'E88A7C'          # stream y fill; its light line
FILL = {'x': BLUE, 'y': RED}
LINE = {'x': BLUE_L, 'y': RED_L}
COMP = 'D9D9D9'                          # model components
POOL = 'EEEEEE'                          # the expert pool band
GREEN, GREEN_L = 'C5E0B4', '6AA84F'      # x-stream cross path: box fill, line
YELLOW, YELLOW_L = 'FFF2CC', 'E69138'    # y-stream cross path: box fill, line
CROSS = {'x': (GREEN, GREEN_L), 'y': (YELLOW, YELLOW_L)}
GREY_L = '8C8C8C'                        # self-attention lines
INK = '303030'
LW = 1.3                                 # outline weight of every box
AW = 1.0                                 # arrow weight
