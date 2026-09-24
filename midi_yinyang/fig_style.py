"""Shared style of the model-figure panels, taken from the hand sketch of
2026-09-23 (second version): light blue for stream x, salmon for stream
y, grey for every model component, green for the x-stream cross path
(Cross Attn box, its gate, its routed lines) and yellow/orange for the
y-stream cross path, dark outlines at one heavier weight, serif type.
Changing a value here restyles all three panels."""

BLUE, BLUE_L = 'A599B4', '857A94'        # stream x fill (mauve); its line
RED, RED_L = 'BDD2A6', '97AF7E'          # stream y fill (sage); its line
FILL = {'x': BLUE, 'y': RED}
LINE = {'x': BLUE_L, 'y': RED_L}
COMP = 'FFF8E7'                          # model components
POOL = 'FFFCF4'                          # the expert pool band
GREEN, GREEN_L = 'DCE6F2', 'B7C9DE'      # x-stream cross path: the earlier low-saturation blue (box fill, line)
YELLOW, YELLOW_L = 'F4A79A', 'E88A7C'    # y-stream cross path: the earlier low-saturation red (box fill, line)
CROSS = {'x': (GREEN, GREEN_L), 'y': (YELLOW, YELLOW_L)}
GREY_L = '8C8C8C'                        # self-attention lines
INK = '3A3A3A'
LW = 1.5                                 # outline weight of every box
AW = 1.1                                 # arrow weight
FS = 10.5                                # label size in boxes
BH = 0.4                                 # box height
