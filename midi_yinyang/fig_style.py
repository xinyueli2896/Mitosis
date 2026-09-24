"""Shared style of the model-figure panels (v72 = the v64 palette with
near-white components): light blue for stream x, salmon for stream y,
white-a-touch-grey for every model component, green for the x-stream
cross path (Cross Attn box, its gate, its routed lines) and yellow/
orange for the y-stream cross path, dark outlines at one heavier
weight, serif type. Changing a value here restyles all three panels."""

BLUE, BLUE_L = 'DCE6F2', 'B7C9DE'        # stream x fill; its light line (the v64 palette)
RED, RED_L = 'F4A79A', 'E88A7C'          # stream y fill; its light line
FILL = {'x': BLUE, 'y': RED}
LINE = {'x': BLUE_L, 'y': RED_L}
COMP = 'F4F4F4'                          # model components: white, a touch grey
POOL = 'BFBFBF'                          # the expert pool band: grey, drawn translucent
GREEN, GREEN_L = 'C5E0B4', '6AA84F'      # x-stream cross path: box fill, line
YELLOW, YELLOW_L = 'FFF2CC', 'E69138'    # y-stream cross path: box fill, line
CROSS = {'x': (GREEN, GREEN_L), 'y': (YELLOW, YELLOW_L)}
GREY_L = '8C8C8C'                        # self-attention lines
INK = '3A3A3A'                           # outlines, arrows and lines
LW = 1.5                                 # outline weight of every box
AW = 1.1                                 # arrow weight
FS = 10.5                                # label size in boxes
BH = 0.4                                 # box height
