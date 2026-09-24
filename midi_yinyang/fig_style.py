"""Shared style of the model-figure panels (2026-09-24, third set):
Breeze for stream x and Air-Kiss for stream y (tokens, banks, wedges),
near-white for every model component, Coconut Macaroon for the two
cross-attention boxes and gates and for the self-attention lines,
Morning Mist for the translucent expert band, Phantom for outlines,
arrows and cross-attention lines, black type. Changing a value here
restyles all three panels."""

BLUE, BLUE_L = 'C3DDE4', 'C3DDE4'        # stream x: Breeze (fill and wedge)
RED, RED_L = 'F0D3C8', 'F0D3C8'          # stream y: Air-Kiss (fill and wedge)
FILL = {'x': BLUE, 'y': RED}
LINE = {'x': BLUE_L, 'y': RED_L}
COMP = 'F4F4F4'                          # model components: white, a touch grey
POOL = 'E1EAEC'                          # the expert pool band: Morning Mist, drawn translucent
GREEN, GREEN_L = 'D7C7BD', '6F7C80'      # x-stream cross path: Coconut Macaroon box, Phantom lines
YELLOW, YELLOW_L = 'D7C7BD', '6F7C80'    # y-stream cross path: Coconut Macaroon box, Phantom lines
CROSS = {'x': (GREEN, GREEN_L), 'y': (YELLOW, YELLOW_L)}
GREY_L = 'D7C7BD'                        # self-attention lines: Coconut Macaroon
INK = '6F7C80'                           # outlines, arrows and lines: Phantom (type is black, fig_arch_pptx.INK)
LW = 1.5                                 # outline weight of every box
AW = 1.1                                 # arrow weight
FS = 10.5                                # label size in boxes
BH = 0.4                                 # box height
