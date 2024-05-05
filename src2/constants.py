from enum import Enum

TABLE_LEN_LONG = 2844
TABLE_LEN_SHORT = 1422
BALL_SIZE = 61.5
CUSHION_WIDTH = 30

COLOR_RANGE = {
    "table":  {"lower": [80, 10, 50], "upper":[165,255,255]},
    "white":  {"lower": [0, 0, 220], "upper":[180, 150, 255]},
    "yellow": {"lower": [15, 0, 0], "upper":[30, 255, 255]},
    "red":    [
        {"lower": [0, 100, 50], "upper":[10, 255, 255]},
        {"lower": [165, 50, 50], "upper":[180 ,255,255]}
    ]
}

class Color(Enum):
    RED = 1
    YELLOW = 2
    WHITE = 3

