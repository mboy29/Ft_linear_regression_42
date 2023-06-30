# Imports
# -------
import sys, pandas, math, os, csv
import matplotlib.pyplot as pyplot
from sty import fg


# Globals
# -------

ITERATIONS = 1000
LEARNING_RATE = 0.5
PATH_THETAS = 'thetas.csv'

# Colored output
# --------------

def error(message: str) -> str:
    return f'{ fg(255, 218, 185) }{ message }{ fg.rs }'

def header(message: str) -> str:
    return f'{ fg(240, 128, 128) }{ message }{ fg.rs }'

def message(message: str) -> str:
    return f'{ fg(248, 173, 157) }{ message }{ fg.rs }'