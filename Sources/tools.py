# Imports
# -------
import sys, pandas, math, os, csv
import matplotlib.pyplot as pyplot
from sty import fg


# Globals
# -------

BONUS = False
ITERATIONS = 500
LEARNING_RATE = 0.5
PATH_DATA = '../DataSets/data.csv'
PATH_THETAS = '../DataSets/thetas.csv'


# Colored output
# --------------

def error(message: str) -> str:

    """
    Returns a colored error message.

    Args:
        message (str): Message to color.

    Returns:
        colored_message (str): Colored message.
    """

    return f'{ fg(255, 218, 185) }{ message }{ fg.rs }'

# ----------

def header(message: str) -> str:

    """
    Returns a colored header message.
    
    Args:
        message (str): Message to color.
    
    Returns:
        colored_message (str): Colored message.
    """

    return f'{ fg(240, 128, 128) }{ message }{ fg.rs }'

# ----------

def message(message: str) -> str:

    """
    Returns a colored message.

    Args:
        message (str): Message to color.
    
    Returns:
        colored_message (str): Colored message.
    """

    return f'{ fg(248, 173, 157) }{ message }{ fg.rs }'


# Normalization / Denormalization
# -------------------------------

def ft_normalize_list(data: list) -> list:

    """
    Normalizes a set of data stored in a list (between 0 and 1).
    Normalizing a set of data means to scale the values to a range of [0, 1].

    Args:
        data (list): List of data to normalize.
    
    Returns:
        normalized_list (list): List of normalized data.
    """
    return [(value - min(data)) / (max(data) - min(data)) for value in data]

# ----------

def ft_normalize_value(data: list, value: float) -> list:

    """
    Normalizes a value (between 0 and 1) from a set of data stored in a list.
    Normalizing a set of data means to scale the values to a range of [0, 1].

    Args:
        data (list): List of data to normalize.
        value (float): Value to normalize.
    
    Returns:
        normalized_value (flaot): Normalized value.
    """

    return ((value - min(data)) / (max(data) - min(data)))

# ----------

def ft_denormalize_value(data: list, value: float) -> list:


    """
    Denormalizes a value from a set of data stored in a list.
    Denormalizing a set of data means to scale the values to their original range.

    Args:
        data (list): List of data to denormalize.
        value (float): Value to denormalize.
    
    Returns:
        denormalized_value (flaot): Denormalized value.
    """

    return ((value * (max(data) - min(data))) + min(data))