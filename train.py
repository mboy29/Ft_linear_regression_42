# Imports
# -------
import sys, pandas, math
import matplotlib.pyplot as pyplot
from tools import *


# Globals
# -------
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
RED = '\033[0;91m'
GREEN='\033[0;92m'
YELLOW = '\033[0;93m'
BLUE = '\033[0;94m'
PURPLE = '\033[0;95m'
NONE = '\033[0;37m'


# Outputs
# -------
def red(str): return f'{RED}{str}{NONE}'
def green(str): return f'{GREEN}{str}{NONE}'
def yellow(str): return f'{YELLOW}{str}{NONE}'
def blue(str): return f'{BLUE}{str}{NONE}'
def purple(str): return f'{PURPLE}{str}{NONE}'


# Functions
# ---------

# Load data from csv file
# -----------------------
def ft_load() -> pandas.DataFrame:

    try:
        path: str = 'data.csv'
        data: pandas.DataFrame = pandas.read_csv(path)

        if len(data.columns) > 2: raise Exception('Data file is corrupted, unecessary column(s).')
        elif len(data.columns) < 2: raise Exception('Data file is corrupted, missing column(s).')
        elif 'price' not in data or 'km' not in data: raise Exception('Data file is corrupted, wrong column(s).')
        elif len(data['price']) < 2 or len(data['km']) < 2: raise Exception('Data file is corrupted, not enought data to train program.')
        elif any(math.isnan(data) for data in data['price'].tolist()) or any(math.isnan(data) for data in data['km'].tolist()): raise Exception('Data file is corrupted, Nan values.')
        elif any(int(price) < 0 for price in data['price'].tolist()) or any(int(price) < 0 for price in data['km'].tolist()) : raise Exception('Data file is corrupted, negative value(s).')
        return data

    except TypeError: raise Exception("Data file is corrupted, wrong data type (must be int or float).")
    except PermissionError: raise Exception("Data file is corrupted, permission denied.")
    except FileNotFoundError: raise Exception("Data file not found, please download it from the 42 intranet.")
    except IsADirectoryError: raise Exception(f"Data file is corrupted, '{ path }' is a directory.")


def ft_plot(x_km: list, y_price: list) -> None:
    pyplot.plot(x_km, y_price, 'ro')
    pyplot.xlabel('km')
    pyplot.ylabel('price')
    pyplot.show()


# Main function
# -------------
def main(*args) -> int:

    if len(args) > 0:
        raise Exception('No arguments expected.')

    data: pandas.DataFrame = ft_load()
    x_km: list = ft_normalize(data['km'].tolist())
    y_price: list = ft_normalize(data['price'].tolist())
    

    print(purple(f'Data: { data }'))
    print(purple(x_km))
    print(blue(y_price))

    ft_plot(x_km, y_price)

    # thetas = ft_coefficients(x_km, y_price)
    # print(purple(f'Theta0: { thetas[0] }'))
    # print(purple(f'Theta1: { thetas[1] }'))


    # predictions = simple_linear_regression(x_km, y_price)
    # print(blue(f'Predictions: { predictions }'))
    return 0


# Main
# ----
if __name__ == '__main__':
    try:
        sys.exit(main(*sys.argv[1:]))
    
    except (KeyboardInterrupt, EOFError): print(yellow('Program interrupted by user.'))
    except Exception as exc:
        print(red(f'[ ERROR ]: { exc }'))
        sys.exit(1)


        # https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/