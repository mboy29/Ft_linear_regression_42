# Imports
# -------
from tools import *


# Functions
# ---------

def ft_thetas() -> tuple :

    """
    Load thetas from csv file.
    Check if data is corrupted, this includes:
    - Missing or unecessary column(s) (must be 2)
    - Wrong column(s) (not 'theta0' or 'theta1')
    - Not enought data to train program
    - Nan or negative value(s)
    - Wrong data type (must be int or float)
    - Permission denied
    - File not found
    - Is a directory

    If thetas file is not found, default values 0.0
    will be applied to prediction and the user will be warned.

    Args:
        None
    
    Returns:
        thetas (tuple): Tuple of thetas loaded from csv file.
    """

    try:

        print(message('1. Fetching thetas from thetas.csv...'), end='\r')        
        data: pandas.DataFrame = pandas.read_csv(PATH_THETAS)
        if len(data.columns) > 2: raise Exception('Thetas file is corrupted, unecessary column(s).')
        elif len(data.columns) < 2: raise Exception('Thetas file is corrupted, missing column(s).')
        elif 'theta0' not in data or 'theta1' not in data: raise Exception('Thetas file is corrupted, wrong column(s).')
        elif len(data['theta0']) != 1 or len(data['theta1']) != 1: raise Exception('Thetas file is corrupted, should only contain one value for each column.')
        elif any(math.isnan(data) for data in data['theta0'].tolist()) or any(math.isnan(data) for data in data['theta1'].tolist()): raise Exception('Thetas file is corrupted, Nan values.')
        print(message('1. Fetching thetas from thetas.csv... Done √'))
        return (data['theta0'], data['theta1'])
    
    except TypeError: raise Exception("Data file is corrupted, wrong data type (must be int or float).")
    except PermissionError: raise Exception("Data file is corrupted, permission denied.")
    except IsADirectoryError: raise Exception(f"Data file is corrupted, '{ path }' is a directory.")
    except FileNotFoundError:
        print(error(f'[ WARNING ]: { PATH_THETAS } not found, default values 0.0 will be applied to prediction.\nTrain beforehand to avoid this warning (python3 train.py).'))
        return (0.0, 0.0)


def ft_meliage() -> float:

    """
    Asks the user for a mileage and returns it.

    Args:
        None
    
    Returns:
        mileage (float): Mileage entered by the user.
    """

    try:
        mileage: float = float(input(message('2. Enter a mileage (in km): ')))
        if mileage < 0: raise Exception('Mileage cannot be negative.')
        return mileage
    
    except (ValueError, KeyboardInterrupt): raise Exception('Mileage must be a number.')
    except EOFError: raise Exception('Mileage must be a number.')


def     ft_predict(mileage: float, thetas: tuple) -> float:
    price = thetas[1] * normalizeElem(mileages, mileage) + thetas[0]


# Main function
# -------------

def ft_main(*args) -> None:

    """
        Main function.

        Args:
            args (list): List of arguments passed to the program.
    """

    thetas: tuple = ft_thetas()
    mileage: float = ft_meliage()



# Main
# ----

if __name__ == '__main__':

    try:
        print(header(f'___ ___   _   _ _  _ __ ___ ___   ___ __ ___ ___ __ ___ ___ _ ___ _  _'))
        print(header(f'|_   |    |   | |\ | |_ |_| |_|   |_| |_ | _ |_| |_ |_  |_  | | | |\ |'))
        print(header(f'|    |    |__ | | \| |_ | | | \   | \ |_ |_| | \ |_ __| __| | |_| | \|\n'))
        print(header(f'----------------------------- PREDICTION -----------------------------\n'))
        ft_main(*sys.argv[1:])
        sys.exit(0)
    
    except (KeyboardInterrupt, EOFError): print(error('[ WARNING ]: Program interrupted by user.'))
    except Exception as exc:
        print(error(f'[ ERROR ]: { exc }'))
        sys.exit(1)