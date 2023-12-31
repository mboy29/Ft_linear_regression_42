# Imports
# -------

from train import ft_load
from tools import *


# Functions
# ---------

def ft_thetas(path: str = PATH_THETAS, output: bool = True) -> list :

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
        path (str): Path to csv file (default: 'data.csv').
        output (bool): Print output messages (default: True).
    
    Returns:
        thetas (list): List containing theta0 and theta1.
    """

    try:

        if output: print(message('1. Fetching thetas from thetas.csv...'), end='\r')        
        data: pandas.DataFrame = pandas.read_csv(path)
        if len(data.columns) > 2: raise Exception('Thetas file is corrupted, unecessary column(s).')
        elif len(data.columns) < 2: raise Exception('Thetas file is corrupted, missing column(s).')
        elif 'theta0' not in data or 'theta1' not in data: raise Exception('Thetas file is corrupted, wrong column(s).')
        elif len(data['theta0']) != 1 or len(data['theta1']) != 1: raise Exception('Thetas file is corrupted, should only contain one value for each column.')
        elif any(math.isnan(data) for data in data['theta0'].tolist()) or any(math.isnan(data) for data in data['theta1'].tolist()): raise Exception('Thetas file is corrupted, Nan values.')
        if output: print(message('1. Fetching thetas from thetas.csv... Done √'))
        return (data['theta0'].item(), data['theta1'].item())
    
    except TypeError: raise Exception("Data file is corrupted, wrong data type (must be int or float).")
    except PermissionError: raise Exception("Data file is corrupted, permission denied.")
    except IsADirectoryError: raise Exception(f"Data file is corrupted, '{ path }' is a directory.")
    except FileNotFoundError:
        if output: print(message('1. Fetching thetas from thetas.csv... ⤫'), end='\r')        
        print(error(f'\n\n[ WARNING ]: { path } not found, default values 0.0 will be applied to prediction.\nTrain beforehand to avoid this warning (python3 train.py).\n'))
        return (None, None)

# ----------

def ft_kilometers() -> float:

    """
    Asks the user for a mileage and returns it.

    Args:
        None
    
    Returns:
        mileage (float): Mileage entered by the user.
    """

    try:
        km: float = float(input(message('2. Enter a mileage (in km): ')))
        if km < 0: raise Exception('Mileage cannot be negative.')
        return km
    
    except (ValueError, KeyboardInterrupt): raise Exception('Mileage must be a number.')
    except EOFError: raise Exception('Mileage must be a number.')

# ----------

def ft_predict(thetas: list, km: float, output: bool = True) -> float:
    
        """
        Predicts the price of a car with a given mileage and 
        outputs the result to the user.
    
        Args:
            thetas (list): List containing thetas loaded from csv file.
            mileage (float): Mileage entered by the user.
            output (bool): Print output messages (default: True).
        
        Returns:
            price (float): Predicted price of the car.
        """
    
        price: float = 0.0
        data: pandas.DataFrame = ft_load(PATH_DATA, False)
        x_km: list = data['km'].tolist()
        y_price: list = data['price'].tolist()

        if output: print(message(f'3. Predicting price for a car with a mileage of { km } km...'), end='\r')
        if thetas[0] == None or thetas[1] == None: price = 0.0
        else: price = ft_denormalize_value(y_price, (thetas[0] + (thetas[1] * ft_normalize_value(x_km, km))))
        if price < 0: price = 0.0
        if output: print(message(f'3. Predicting price for a car with a mileage of { km } km... Done √'))
        return price

# ----------

def ft_precision(thetas: list) -> float:
    
    """
    Calculates the precision of the program and outputs
    the result to the user.
    The precision is the average of the difference between
    the predicted price and the actual price.

    Args:
        thetas (list): List of thetas loaded from csv file.

    Returns:
        precision (float): Precision of the program.
    """

    prices: list = []
    precision: float = 0.0
    data: pandas.DataFrame = ft_load(PATH_DATA, False)
    x_km: list = data['km'].tolist()
    y_price: list = data['price'].tolist()

    print(message(f'4. Calculating precision...'), end='\r')
    if thetas[0] == -1.0 or thetas[1] == -1.0: precision = 0.0
    else:
        for km, _ in zip(x_km, y_price):
            prices.append(ft_denormalize_value(y_price, (thetas[0] + (thetas[1] * ft_normalize_value(x_km, km)))))
        precision = sum([abs(price - prices[index]) for index, price in enumerate(y_price)]) / len(y_price)
    
    print(message(f'4. Calculating precision... Done √'))
    return precision


# Main function
# -------------

def ft_main(args: list) -> None:

    """
        Main function.

        Args:
            -bonus (bool): Optional argument to enable bonus mode (default: False).
    """

    global BONUS
    km: float = None
    thetas: tuple = None
    price: float = None
    precision: float = None

    if '-bonus' in args:
        BONUS = True
        args.remove('-bonus')
    if len(args) > 1:
        raise Exception("Please provide path to CSV or/and -bonus flag only, or no arguments at all to use default path 'thetas.csv'.")
    thetas = ft_thetas() if len(args) == 0 else ft_thetas(args[0])
    km = ft_kilometers()
    price = ft_predict(thetas, km)
    if BONUS: precision = ft_precision(thetas)
    print(message(f"\nFinal price: { round(price, 2) } €"))
    if BONUS: print(message(f"Precision: { round(precision, 2) } €"))


# Main
# ----

if __name__ == '__main__':

    try:
        print(header(f'___ ___   _   _ _  _ __ ___ ___   ___ __ ___ ___ __ ___ ___ _ ___ _  _'))
        print(header(f'|_   |    |   | |\ | |_ |_| |_|   |_| |_ | _ |_| |_ |_  |_  | | | |\ |'))
        print(header(f'|    |    |__ | | \| |_ | | | \   | \ |_ |_| | \ |_ __| __| | |_| | \|\n'))
        print(header(f'----------------------------- PREDICTION -----------------------------\n'))
        ft_main(sys.argv[1:])
        sys.exit(0)
    
    except (KeyboardInterrupt, EOFError): print(error('[ WARNING ]: Program interrupted by user.'))
    except Exception as exc:
        print(error(f'[ ERROR ]: { exc }'))
        sys.exit(1)