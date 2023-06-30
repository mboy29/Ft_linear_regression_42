# Imports
# -------

from tools import *

# Functions
# ---------

def ft_load(path: str = 'data.csv') -> pandas.DataFrame:

    """
    Load data from csv file.
    Check if data is corrupted, this includes:
    - Missing or unecessary column(s) (must be 2)
    - Wrong column(s) (not 'price' or 'km')
    - Not enought data to train program
    - Nan or negative value(s)
    - Wrong data type (must be int or float)
    - Permission denied
    - File not found
    - Is a directory

    Args:
        path (str): Path to csv file (default: 'data.csv').

    Returns:
        data (pandas.DataFrame): Data loaded from csv file.
    """

    try:

        print(message('1. Loading data from data.csv...'), end='\r')
        data: pandas.DataFrame = pandas.read_csv(path)
        if len(data.columns) > 2: raise Exception('Data file is corrupted, unecessary column(s).')
        elif len(data.columns) < 2: raise Exception('Data file is corrupted, missing column(s).')
        elif 'price' not in data or 'km' not in data: raise Exception('Data file is corrupted, wrong column(s).')
        elif len(data['price']) < 2 or len(data['km']) < 2: raise Exception('Data file is corrupted, not enought data to train program.')
        elif any(math.isnan(data) for data in data['price'].tolist()) or any(math.isnan(data) for data in data['km'].tolist()): raise Exception('Data file is corrupted, Nan values.')
        elif any(int(price) < 0 for price in data['price'].tolist()) or any(int(price) < 0 for price in data['km'].tolist()) : raise Exception('Data file is corrupted, negative value(s).')
        print(message('1. Loading data from data.csv... Done √'))
        return data

    except TypeError: raise Exception("Data file is corrupted, wrong data type (must be int or float).")
    except PermissionError: raise Exception("Data file is corrupted, permission denied.")
    except FileNotFoundError: raise Exception("Data file not found, please download it from the 42 intranet.")
    except IsADirectoryError: raise Exception(f"Data file is corrupted, '{ path }' is a directory.")

# ----------

def ft_normalize(data: list) -> list:

    """
    Normalizes a set of data stored in a list (between 0 and 1).
    Normalizing a set of data means to scale the values to a range of [0, 1].

    Args:
        data (list): List of data to normalize.
    
    Returns:
        normalized_data (list): List of normalized data.
    """

    normalized_data: list = []

    print(message('2. Normalizing data...'), end='\r')
    for idx in range(len(data)):
        normalized_data.append((data[idx] - min(data)) / (max(data) - min(data)))
    print(message('2. Normalizing data... Done √'))
    return normalized_data

# ----------

def ft_save(thetas: tuple) -> None:

    """
    Saves the final values of theta0 and theta1 in a csv file.
    If the file already exists, delete it and create a new one.

    Args:
        theta0 (float): Final value of theta0.
        theta1 (float): Final value of theta1.
    
    Returns:
        None
    """

    print(message('4. Saving thetas in thetas.csv...'), end='\r')
    if os.path.exists(PATH_THETAS):
        os.remove(PATH_THETAS)
    with open(PATH_THETAS, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['theta0', 'theta1'])
        writer.writerow([thetas[0], thetas[1]])
    print(message('4. Saving thetas in thetas.csv... Done √'))
    
# ----------

def ft_train(x_km: list, y_price: list) -> tuple:
    
    """
    Trains a linear regression model using the provided data points by iterative
    gradient descent to update theta0 and theta1 based on the training data.
    
    Args:
        x_km (list): List of mileage values (independent variable).
        y_price (list): List of corresponding price values (dependent variable).
    
    Returns:
        (theta0, theta1) (tuple): Final values of theta0 and theta1.
    """

    theta0: float = 0.0
    theta1: float = 0.0

    print(message('3. Training model...'), end='\r')
    for idx in range(0, ITERATIONS):
        tmp0: float = 0.0
        tmp1: float = 0.0
        for km, price in zip(x_km, y_price):
            tmp0 += (theta0 + theta1 * km) - price
            tmp1 += ((theta0 + theta1 * km) - price) * km
        theta0 -= LEARNING_RATE * (1 / len(x_km)) * tmp0
        theta1 -= LEARNING_RATE * (1 / len(x_km)) * tmp1
    print(message('3. Training model... Done √'))
    return (theta0, theta1)


# Main function
# -------------

def ft_main(*args) -> None:

    """
        Main function.

        Args:
            args (list): List of arguments passed to the program.
        
        Returns:
            None
    """

    if len(args) > 0:
        raise Exception('No arguments expected.')
    data: pandas.DataFrame = ft_load()
    x_km: list = ft_normalize(data['km'].tolist())
    y_price: list = ft_normalize(data['price'].tolist())
    thetas: tuple = ft_train(x_km, y_price)
    
    ft_save(thetas)
    print(message('\nFinal thetas value :'))
    print(message(f'   - Theta0: { thetas[0] }'))
    print(message(f'   - Theta1: { thetas[1] }'))


# ADD DATA.CSV FILE TO ARGUMENT PROGRAM (DEFAULT: DATA.CSV)

# Main
# ----

if __name__ == '__main__':

    try:
        print(header(f'___ ___   _   _ _  _ __ ___ ___   ___ __ ___ ___ __ ___ ___ _ ___ _  _'))
        print(header(f'|_   |    |   | |\ | |_ |_| |_|   |_| |_ | _ |_| |_ |_  |_  | | | |\ |'))
        print(header(f'|    |    |__ | | \| |_ | | | \   | \ |_ |_| | \ |_ __| __| | |_| | \|\n'))
        print(header(f'--------------------------- TRAINING MODEL ---------------------------\n'))
        ft_main(*sys.argv[1:])
        sys.exit(0)
    
    except (KeyboardInterrupt, EOFError): print(error('[ WARNING ]: Program interrupted by user.'))
    except Exception as exc:
        print(error(f'[ ERROR ]: { exc }'))
        sys.exit(1)
