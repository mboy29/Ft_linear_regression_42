# Imports
# -------

from tools import *


# Functions
# ---------

def ft_load(path: str = PATH_DATA, output: bool = True) -> pandas.DataFrame:

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
        output (bool): Print output messages (default: True).

    Returns:
        data (pandas.DataFrame): Data loaded from csv file.
    """

    try:

        if output: print(message(f'1. Loading data from { path }...'), end='\r')
        data: pandas.DataFrame = pandas.read_csv(path)
        if len(data.columns) > 2: raise Exception(f"Data file '{ path }' is corrupted, unecessary column(s).")
        elif len(data.columns) < 2: raise Exception(f"Data file '{ path }' is corrupted, missing column(s).")
        elif 'price' not in data or 'km' not in data: raise Exception(f"Data file '{ path }' is corrupted, wrong column(s).")
        elif len(data['price']) < 2 or len(data['km']) < 2: raise Exception(f"Data file '{ path }' is corrupted, not enought data to train program.")
        elif any(math.isnan(data) for data in data['price'].tolist()) or any(math.isnan(data) for data in data['km'].tolist()): raise Exception(f"Data file '{ path }' is corrupted, Nan values.")
        elif any(int(price) < 0 for price in data['price'].tolist()) or any(int(price) < 0 for price in data['km'].tolist()) : raise Exception(f"Data file '{ path }' is corrupted, negative value(s).")
        elif path.endswith('.csv') == False: raise Exception(f"Data file '{ path }' is corrupted, wrong file extension (must be .csv).")
        if output: print(message(f'1. Loading data from { path }... Done √'))
        return data

    except TypeError: raise Exception(f"Data file '{ path }' is corrupted, wrong data type (must be int or float).")
    except PermissionError: raise Exception(f"Data file '{ path }' is corrupted, permission denied.")
    except FileNotFoundError: raise Exception(f"Data file '{ path }' not found, please download it from the 42 intranet.")
    except IsADirectoryError: raise Exception(f"Data file is corrupted, '{ path }' is a directory.")

# ----------

def ft_save(thetas: list, path: str = PATH_THETAS, output: bool = True) -> None:

    """
    Saves the final values of theta0 and theta1 in a csv file.
    If the file already exists, delete it and create a new one.

    Args:
        thetas (list): List of theta0 and theta1.
        path (str): Path to csv file (default: 'thetas.csv').
        output (bool): Print output messages (default: True).
    
    Returns:
        None
    """

    if output: print(message('3. Saving thetas in thetas.csv...'), end='\r')
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['theta0', 'theta1'])
        writer.writerow([thetas[0], thetas[1]])
    if output: print(message('3. Saving thetas in thetas.csv... Done √'))
    
# ----------

def ft_loss(thetas: list, x_km: list, y_price: list) -> float:

    """
    Computes the loss function of the linear regression model,
    this to say the loss between the predicted prices and the
    actual prices for a given set of mileages and prices.
    The loss function is the mean squared error of the model.

    Args:
        thetas (list): List containing theta0 and theta1.
        x_km (list): List of mileage values (independent variable).
        y_price (list): List of corresponding price values (dependent variable).
    
    Returns:
        loss (float): Loss function of the model.
    """

    loss: float = 0.0

    for mileage, price in zip(x_km, y_price):
        loss += (price - (thetas[1] * mileage + thetas[0])) ** 2
    return (loss / len(x_km))

# ----------

def ft_adjust(x_km: list, y_price: list, thetas: list, tmp: list, loss: list) -> list:

    """
    Adjusts the learning rate based on the loss function. 
    If the loss function increases, the learning rate is divided by 2.
    If the loss function decreases, the learning rate is multiplied by 1.05.

    Args:
        x_km (list): List of mileage values (independent variable).
        y_price (list): List of corresponding price values (dependent variable).
        thetas (list): List of theta0 and theta1.
        tmp (list): List of temporary theta0 and theta1.
        loss (list): List of loss function values.

    Returns:
        thetas (list): List of theta0 and theta1.
    """

    global LEARNING_RATE

    if len(loss) > 1:
        if loss[-1] >= loss[-2]:
            thetas[0] += tmp[0] / len(x_km) * LEARNING_RATE
            thetas[1] += tmp[1] / len(y_price) * LEARNING_RATE
            LEARNING_RATE *= 0.5
        else:
            LEARNING_RATE *= 1.05
    return thetas

# ----------

def ft_train(x_km: list, y_price: list, output: bool = True) -> dict:
    
    """
    Trains a linear regression model using the provided data points by iterative
    gradient descent to update theta0 and theta1 based on the training data.
    
    Args:
        x_km (list): List of mileage values (independent variable).
        y_price (list): List of corresponding price values (dependent variable).
        output (bool): Print output messages (default: True).
    
    Returns:
        histories (dict): Dictionary containing the history of theta0, theta1 and loss.
    """

    thetas: list = [0.0, 0.0]
    loss: list = []
    histories: dict = {'theta0': [], 'theta1': [], 'loss': []}

    if output: print(message('2. Training model...'), end='\r')
    for _ in range(ITERATIONS):
        tmp: list = [0.0, 0.0]
        prediction: float = 0.0
        for km, price in zip(x_km, y_price):
            prediction = thetas[1] * km + thetas[0]
            tmp[0] += prediction - price
            tmp[1] += (prediction - price) * km
        thetas[0] -= tmp[0] / len(x_km) * LEARNING_RATE
        thetas[1] -= tmp[1] / len(y_price) * LEARNING_RATE
        histories['theta0'].append(thetas[0])
        histories['theta1'].append(thetas[1])
        histories['loss'].append(ft_loss(thetas, x_km, y_price))
        thetas = ft_adjust(x_km, y_price, thetas, tmp, loss)
    if output: print(message('2. Training model... Done √'))
    return histories

# ----------

def ft_plot(x_km: list, y_price: list, thetas: list, histories: dict, output: bool = True) -> None:

    """
    Plots the data points and the linear regression model.

    Args:
        x_km (list): List of mileage values (independent variable).
        y_price (list): List of corresponding price values (dependent variable).
        thetas (list): List containing theta0 and theta1.
        histories (dict): Dictionary containing the history of theta0, theta1 and loss.
        output (bool): Print output messages (default: True).
    
    Returns:
        None
    """

    y_plot: list = []
    x_plot: list = [float(min(x_km)), float(max(x_km))]

    if output: print(message('\n4. Plotting data points and linear regression model...'), end='\r')
    for elem in x_plot:
        elem = thetas[1] * ft_normalize_value(x_km, elem) + thetas[0]
        y_plot.append(ft_denormalize_value(y_price, elem))
    fig, axes = pyplot.subplots(nrows=2, ncols=2, figsize=(10, 8))
    fig.canvas.manager.set_window_title("MBOY'S LINEAR REGRESSION") 
    axes[0, 0].plot(x_km, y_price, color=(240 / 255, 128 / 255, 128 / 255), marker='o', linestyle='None')
    axes[0, 0].plot(x_plot, y_plot, color=(248 / 255, 173 / 255, 157 / 255))
    axes[0, 0].set_xlabel('Kilometers', fontdict={'family': 'arial', 'size': 10})
    axes[0, 0].set_ylabel('Prices', fontdict={'family': 'arial', 'size': 10})
    axes[0, 0].set_title('Linear regression model', fontdict={'family': 'arial', 'size': 12})
    axes[0, 1].plot(range(ITERATIONS), histories['loss'], color=(240 / 255, 128 / 255, 128 / 255))
    axes[0, 1].set_xlabel('Iterations', fontdict={'family': 'arial', 'size': 10})
    axes[0, 1].set_ylabel('Loss', fontdict={'family': 'arial', 'size': 10})
    axes[0, 1].set_title('Loss Evolution', fontdict={'family': 'arial', 'size': 12})
    axes[1, 0].plot(range(ITERATIONS), histories['theta0'], color=(240 / 255, 128 / 255, 128 / 255))
    axes[1, 0].set_xlabel('Iterations', fontdict={'family': 'arial', 'size': 10})
    axes[1, 0].set_ylabel('Theta0', fontdict={'family': 'arial', 'size': 10})
    axes[1, 0].set_title('Theta0 Evolution', fontdict={'family': 'arial', 'size': 12})
    axes[1, 1].plot(range(ITERATIONS), histories['theta1'], color=(240 / 255, 128 / 255, 128 / 255))
    axes[1, 1].set_xlabel('Iterations', fontdict={'family': 'arial', 'size': 10})
    axes[1, 1].set_ylabel('Theta1', fontdict={'family': 'arial', 'size': 10})
    axes[1, 1].set_title('Theta1 Evolution', fontdict={'family': 'arial', 'size': 12})
    pyplot.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.tight_layout(pad=3.0)
    if output: print(message('4. Plotting data points and linear regression model... Done √'))
    pyplot.show()


# Main function
# -------------

def ft_main(args: list) -> None:

    """
        Main function.

        Args:
            args (list): List of arguments that can be either:
                path (str): Optional argument to specify path to csv file (default: 'data.csv').
                -bonus (bool): Optional argument to enable bonus mode (default: False).
        
        Returns:
            None
    """
    
    global BONUS
    x_km: list = None
    y_price: list = None
    thetas: tuple = None
    histories: dict = None
    data: pandas.DataFrame = None

    if '-bonus' in args:
        BONUS = True
        args.remove('-bonus')
    if len(args) > 1:
        raise Exception("Please provide path to CSV or/and -bonus flag only, or no arguments at all to use default path 'data.csv'.")
    data = ft_load() if len(args) == 0 else ft_load(args[0])
    x_km = data['km'].tolist()
    y_price = data['price'].tolist()
    histories = ft_train(ft_normalize_list(x_km), ft_normalize_list(y_price))
    thetas = histories['theta0'][-1], histories['theta1'][-1]
    ft_save(thetas)
    print(message('\nFinal thetas value :'))
    print(message(f'   - Theta0: { thetas[0] }'))
    print(message(f'   - Theta1: { thetas[1] }'))
    if BONUS: ft_plot(x_km, y_price, thetas, histories)


# Main
# ----

if __name__ == '__main__':

    try:
        print(header(f'___ ___   _   _ _  _ __ ___ ___   ___ __ ___ ___ __ ___ ___ _ ___ _  _'))
        print(header(f'|_   |    |   | |\ | |_ |_| |_|   |_| |_ | _ |_| |_ |_  |_  | | | |\ |'))
        print(header(f'|    |    |__ | | \| |_ | | | \   | \ |_ |_| | \ |_ __| __| | |_| | \|\n'))
        print(header(f'--------------------------- TRAINING MODEL ---------------------------\n'))
        ft_main(sys.argv[1:])
        sys.exit(0)
    
    except (KeyboardInterrupt, EOFError): print(error('[ WARNING ]: Program interrupted by user.'))
    except Exception as exc:
        print(error(f'[ ERROR ]: { exc }'))
        sys.exit(1)
