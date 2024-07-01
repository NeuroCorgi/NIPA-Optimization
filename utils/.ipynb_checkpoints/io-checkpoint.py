import polars as pl
from os.path import join, abspath, dirname, isfile, isdir
from pathlib import Path


ROOT_PATH = Path(abspath(dirname(__file__))).parent
DATA_PATH = join(ROOT_PATH, 'network-data')
RESULTS_PATH = join(ROOT_PATH, 'results')

def save_train_data(dataframe, country, filename):
    """
    Save the training data to a CSV file.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to save.
    country : Country
        The country associated with the data.
    filename : str
        The name of the file to save the data as.
    """
    path = join(DATA_PATH, country.name)
    if not isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    full_path = join(path, f"{filename}.csv")    
    dataframe.write_csv(full_path)
    print(f"Saved to {full_path}")

def get_train_data_path(country):
    """
    Get the path to the training data directory for a given country.

    Parameters
    ----------
    country : Country
        The country associated with the data.

    Returns
    -------
    str
        The path to the training data directory.
    """
    return f"{join(DATA_PATH, country.name)}"

def save_results_data(dataframe, country, NIPA_type, optimizer, date, filename):
    """
    Save the results data to a CSV file.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to save.
    country : Country
        The country associated with the data.
    NIPA_type : Type
        The type of NIPA model.
    optimizer : Optimizer
        The optimizer used in the model.
    date : datetime.date
        The date associated with the data.
    filename : str
        The name of the file to save the data as.
    """

    path = join(RESULTS_PATH, country.name, NIPA_type.name, optimizer.name, date.strftime('%d-%m-%Y'))
    if not isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    full_path = join(path, f"{filename}.csv")    
    dataframe.write_csv(full_path)
    print(f"Saved to {full_path}")

def get_results_data_path(country, NIPA_type, optimizer, date):
    """
    Get the path to the results data directory for a given configuration.

    Parameters
    ----------
    country : Country
        The country associated with the data.
    NIPA_type : Type
        The type of NIPA model.
    optimizer : Optimizer
        The optimizer used in the model.
    date : datetime.date
        The date associated with the data.

    Returns
    -------
    str
        The path to the results data directory.
    """
    return f"{join(RESULTS_PATH, country.name, NIPA_type.name, optimizer.name, date.strftime('%d-%m-%Y'))}"