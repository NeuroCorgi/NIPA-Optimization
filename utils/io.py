import polars as pl
from os.path import join, abspath, dirname, isfile, isdir
from pathlib import Path


ROOT_PATH = Path(abspath(dirname(__file__))).parent
DATA_PATH = join(ROOT_PATH, 'network-data')
RESULTS_PATH = join(ROOT_PATH, 'results')

def save_train_data(dataframe, country, filename):
    path = join(DATA_PATH, country.name)
    if not isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    full_path = join(path, f"{filename}.csv")    
    dataframe.write_csv(full_path)
    print(f"Saved to {full_path}")

def get_train_data_path(country):
    return f"{join(DATA_PATH, country.name)}\\"

def save_results_data(dataframe, country, NIPA_type, optimizer, date, filename):
    path = join(RESULTS_PATH, country.name, NIPA_type.name, optimizer.name, date.strftime("%d-%m-%Y"))
    if not isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    full_path = join(path, f"{filename}.csv")    
    dataframe.write_csv(full_path)
    print(f"Saved to {full_path}")

def get_results_data_path(country, NIPA_type, optimizer, date):
    return f"{join(RESULTS_PATH, country.name, NIPA_type.name, optimizer.name, date.strftime("%d-%m-%Y"))}\\"