import utils.data_parser as data_parser
import utils.show_data as show_data
from utils.Country import Country
import utils.io as io
from utils.Optimizers import Optimizers
import utils.evaluation as evaluation
import utils.visualisation as visualisation
from utils.Type import Type
from utils.Settings import Settings

from os.path import join, abspath, dirname, isfile, isdir
from pathlib import Path
from enum import Enum

import time
import atexit

import polars as pl
import datetime as dt
import numpy as np
import argparse

from NIPA import NIPA

from sklearn.exceptions import ConvergenceWarning
import sklearn.metrics as metrics
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

LAST_DAY_PATH = join(Path(abspath(dirname(__file__))), "training-data/")

def get_args():
    parser = argparse.ArgumentParser(description='Train and predict the NIPA model')
    parser.add_argument(
        '--country', type=str, default='mexico', 
        choices=['mexico', 'hubei', 'netherlands'],
        help='Country to use (mexico, hubei, netherlands)')
    
    parser.add_argument(
        '--optimizer', type=str, default='lassocv', 
        choices=['lassocv', 'lasso', 'sa'],
        help='Optimizer to use (lassocv, lasso, sa (simulated annealing))'
        )
    
    parser.add_argument(
        '--type', type=str, default='original', 
        choices=['original', 'dynamic'],
        help='Type of NIPA to use (original, [NOT READY] dynamic)'
        )
    
    parser.add_argument(
        '--n_days', type=int, default=None, 
        help='Number of days to train on (default is all days - prediction days)'
        )
    
    parser.add_argument(
        '--train_days', type=int, default=None, 
        help='Number of days to train on when using Dynamic NIPA'
        )
    
    parser.add_argument(
        '--pred_days', type=int, default=3, 
        help='Number of days to predict (n+prediction_days)'
        )
    
    parser.add_argument(
        '--compensate_fluctuations', type=bool, default=True, 
        help='Compensate for fluctuations in the data where each weekend has lower reported cases (thus 7 week rolling-mean)'
        )
    return parser.parse_args()

#MARK: TODO
# TODO: Implement evaluation of the prediction
class Evaluation(Enum):
    MSE = 1
    MAPE = 2
    sMAPE = 3

# Save the day and time when the program is stopped
def save_day(k, time):
    with open(LAST_DAY_PATH, 'w+') as outfile:
        outfile.write('%d \n' % k)
        outfile.write('%s' % time)
    
    print(f"Training took {to_hours(time)}")

# Convert seconds to hours, minutes and seconds
def to_hours(seconds):
    hours = seconds//3600
    seconds = seconds%3600
    minutes = seconds//60
    seconds = seconds%60
    return f"{hours} hours, {minutes} minutes and {seconds} seconds"

# Check if the model has already been trained with the specific parameters
def check_if_already_trained(country, optimizer, type):
    global LAST_DAY_PATH 
    LAST_DAY_PATH = LAST_DAY_PATH + f"/{country.name}/trained_info_{optimizer.name.lower()}_{type.name.lower()}"

    if not isfile(LAST_DAY_PATH):
        last_day = 1
        already_trained = 0
    else:
        with open(LAST_DAY_PATH, 'r') as infile:
            last_day = int(infile.readline())
            already_trained = float(infile.readline())
            atexit.register(save_day, last_day)
    
    return already_trained, last_day
    
# Start the training and prediction
def start(I_data, dates, regions, country=Country.Mexico, optimizer=Optimizers.LASSOCV, type=Type.ORIGINAL_NIPA, n_days=875, pred_days=3):
    if type == Type.ORIGINAL_NIPA:
        x, y, train_dates = data_parser.get_NIPA_data(I_data, dates, n_days)
        until = 2
    elif type == Type.DYNAMIC_NIPA:
        x, y, train_dates = data_parser.get_dynamic_NIPA_data(I_data, dates, n=n_days)
        until = len(x.keys())   # Run the program until the end of the dates

    # Check whether previous run exists
    already_trained, last_day = check_if_already_trained(country, optimizer, type)

    nipa = NIPA(data=x, regions=regions, country=country, type=type, dates=dates, parameter_optimizer=optimizer)

    train_start = time.time()-already_trained

    # Where k = 1 for Original NIPA and k = 1, ..., n for Dynamic NIPA and N_DAYS
    for k in range(last_day, until):
        # Save the day and time when the program is stopped
        atexit.unregister(save_day)
        atexit.register(save_day, k, time.time()-train_start)

        ts = time.time()        # Start time for training day k

        # Get the training and prediction dates
        train_idx = len(train_dates[k])-pred_days   
        train_days = train_dates[k][:train_idx]
        predict_days = train_dates[k][train_idx:]

        print(f"Training for k={k} for {train_days[0].strftime('%d-%m-%Y')} to {train_days[-1].strftime('%d-%m-%Y')}")
        B = nipa.train(train_days)

        print(f'Predicting {predict_days[0].strftime('%d-%m-%Y')} to {predict_days[-1].strftime('%d-%m-%Y')}')
        pred_I, pred_R = nipa.predict(B, predict_days, nipa.curing_probs)

        io.save_results_data(pred_I, country, type, nipa.parameter_optimizer, train_days[0], "I_pred")
        io.save_results_data(pred_R, country, type, nipa.parameter_optimizer, train_days[0], "R_pred")

        evaluation.get_prediction_evaluation(I_data, dates, train_days[0], country, type, nipa.parameter_optimizer)

        te = time.time()
        print(f"Training for k={k} on date {train_days[0].strftime('%d-%m-%Y')} to {train_days[-1].strftime('%d-%m-%Y')} took {te-ts} seconds")
        print()

        save_day(k, te-ts)
    
    train_end = time.time()

    print(f"Training took {to_hours(train_end-train_start)}")


# Predict the next days
def predict(I_data, regions, dates, country, optimizer, type, final_date, n_pred_days=3):
    if type == Type.ORIGINAL_NIPA:
        data_date = dates[1]
    else:
        data_date = final_date

    print(final_date)

    nipa = NIPA(I_data, regions, country, type, dates, parameter_optimizer=optimizer)

    B = pl.read_csv(io.get_results_data_path(country, type, optimizer, data_date) + "B.csv")
    curing_probs = pl.read_csv(io.get_results_data_path(country, type, optimizer, data_date) + "curing_probs.csv")

    # visualisation.heatmap_B(B)

    pred_days = []
    for k in dates:
        date = dates[k]
        if date > final_date and date <= final_date + dt.timedelta(days=n_pred_days):
            pred_days.append(date)

    pred_I, pred_R = nipa.predict(B, pred_days, curing_probs)

    visualisation.plot_predictions(I_data, pred_I, dates, regions, fig_path=f"figures/{country.name}/{optimizer}_{n_pred_days}.png", amount_days=25)

    # io.save_results_data(pred_I, country, type, nipa.parameter_optimizer, data_date, "I_pred")
    # io.save_results_data(pred_R, country, type, nipa.parameter_optimizer, data_date, "R_pred")

    evaluation.get_prediction_evaluation(I_data, dates, data_date, country, type, nipa.parameter_optimizer)


if __name__ == '__main__':
    args = get_args()

    settings = Settings(args)

    # Program data
    country = settings.country                                      # Country to use (Mexico, Netherlands)
    optimizer = settings.optimizer                                  # Optimizer to use (LASSOCV, LASSO_OWN, SIMULATED_ANNEALING)
    NIPA_type = settings.type                                       # Type of NIPA to use (ORIGINAL_NIPA, N_DAYS, DYNAMIC_NIPA)
    n_days = settings.n_days                                        # Beitskes final date is 875
    train_days = settings.train_days                                # Train days when using Dynamic NIPA
    prediction_days = settings.pred_days                            # Days to predict (n+prediction_days)
    compensate_fluctuations = settings.compensate_fluctuations      # Compensate for fluctuations in the data where each weekend has lower reported cases (thus 7 week rolling-mean)

    I_data, dates, regions = data_parser.get_data(country=country, compensate_fluctuations=compensate_fluctuations)

    if n_days is None:
        n_days = len(dates)-prediction_days
    else:
        if n_days+prediction_days > len(dates):
            raise Exception("Days to train on + prediction days is larger than the amount of days in the data")

    print(f"{len(dates)} n: {n_days} pred: {prediction_days}")
    
    start(I_data, dates, regions, country, optimizer=optimizer, type=NIPA_type, n_days=n_days, pred_days=prediction_days)
    predict(I_data, regions, dates, country, optimizer, NIPA_type, dates[n_days], n_pred_days=prediction_days)

    