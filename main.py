import utils.data_parser as data_parser
from utils.Country import Country
import utils.io as io
from utils.Optimizers import Optimizers
import utils.evaluation as evaluation
from utils.Evaluations import Evaluation
import utils.visualisation as visualisation
from utils.visualisation import Visualizations
from utils.Type import Type
from utils.Settings import Settings

import os
from os.path import join, abspath, dirname
from pathlib import Path
from enum import Enum

import time
import atexit

import polars as pl
import datetime as dt
import numpy as np
import argparse

from NIPA import NIPA
from NIPA_ALL import NIPA as NIPA_ALL

from sklearn.exceptions import ConvergenceWarning
import sklearn.metrics as metrics
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

LAST_DAY_PATH = join(Path(abspath(dirname(__file__))), "training-data")
DATA_PATH = join(Path(abspath(dirname(__file__))), "data")


def get_args():
    """
    Parse and return command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train and predict the NIPA model')
    parser.add_argument(
        '--country', type=str, default='mexico',
        choices=['mexico', 'hubei', 'netherlands'],
        help='Country to use [mexico, hubei, netherlands]')
    parser.add_argument(
        '--optimizers', type=str, default='cv,dsa',
        help='Optimizers to compare, separated by a comma [cv, cv_own, gsa, dsa]')
    parser.add_argument(
        '--visuals', type=str, default='all_pred,all_eval',
        help='Visualisations to show, separated by a comma [all, all_pred, all_eval, heatmap, optimizer_pred, optimizer_eval]')
    parser.add_argument(
        '--visual_days', type=int, default=30,
        help='Amount of days to show on the prediction visualization [default is 30]')
    parser.add_argument(
        '--evaluations', type=str, default='mse,smape',
        help='Evaluations to compare, separated by a comma [mse, mape, smape]')
    parser.add_argument(
        '--type', type=str, default='original',
        choices=['original', 'dynamic'],
        help='Type of NIPA to use [original, dynamic (NOT IMPLEMENTED)]')
    parser.add_argument(
        '--n_days', type=int, default=None,
        help='Number of days to iterate the model over [default is all days]')
    parser.add_argument(
        '--train_days', type=int, default=None,
        help='Number of days to train on when using Dynamic NIPA')
    parser.add_argument(
        '--pred_days', type=str, default="3",
        help='Number of days to predict; can be multiple days separated by a comma')
    parser.add_argument(
        '--compensate_fluctuations', type=bool, default=True,
        help='Compensate for fluctuations in the data where each weekend has lower reported cases')
    parser.add_argument(
        '--predict', type=bool, default=False,
        help='Only predict the next days based on the trained model')
    parser.add_argument(
        '--random', type=bool, default=False,
        help='Randomize the seed for LASSO and simulated annealing used for the NIPA (otherwise random_state=42 is used)')
    return parser.parse_args()


def save_computation_time(time_elapsed, optimizer, country):
    """
    Save the computation time to a file.

    Parameters
    ----------
    time_elapsed : float
        Computation time in seconds.
    optimizer : Optimizers
        Optimizer used.
    country : Country
        Country of the model.
    """
    path = join(DATA_PATH, optimizer.name, country.name, "computation_time.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a+') as outfile:
        outfile.write(f"{time_elapsed}\n")


def to_hours(seconds):
    """
    Convert seconds to a human-readable string of hours, minutes, and seconds.

    Parameters
    ----------
    seconds : int
        Time in seconds.

    Returns
    -------
    str
        Human-readable time string.
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours} hours, {minutes} minutes and {seconds} seconds"


def start(I_data, dates, regions, settings):
    """
    Start the training and prediction process.

    Parameters
    ----------
    I_data : DataFrame
        The input data.
    dates : list
        List of dates.
    regions : list
        List of regions.
    settings : Settings
        Settings object containing configuration.
    """
    country = settings.country
    optimizers = settings.optimizers
    evaluations = settings.evaluations
    type = settings.type
    n_days = settings.n_days
    n_pred_days = settings.pred_days
    random = settings.random

    # Iterate over each prediction day setting
    for idx, pred_days in enumerate(n_pred_days):
        # Get the training data based on the model type
        if type == Type.ORIGINAL_NIPA:
            x, y, train_dates = data_parser.get_NIPA_data(I_data, dates, n_days[idx], pred_days=pred_days, over_days=-1)
        elif type == Type.DYNAMIC_NIPA:
            continue  # NOT YET IMPLEMENTED

        # Iterate over each optimizer
        for optimizer in optimizers:
            nipa = (NIPA_ALL if optimizer == Optimizers.DUAL_SIMULATED_ANNEALING_ALL else NIPA)(
                data=x, regions=regions, country=country, type=type, dates=dates, parameter_optimizer=optimizer, random=random)

            train_start = time.time()

            # Training and prediction loop for each day k
            for k in range(1, len(x.keys()) + 1):
                ts = time.time()

                train_idx = len(train_dates[k]) - pred_days
                train_days = train_dates[k][:train_idx]
                predict_days = train_dates[k][train_idx:]

                print(f"Training for k={k} for {train_days[0].strftime('%d-%m-%Y')} to {train_days[-1].strftime('%d-%m-%Y')}")
                B = nipa.train(train_days)

                print(f"Predicting {predict_days[0].strftime('%d-%m-%Y')} to {predict_days[-1].strftime('%d-%m-%Y')}")
                pred_I, pred_R = nipa.predict(B, predict_days, nipa.curing_probs)

                suffix = f"_{np.random.randint(0, 1000)}" if random else ""
                io.save_results_data(pred_I, country, type, nipa.parameter_optimizer, train_days[-1], f"I_pred{suffix}")
                io.save_results_data(pred_R, country, type, nipa.parameter_optimizer, train_days[-1], f"R_pred{suffix}")

                evaluation.get_prediction_evaluation(I_data, dates, train_days[-1], country, type, nipa.parameter_optimizer, evaluations)

                te = time.time()
                print(f"Training for k={k} on date {train_days[0].strftime('%d-%m-%Y')} to {train_days[-1].strftime('%d-%m-%Y')} took {te - ts} seconds")
                print()

            train_end = time.time()
            print(f"Training took {to_hours(train_end - train_start)}")
            save_computation_time(train_end - train_start, optimizer, country)


def predict(I_data, regions, dates_list, settings, final_dates, visualisations=[Visualizations.ALL], save=False):
    """
    Predict the next days based on the trained model.

    Parameters
    ----------
    I_data : DataFrame
        The input data.
    regions : list
        List of regions.
    dates_list : list
        List of dates.
    settings : Settings
        Settings object containing configuration.
    final_dates : list
        List of final dates for prediction.
    visualisations : list, optional
        List of visualisations to show, by default [Visualizations.ALL]
    save : bool, optional
        Whether to save the results, by default False.
    """
    country = settings.country
    optimizers = settings.optimizers
    evaluations = settings.evaluations
    type = settings.type
    n_pred_days = settings.pred_days
    n_days = settings.n_days
    visual_days = settings.visual_days

    all_predictions = {}
    all_evaluations = {}

    # Iterate over each prediction day setting
    for idx, pred_days in enumerate(n_pred_days):
        data_date = final_dates[idx]
        count_days = n_days[idx] + pred_days

        # Limit the dates to the visual_days
        if len(list(dates_list.keys())) > visual_days:
            dates = {k: dates_list[k] for k in list(dates_list.keys())[(count_days - visual_days):(count_days)]}
        else:
            dates = dates_list

        B, curing_probs = [], []

        # Load the trained model parameters
        try:
            for optimizer in optimizers:
                B.append(pl.read_csv(io.get_results_data_path(country, type, optimizer, data_date) + "B.csv"))
                curing_probs.append(pl.read_csv(io.get_results_data_path(country, type, optimizer, data_date) + "curing_probs.csv"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No trained model found for optimizer {optimizer.name} and date {data_date}") from e

        nipas, predictions, pred_dates = [], [], []
        for k in dates:
            date = dates[k]
            if final_dates[idx] < date <= final_dates[idx] + dt.timedelta(days=pred_days):
                pred_dates.append(date)

        evaluations_array = []

        # Iterate over each optimizer
        for i, optimizer in enumerate(optimizers):
            evaluations_df = pl.DataFrame({"k": pred_dates})
            nipa = NIPA(I_data, regions, country, type, dates, parameter_optimizer=optimizer)
            nipas.append(nipa)

            B_data = B[i]
            curing_probs_data = curing_probs[i]

            if Visualizations.HEATMAP in visualisations:
                visualisation.heatmap_B(B)

            pred_I, pred_R = nipa.predict(B_data, pred_dates, curing_probs_data)
            predictions.append(pred_I)

            if Visualizations.ALL in visualisations or Visualizations.ITER_PREDICTIONS in visualisations:
                visualisation.plot_predictions(I_data, pred_I, dates, regions[:5], fig_path=f"figures/{country.name}/{optimizer}_{pred_days}.png", amount_days=25)

            if save:
                io.save_results_data(pred_I, country, type, nipa.parameter_optimizer, data_date, "I_pred")
                io.save_results_data(pred_R, country, type, nipa.parameter_optimizer, data_date, "R_pred")

            print(f"Predicting {pred_days} days for {data_date.strftime('%d-%m-%Y')} with optimizer {optimizer.name}:")
            eval = evaluation.get_prediction_evaluation(I_data, dates, data_date, country, type, nipa.parameter_optimizer, evaluations)
            
            for key in eval.keys():
                evaluations_df = evaluations_df.with_columns(pl.Series(key, eval[key]))

            evaluations_array.append(evaluations_df)
            optimizers_str = [opt.name for opt in optimizers]

        all_predictions[pred_days] = predictions
        all_evaluations[pred_days] = evaluations_array

        if Visualizations.ALL in visualisations or Visualizations.ITER_PREDICTIONS in visualisations:
            if len(optimizers) > 1:
                visualisation.plot_multiple_predictions(I_data, predictions, optimizers, dates, regions[:5], fig_path=f"figures/{country.name}/multiple_{'_'.join(optimizers_str)}_{pred_days}.png", amount_days=25)

    if Visualizations.ALL in visualisations or Visualizations.ALL_PREDICTIONS in visualisations:
        visualisation.plot_all_predictions(I_data, all_predictions, optimizers, dates, regions[:5], fig_path=f"figures/{country.name}/all_predictions_{'_'.join(optimizers_str)}_{','.join([str(days) for days in n_pred_days])}.png", amount_days=25)
    if Visualizations.ALL in visualisations or Visualizations.ALL_EVALUATIONS in visualisations:
        visualisation.plot_evaluations(all_evaluations, optimizers, fig_path=f"figures/{country.name}/all_evaluations_{'_'.join(optimizers_str)}_{','.join([str(days) for days in n_pred_days])}.png", evaluations=evaluations)


if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()

    # Initialize settings from parsed arguments
    settings = Settings(args)

    country = settings.country
    n_days = settings.n_days
    prediction_days = settings.pred_days
    compensate_fluctuations = settings.compensate_fluctuations
    visualizations = settings.visualizations

    # Load data
    I_data, dates, regions = data_parser.get_data(country=country, compensate_fluctuations=compensate_fluctuations)

    # Set the number of training days if not specified
    if n_days is None:
        n_days = [len(dates) - pred_days for pred_days in prediction_days]
        settings.set_n_days(n_days)
    else:
        if n_days > len(dates):
            raise ValueError(f"n_days ({n_days}) is greater than the total amount of days ({len(dates)})")
        n_days = [n_days - pred_days for pred_days in prediction_days]
        settings.set_n_days(n_days)

    settings.print()

    # Start training and prediction process
    if not args.predict:
        start(I_data, dates, regions, settings)
    predict(I_data, regions, dates, settings, final_dates=[dates[n_day] for n_day in n_days], visualisations=visualizations)
