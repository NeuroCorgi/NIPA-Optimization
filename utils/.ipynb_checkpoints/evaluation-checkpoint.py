import numpy as np
import polars as pl
from sklearn import metrics
from os.path import join

import utils.io as io
from utils.Evaluations import Evaluation

def sMAPE(true, pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (sMAPE).

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True values.
    pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        sMAPE value.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = np.abs(pred-true) / ((np.abs(true) + np.abs(pred))/2)
    
    tmp[np.isnan(tmp)] = 0
    sMAPE = np.sum(tmp) / len(tmp)
    return sMAPE

def MSE(true, pred):
    """
    Calculate the Mean Squared Error (MSE).

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True values.
    pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        MSE value.
    """
    return metrics.mean_squared_error(true, pred)
        
def MAPE(true, pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True values.
    pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        MAPE value.
    """
    return metrics.mean_absolute_percentage_error(true, pred)

def get_evaluation_metrics(true, pred, evaluation):
    """
    Get the specified evaluation metric.

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True values.
    pred : array-like of shape (n_samples,)
        Predicted values.
    evaluation : Evaluation
        The evaluation metric to calculate.

    Returns
    -------
    float
        The calculated evaluation metric.
    """
    if evaluation == Evaluation.MSE:
        return MSE(true, pred)
    elif evaluation == Evaluation.MAPE:
        return MAPE(true, pred)
    elif evaluation == Evaluation.sMAPE:
        return sMAPE(true, pred)
    else:
        raise Exception("Invalid evaluation metric")

def get_prediction_evaluation(I_data, dates, date, country, type, optimizer, evaluations):
    """
    Evaluate the prediction using specified metrics.

    Parameters
    ----------
    I_data : DataFrame
        The actual infection data.
    dates : dict
        A dictionary mapping indices to dates.
    date : datetime.date
        The date for which the evaluation is performed.
    country : Country
        The country for which the evaluation is performed.
    type : Type
        The type of NIPA model.
    optimizer : Optimizer
        The optimizer used in the model.
    evaluations : list of Evaluation
        The list of evaluation metrics to calculate.

    Returns
    -------
    dict
        A dictionary with evaluation metrics for each prediction day.
    """

    pred_I_path = io.get_results_data_path(country, type, optimizer, date)
    pred_I = pl.read_csv(join(pred_I_path, "I_pred.csv"))

    eval_results = {metric.name: [] for metric in evaluations}

    # Calculate evaluation metrics for each prediction day
    for idx, col in enumerate(pred_I[:, 1:].get_columns()):
        day = col.name
        date = dates[int(day)].strftime('%d-%m-%Y')
        pred_Ik = np.array(pred_I.get_column(day).to_list())
        yk = np.array(I_data.get_column(day).to_list())

        print(f"Day {day} ({date}):")
        for evaluation in evaluations:
            metric_value = get_evaluation_metrics(yk, pred_Ik, evaluation)
            eval_results[evaluation.name].append(metric_value)
            print(f"\t{evaluation.name}: {metric_value}")
        
        print()

    return eval_results