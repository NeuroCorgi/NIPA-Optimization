import numpy as np
import polars as pl
from sklearn import metrics

import utils.io as io

def sMAPE(true, pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = np.abs(pred-true) / ((np.abs(true) + np.abs(pred))/2)
    
    tmp[np.isnan(tmp)] = 0
    sMAPE = np.sum(tmp) / len(tmp)
    return sMAPE

def MSE(true, pred):
    return metrics.mean_squared_error(true, pred)
        
def MAPE(true, pred):
    return metrics.mean_absolute_percentage_error(true, pred)


# Evaluate the prediction 
def get_prediction_evaluation(I_data, dates, date, country, type, optimizer):
    pred_I = io.get_results_data_path(country, type, optimizer, date)

    pred_I = pl.read_csv(pred_I + "I_pred.csv")

    # Go through each day and calculate the MSE, MAPE and sMAPE
    for idx, col in enumerate(pred_I[:, 1:].get_columns()):
        day = col.name
        pred_Ik = np.array(pred_I.get_column(day).to_list())
        yk = np.array(I_data.get_column(day).to_list())

        MSE_pred_I = MSE(yk, pred_Ik)
        MAPE_pred_I = MAPE(yk, pred_Ik)
        sMAPE_pred_I = sMAPE(yk, pred_Ik)

        print(f"MSE for I for prediction day {day} ({dates[int(day)].strftime('%d-%m-%Y')}): {MSE_pred_I}")
        print(f"MAPE for I for prediction day {day} ({dates[int(day)].strftime('%d-%m-%Y')}): {MAPE_pred_I}")
        print(f"sMAPE for I for prediction day {day} ({dates[int(day)].strftime('%d-%m-%Y')}): {sMAPE_pred_I}\n")