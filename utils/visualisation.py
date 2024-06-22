import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import os
from os.path import join
from enum import Enum
import datetime as dt

from utils.Optimizers import Optimizers
from utils.Evaluations import Evaluation

class Visualizations(Enum):
    """
    Enum class for representing different types of visualizations.

    Attributes
    ----------
    ALL : int
        Enum value for all visualizations.
    ALL_PREDICTIONS : int
        Enum value for all predictions.
    ALL_EVALUATIONS : int
        Enum value for all evaluations.
    HEATMAP : int
        Enum value for heatmap visualization.
    ITER_PREDICTIONS : int
        Enum value for iterative predictions.
    ITER_EVALUATIONS : int
        Enum value for iterative evaluations.
    """
    ALL = 1
    ALL_PREDICTIONS = 2
    ALL_EVALUATIONS = 3
    HEATMAP = 4
    ITER_PREDICTIONS = 5
    ITER_EVALUATIONS = 6



def get_evaluation_labels(evaluation):
    """
    Get the label for the specified evaluation metric.

    Parameters
    ----------
    evaluation : Evaluation
        The evaluation metric.

    Returns
    -------
    str
        The label for the evaluation metric.
    """
    if evaluation == Evaluation.MSE:
        return "Mean Squared Error (MSE)"
    elif evaluation == Evaluation.MAPE:
        return "Mean Absolute Percentage Error (MAPE)"
    elif evaluation == Evaluation.sMAPE:
        return "Symmetric Mean Absolute Percentage Error (sMAPE)"
    else:
        raise Exception("Invalid evaluation metric")
    
def get_optimizer_labels(optimizer):
    if optimizer == Optimizers.CV:
        return "Cross-Validation"
    if optimizer == Optimizers.GENERALIZED_SIMULATED_ANNEALING:
        return "Generalized Simulated Annealing"
    if optimizer == Optimizers.DUAL_SIMULATED_ANNEALING:
        return "Dual Simulated Annealing"

def get_summed_data(true_data, pred_data, regions, final_idx):
    """
    Calculate the summed true and predicted data for each region.

    Parameters
    ----------
    true_data : DataFrame
        The true data.
    pred_data : DataFrame
        The predicted data.
    regions : list of str
        The list of region names.
    final_idx : int
        The final index to consider.

    Returns
    -------
    tuple of dict
        Two dictionaries containing the summed true and predicted data for each region.
    """
    region_summed_true_data = {}
    region_summed_pred_data = {}

    for region in regions:
        summed_true_data = {}
        summed_pred_data = {}
        for i in range(final_idx):
            if i == 0:
                summed_true_data[i] = 0

            k = i+1

            summed_true_data[k] = summed_true_data[i] + true_data.filter(pl.col('regions') == region).get_column(str(k)).to_list()[0]

            if str(k) == pred_data.columns[1]:
                summed_pred_data[k] = summed_true_data[i] + pred_data.filter(pl.col('regions') == region).get_column(str(k)).to_list()[0]
            elif str(k) in pred_data.columns:
                summed_pred_data[k] = summed_pred_data[i] + pred_data.filter(pl.col('regions') == region).get_column(str(k)).to_list()[0]

        region_summed_true_data[region] = summed_true_data
        region_summed_pred_data[region] = summed_pred_data

    return region_summed_true_data, region_summed_pred_data

def visualise_simulated_annealing(all_p, all_MSE, fig_path=None):
    """
    Visualize the simulated annealing process.

    Parameters
    ----------
    all_p : list of float
        List of regularization parameter values over time.
    all_MSE : list of float
        List of MSE values over time.
    fig_path : str, optional
        Path to save the figure. Default is None.
    """
    combined_p_MSE = zip(all_p, all_MSE)
    sorted_list = sorted(combined_p_MSE, key=lambda x: x[0])
    sorted_list = [item for item in sorted_list if item[1] is not 1.0]

    # Find the first index where MSE does not change significantly
    first_index_double = [idx for idx, item in enumerate(sorted_list) if (item[1] == sorted_list[idx-1][1] and idx > 50)][0]

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
    
    # Plot regularization parameter over time
    axs[0,0].plot(np.arange(len(all_p)), all_p, 'bo')
    axs[0,0].set(xlabel='Time', ylabel='regularization value',
        title='Regularization parameter over time')
    axs[0,0].set_yscale('log')
    axs[0,0].grid()

    # Plot MSE over time
    axs[0,1].plot(np.arange(len(all_MSE)), all_MSE, 'bo')
    axs[0,1].set(xlabel='Time', ylabel='MSE',
        title='MSE over time')
    axs[0,1].grid()

    p_sorted, MSE_sorted = zip(*sorted_list)

    # Plot regularization parameter against MSE
    axs[1,0].plot(p_sorted[:first_index_double], MSE_sorted[:first_index_double], 'b')
    axs[1,0].set(xlabel='Regularization parameter', ylabel='MSE',
        title='Regularization parameter and MSE')
    axs[1,0].grid()

    if fig_path is not None:
        fig.savefig(fig_path)
    plt.show()


def plot_p_vs_evaluation(all_p, all_evals, evaluation=Evaluation.MSE, fig_path=None):
    """
    Plot regularization parameter against evaluation metric.

    Parameters
    ----------
    all_p : list of float
        List of regularization parameter values.
    all_evals : list of float
        List of evaluation metric values.
    evaluation : Evaluation, optional
        The evaluation metric. Default is Evaluation.MSE.
    fig_path : str, optional
        Path to save the figure. Default is None.
    """
    eval_label = get_evaluation_labels(evaluation)

    combined_p_eval = zip(all_p, all_evals)
    sorted_list = sorted(combined_p_eval, key=lambda x: x[0])
    sorted_list = [item for item in sorted_list if item[1] is not 1.0]

    # Find the first index where the evaluation metric does not change significantly
    first_index_double = [idx for idx, item in enumerate(sorted_list) if (item[1] == sorted_list[idx-1][1] and idx > 50)]
    first_index_double = first_index_double[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    p_sorted, eval_sorted = zip(*sorted_list)

    # Plot regularization parameter against evaluation metric
    ax.plot(p_sorted[:first_index_double+10], eval_sorted[:first_index_double+10], 'b')
    ax.set(xlabel='Regularization parameter', ylabel=eval_label,
        title=f'Regularization parameter against {eval_label}')
    ax.grid()

    if fig_path is not None:
        fig.savefig(fig_path)
    plt.show()

def heatmap_B(B):
    """
    Plot a heatmap of the infection probability matrix B.

    Parameters
    ----------
    B : DataFrame
        The matrix B.
    """
    regions = B[:, 0].to_list()
    B = B.transpose(include_header=True, header_name="regions", column_names="regions")[:, 1:].to_numpy()

    B = np.round(B, 6)
    B[B == 0.0] = 0

    fig, ax = plt.subplots(figsize=(18, 18))
    im = ax.imshow(B, aspect='auto', vmin=0, vmax=0.01)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(regions)), labels=regions)
    ax.set_yticks(np.arange(len(regions)), labels=regions)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(regions)):
        for j in range(len(regions)):
            text = ax.text(j, i, B[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Heatmap of Matrix B")
    fig.tight_layout()
    plt.show()

def plot_predictions(I_data, pred, dates, regions, fig_path=None, amount_days=25, amount_regions=None):
    """
    Plot true vs. predicted data.

    Parameters
    ----------
    I_data : DataFrame
        The true infection data.
    pred : DataFrame
        The predicted infection data.
    dates : dict
        Dictionary of dates.
    regions : list of str
        List of region names.
    fig_path : str, optional
        Path to save the figure. Default is None.
    amount_days : int, optional
        Number of days to show. Default is 25.
    amount_regions : int, optional
        Number of regions to show. Default is None.
    """
    if amount_regions is not None:
        regions = regions[:(amount_regions)]

    fig, ax = plt.subplots(figsize=(10, 6))

    final_idx = int(pred.get_columns()[-1].name)
    amount_days = min(amount_days, final_idx - 1)

    true_data = I_data.filter(pl.col('regions').is_in(regions))
    pred_data = pred.filter(pl.col('regions').is_in(regions))

    region_summed_true_data, region_summed_pred_data = get_summed_data(true_data, pred_data, regions, final_idx)

    for region in regions:
        x_true, y_true = zip(*region_summed_true_data[region].items())
        print(x_true)
        x_true = [dates[x] for x in x_true[1:]]
        ax.plot(x_true, y_true[1:], 'bo-')

        x_pred, y_pred = zip(*region_summed_pred_data[region].items())
        x_pred = [dates[x] for x in x_pred]
        ax.plot(x_pred, y_pred, 'rs--')

    ax.set(xlabel='Timestep', ylabel='Infected',
        title='Predicted vs True')
    ax.grid()

    ax.legend(['True Data', 'Predicted (NIPA)'])
    plt.show()

    if fig_path is not None:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path)
        print(f'saving plotted graph to {fig_path}')

def plot_multiple_predictions(I_data, pred, pred_optimizers, dates, regions, fig_path=None, amount_days=25, amount_regions=None):
    """
    Plot true vs. multiple predicted data from different optimizers.

    Parameters
    ----------
    I_data : DataFrame
        The true infection data.
    pred : list of DataFrame
        The list of predicted infection data.
    pred_optimizers : list of Optimizers
        The list of optimizers used.
    dates : dict
        Dictionary of dates.
    regions : list of str
        List of region names.
    fig_path : str, optional
        Path to save the figure. Default is None.
    amount_days : int, optional
        Number of days to show. Default is 25.
    amount_regions : int, optional
        Number of regions to show. Default is None.
    """
    colors = ['rs--', 'g^--', 'y2--', 'm8--', 'cx--', 'kd--', 'w_--']

    if amount_regions is not None:
        regions = regions[:(amount_regions)]

    fig, ax = plt.subplots(figsize=(10, 6))

    final_idx = int(pred[0].get_columns()[-1].name)
    amount_days = min(amount_days, final_idx - 1)

    true_data = I_data.filter(pl.col('regions').is_in(regions))

    labels = []
    first = True

    for i, pred_data in enumerate(pred):
        labels.append(f"Predicted ({get_optimizer_labels(pred_optimizers[i])})")

        pred_data = pred[i].filter(pl.col('regions').is_in(regions))

        region_summed_true_data, region_summed_pred_data = get_summed_data(true_data, pred_data, regions, final_idx)

        no_legend = False

        for region in regions:
            # When first, plot the true data
            if first:
                x_true, y_true = zip(*region_summed_true_data[region].items())
                x_true = [dates[x] for x in x_true[1:]]
                ax.plot(x_true, y_true[1:], 'bo-', label="_nolegend_" if no_legend else 'True Data')

            x_pred, y_pred = zip(*region_summed_pred_data[region].items())
            x_pred = [dates[x] for x in x_pred]
            ax.plot(x_pred, y_pred, colors[i], label="_nolegend_" if no_legend else labels[i])

            no_legend = True

        first = False

    ax.set(xlabel='Timestep', ylabel='Infected',
        title='Predicted vs True')
    ax.grid()
    ax.legend(labels)
    plt.show()

    if fig_path is not None:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path)
        print(f'saving plotted graph to {fig_path}')

def plot_all_predictions(I_data, pred, pred_optimizers, dates, regions, fig_path=None, amount_days=25, amount_regions=None):
    """
    Plot true vs. all predicted data from different optimizers across multiple time horizons.

    Parameters
    ----------
    I_data : DataFrame
        The true infection data.
    pred : dict of list of DataFrame
        Dictionary of predicted infection data for different time horizons.
    pred_optimizers : list of Optimizers
        The list of optimizers used.
    dates : dict
        Dictionary of dates.
    regions : list of str
        List of region names.
    fig_path : str, optional
        Path to save the figure. Default is None.
    amount_days : int, optional
        Number of days to show. Default is 25.
    amount_regions : int, optional
        Number of regions to show. Default is None.
    """
    colors = ['rs--', 'g^--', 'y2--', 'm8--', 'cx--', 'kd--', 'w_--']
    graph_names = list('abcdefghijklmnopqrstuvwxyz')

    if amount_regions is not None:
        regions = regions[:(amount_regions)]

    cols = int(np.ceil(len(list(pred.keys()))/2))

    start_date = dates[list(dates.keys())[0]] - dt.timedelta(days=1)
    end_date = dates[list(dates.keys())[-1]] + dt.timedelta(days=1)

    plt.style.use('classic')    
    plt.rcParams.update({'font.size': 18})     
    fig, axs = plt.subplots(ncols=cols, nrows=2, figsize=((15*cols), 25))
    fig.patch.set_facecolor('white')

    for idx, key in enumerate(list(pred.keys())):
        x = idx // cols
        y = idx % cols

        final_idx = int(pred[key][0].get_columns()[-1].name)
        amount_days = min(amount_days, final_idx - 1)

        true_data = I_data.filter(pl.col('regions').is_in(regions))

        labels = ['True Data']
        first = True

        for i, pred_data in enumerate(pred[key]):
            labels.append(f"Predicted ({get_optimizer_labels(pred_optimizers[i])})")

            pred_data = pred[key][i].filter(pl.col('regions').is_in(regions))

            region_summed_true_data, region_summed_pred_data = get_summed_data(true_data, pred_data, regions, final_idx)

            no_legend = False

            for region in regions:
                if first:
                    x_true, y_true = zip(*region_summed_true_data[region].items())
                    x_true = [dates[x] for x in x_true[1:] if x in list(dates.keys())]
                    y_true = [y_true[idx] for idx in range(1, len(y_true)) if idx in list(dates.keys())]

                    if cols == 1:
                        axs[y].plot(x_true, y_true, 'bo-', label="_nolegend_" if no_legend else labels[i])
                    else:
                        axs[x,y].plot(x_true, y_true, 'bo-', label="_nolegend_" if no_legend else labels[i])

                x_pred, y_pred = zip(*region_summed_pred_data[region].items())
                x_pred = [dates[x] for x in x_pred]

                if cols == 1:
                    axs[y].plot(x_pred, y_pred, colors[i], label="_nolegend_" if no_legend else None)
                else:
                    axs[x,y].plot(x_pred, y_pred, colors[i], label="_nolegend_" if no_legend else None)

                no_legend = True

            first = False

            if cols == 1:
                axs[y].set_xlim(start_date, end_date)
                axs[y].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
                axs[y].xaxis.set_major_locator(mdates.DayLocator(interval=4))

                axs[y].set(xlabel='Day', ylabel='Fraction Infected (cumulative)',
                    title=f'({graph_names[idx]}) {key} days ahead')
                axs[y].grid()
                legend = axs[y].legend(labels, loc='upper left')
            else:
                axs[x,y].set_xlim(start_date, end_date)
                axs[x,y].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
                axs[x,y].xaxis.set_major_locator(mdates.DayLocator(interval=3))

                axs[x,y].set(xlabel='Day', ylabel='Fraction Infected (cumulative)',
                    title=f'({graph_names[idx]}) {key} days ahead')
                axs[x,y].grid(visible=True)
                legend = axs[x,y].legend(labels, loc='upper left')
            
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_facecolor('none')
    if fig_path is not None:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path)
        print(f'saving plotted graph to {fig_path}')
    plt.show()

def plot_evaluations(eval_dfs, eval_optimizers, fig_path=None, amount_days=25, evaluations=[Evaluation.MSE]):
    """
    Plot evaluation metrics over time for different optimizers.

    Parameters
    ----------
    eval_dfs : dict of DataFrame
        Dictionary of evaluation DataFrames.
    eval_optimizers : list of Optimizers
        The list of optimizers used.
    fig_path : str, optional
        Path to save the figure. Default is None.
    amount_days : int, optional
        Number of days to show. Default is 25.
    evaluations : list of Evaluation, optional
        List of evaluation metrics to plot. Default is [Evaluation.MSE].
    """
    colors = ['rs-', 'g^-', 'y2-', 'm8-', 'cx-', 'kd-', 'w_-']
    graph_names = list('abcdefghijklmnopqrstuvwxyz')

    labels = []

    cols = int(np.ceil(len(list(eval_dfs.keys()))/2))

    for evaluation in evaluations:
        fig, axs = plt.subplots(ncols=cols, nrows=2, figsize=((15*cols), 20))
        fig.patch.set_facecolor('white')

        if evaluation == Evaluation.sMAPE:
            eval = 'sMAPE'
        elif evaluation == Evaluation.MSE:
            eval = 'MSE'
        else:
            raise Exception("Invalid evaluation metric")

        for idx, pred_days in enumerate(list(eval_dfs.keys())):
            x = idx // cols
            y = idx % cols

            eval_df = eval_dfs[pred_days]
            
            for i in range(len(eval_df)):
                start_date = eval_df[i]["k"][0] - dt.timedelta(days=1)
                end_date = eval_df[i]["k"][-1] + dt.timedelta(days=1)

                labels.append(get_optimizer_labels(eval_optimizers[i]))

                if cols == 1:
                    axs[y].plot(eval_df[i]["k"], eval_df[i][eval], colors[i])
                else:
                    axs[x,y].plot(eval_df[i]["k"], eval_df[i][eval], colors[i])

            if cols == 1:
                axs[y].set_xlim(start_date, end_date)
                axs[y].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
                axs[y].xaxis.set_major_locator(mdates.DayLocator())

                axs[y].set(xlabel='Day', ylabel=get_evaluation_labels(evaluation),
                    title=f'({graph_names[idx]}) {eval} of {pred_days} days ahead')
                legend = axs[y].legend(labels, loc='upper left')
                axs[y].grid()
            else:
                axs[x,y].set_xlim(start_date, end_date)
                axs[x,y].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
                axs[x,y].xaxis.set_major_locator(mdates.DayLocator())

                axs[x,y].set(xlabel='Day', ylabel=get_evaluation_labels(evaluation),
                    title=f'({graph_names[idx]}) {eval} of {pred_days} days ahead')
                legend = axs[x,y].legend(labels, loc='upper left')
                axs[x,y].grid()

            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_facecolor('none')

        if fig_path is not None:
            splitted = fig_path.split('.png')
            eval_fig_path = f"{splitted[0]}_{eval}.png"
            os.makedirs(os.path.dirname(join(eval_fig_path)), exist_ok=True)
            fig.savefig(eval_fig_path)
            print(f'saving plotted graph to {eval_fig_path}')
        plt.show()