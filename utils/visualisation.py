import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl
import os

def visualise_simulated_annealing(all_p, all_MSE, fig_path=None):
    combined_p_MSE = zip(all_p, all_MSE)
    sorted_list = sorted(combined_p_MSE, key=lambda x: x[0])

    # Check where the first index is where MSE does not change (all values to 0)
    first_index_double = [idx for idx, item in enumerate(sorted_list) if item[1] == sorted_list[idx-1][1]][0]  

    # Show all p_values
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
    
    # P over time
    axs[0,0].plot(np.arange(all_p), all_p, 'bo')
    axs[0,0].set(xlabel='Time', ylabel='regularization value',
        title='Regularization parameter over time')
    axs[0,0].set_yscale('log')
    axs[0,0].grid()

    # MSE over time
    axs[0,1].plot(np.arange(len(all_MSE)), all_MSE, 'bo')
    axs[0,1].set(xlabel='Time', ylabel='MSE',
        title='MSE over time')
    axs[0,1].grid()

    p_sorted, MSE_sorted = zip(*sorted_list)
    # Regularization parameter and MSE
    axs[1,0].plot(p_sorted[:first_index_double], MSE_sorted[:first_index_double], 'bo')
    axs[1,0].set(xlabel='P value', ylabel='MSE',
        title='Regularization parameter and MSE')
    axs[1,0].grid()

    if fig_path != None:
        fig.savefig(fig_path)
    plt.show()


def heatmap_B(B):
    regions = B[:, 0].to_list()
    B = B.transpose(include_header=True, header_name="regions", column_names="regions")[:, 1:].to_numpy()

    for B_row in B:
        for i, val in enumerate(B_row):
            B_row[i] = round(val, 6)
            if val == 0.0:
                B_row[i] = int(0)

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

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()

def plot_predictions(I_data, pred, dates, regions, fig_path=None, amount_days=25, amount_regions=None):
    if amount_regions is not None:
        regions = regions[:(amount_regions)]

    fig, ax = plt.subplots(figsize=(10, 6))

    final_idx = int(pred.get_columns()[-1].name)

    if ((final_idx-1) - amount_days) < 0:
        amount_days = (final_idx-1)

    true_data = I_data.filter(pl.col('i').is_in(regions))
    pred_data = pred.filter(pl.col('regions').is_in(regions))

    region_summed_true_data = {}
    region_summed_pred_data = {}

    for region in regions:
        summed_true_data = {}
        summed_pred_data = {}
        for i in range(final_idx):
            if i == 0:
                summed_true_data[i] = 0

            k = i+1

            summed_true_data[k] = summed_true_data[i] + true_data.filter(pl.col('i') == region).get_column(str(k)).to_list()[0]

            if str(k) == pred_data.columns[1]:
                summed_pred_data[k] = summed_true_data[i] + pred_data.filter(pl.col('regions') == region).get_column(str(k)).to_list()[0]
            elif str(k) in pred_data.columns:
                summed_pred_data[k] = summed_pred_data[i] + pred_data.filter(pl.col('regions') == region).get_column(str(k)).to_list()[0]

        region_summed_true_data[region] = summed_true_data
        region_summed_pred_data[region] = summed_pred_data

    # region_summed_true_data = {k: region_summed_true_data[k][final_idx-1-amount_days:] for k in list(region_summed_true_data)}
    # region_summed_pred_data = {k: region_summed_pred_data[k][final_idx-1-amount_days:] for k in list(region_summed_pred_data)}

    for region in regions:
        x_true, y_true = zip(*region_summed_true_data[region].items())
        x_true = [dates[x] for x in x_true[1:]]
        ax.plot(x_true, y_true[1:], 'bo-')

        x_pred, y_pred = zip(*region_summed_pred_data[region].items())
        x_pred = [dates[x] for x in x_pred]
        ax.plot(x_pred, y_pred, 'rs-')

    ax.set(xlabel='Timestep', ylabel='Infected',
        title='Predicted vs True')
    ax.grid()

    ax.legend(['True Data', 'Predicted (NIPA)'])
    plt.show()

    if fig_path is not None:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path)
        print(f'saving plotted graph to {fig_path}')