import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from scipy.optimize import dual_annealing
import utils.evaluation as evaluation
import utils.visualisation as visualisation

import matplotlib.pyplot as plt

def cost(p, F_region, V_region, all_p, all_MSE, random):
    """
    Calculate the cost for a given regularization parameter using Lasso regression.

    Parameters
    ----------
    p : float
        The regularization parameter.
    F_region : numpy.ndarray
        The feature matrix for the region.
    V_region : numpy.ndarray
        The response variable (viral state) for the region.
    all_p : list of float
        List to store all regularization parameter values.
    all_MSE : list of float
        List to store all MSE values.
    random : bool
        Whether to use a random seed.

    Returns
    -------
    float
        The mean squared error (MSE) for the given regularization parameter.
    """
    lasso = Lasso(alpha=p[0], max_iter=int(2e2), tol=1e-8, positive=True, random_state=42 if not random else None).fit(F_region, V_region)
    MSE = evaluation.MSE(V_region, lasso.predict(F_region))

    if np.sum(lasso.coef_) > 1.0:
        MSE = 1

    #if MSE not in all_MSE:
    all_p.append(p[0])
    all_MSE.append(MSE)

    return MSE

def inference(F_region, V_region, max_iter=1000, tol=1e-8, random=False):
    """
    Perform inference on the COVID-19 data using dual annealing with Lasso regression.

    Parameters
    ----------
    F_region : numpy.ndarray
        The feature matrix for the region.
    V_region : numpy.ndarray
        The response variable (viral state) for the region.
    max_iter : int, optional
        The maximum number of iterations for dual annealing. Default is 1000.
    tol : float, optional
        The tolerance for the Lasso regression. Default is 1e-8.
    random : bool, optional
        Whether to use a random seed. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - region_B : numpy.ndarray
            The coefficients of the Lasso regression model.
        - region_MSE : float
            The mean squared error of the model predictions.
        - best_p : float
            The best alpha value found by dual annealing.
        - p_range : list of float
            The range of candidate alpha values [p_min, p_max].
    """
    F_transposed = np.transpose(F_region)
    Ft_V = np.dot(F_transposed, V_region)

    p_max = 2 * np.linalg.norm(Ft_V, np.inf)
    p_min = 0.0001 * p_max

    V_region = V_region.reshape((-1,))

    all_p = []
    all_MSE = []

    # Handle case where p_max is zero
    if p_max == 0:
        region_B = np.zeros(F_region.shape[1])
        region_MSE = mean_squared_error(V_region, np.dot(F_region, region_B))
        p_value = 0.0
        return region_B, region_MSE, p_value, [p_min, p_max]
    
    annealing_args = {
        'bounds': [(p_min, p_max)],
        'maxiter': max_iter,
        'visit': 2.62,
        'accept': -5,
        'no_local_search': False,
        'initial_temp': 0.3693670547881299,
        'x0': [p_min]
    }

    ret = dual_annealing(cost, args=(F_region, V_region, all_p, all_MSE, random), bounds=[(p_min, p_max)], seed=42 if not random else None, **annealing_args)

    best_p = ret.x[0]
    lasso = Lasso(alpha=best_p, max_iter=int(2e2), tol=1e-8, positive=True, random_state=42 if not random else None).fit(F_region, V_region)
    region_B = lasso.coef_
    region_MSE = evaluation.MSE(V_region, lasso.predict(F_region)) 

    return region_B, region_MSE, best_p, [p_min, p_max]