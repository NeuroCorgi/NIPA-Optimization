import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from scipy.optimize import dual_annealing
from sklearn.model_selection import train_test_split
import utils.evaluation as evaluation
import utils.visualisation as visualisation

import matplotlib.pyplot as plt

def cost(p, F_region, V_region, all_p, all_MSE):
    lasso = Lasso(alpha=p[0], max_iter=int(2e2), tol=1e-8, positive=True, random_state=22).fit(F_region, V_region)
    MSE = evaluation.sMAPE(V_region, lasso.predict(F_region))

    if np.sum(lasso.coef_) > 1.0:
        MSE = 1000

    #if MSE not in all_MSE:
    all_p.append(p[0])
    all_MSE.append(MSE)

    return MSE

def inference(F_region, V_region, max_iter=1000, tol=1e-8):
    F_transposed = np.transpose(F_region)
    Ft_V = np.dot(F_transposed, V_region)

    p_max = 2 * np.linalg.norm(Ft_V, np.inf)
    p_min = 0.0001 * p_max

    V_region = V_region.reshape((-1,))

    all_p = []
    all_MSE = []

    if p_max == 0:
        region_B = np.zeros(F_region.shape[1])
        region_MSE = mean_squared_error(V_region, np.dot(F_region, region_B))
        p_value = 0.0
        return region_B, region_MSE, p_value, [p_min, p_max]

    ret = dual_annealing(cost, args=(F_region, V_region, all_p, all_MSE), bounds=[(p_min, p_max)], 
                         maxiter=max_iter, no_local_search=True, initial_temp=10.0, seed=22, x0=[p_min])

    best_p = ret.x[0]
    lasso = Lasso(alpha=best_p, max_iter=int(2e2), tol=1e-8, positive=True, random_state=22).fit(F_region, V_region)
    region_B = lasso.coef_
    region_MSE = evaluation.sMAPE(V_region, lasso.predict(F_region)) 

    # visualisation.visualise_simulated_annealing(all_p, all_MSE, max_iter, fig_path=None)

    return region_B, region_MSE, best_p, [p_min, p_max]