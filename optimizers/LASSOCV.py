import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def inference(F_region, V_region, k_fold=3, amount_CPUs=-1):
    """
    Perform inference on the COVID-19 data using cross-validation with the sklearn LassoCV.

    Parameters
    ----------
    F_region : numpy.ndarray
        The feature matrix for the region.
    V_region : numpy.ndarray
        The response variable (viral state) for the region.
    k_fold : int, optional
        The number of folds for cross-validation. Default is 3.
    amount_CPUs : int, optional
        The number of CPUs to use for parallel processing. Default is -1 (use all available CPUs).

    Returns
    -------
    tuple
        A tuple containing:
        - region_B : numpy.ndarray
            The coefficients of the Lasso regression model.
        - region_MSE : float
            The mean squared error of the model predictions.
        - p_value : float
            The selected alpha value for the Lasso model.
        - p_range : list of float
            The range of candidate alpha values [p_min, p_max].
    """
    
    F_transposed = np.transpose(F_region)
    Ft_V = np.dot(F_transposed, V_region)

    # print(F_region)
    # print(V_region)
    # print(Ft_V)

    p_max = 2 * np.linalg.norm(Ft_V, np.inf)
    p_min = 0.0001 * p_max

    V_region = V_region.reshape((-1,))

    if p_max == 0:
        region_B = np.zeros(F_region.shape[1])
        region_MSE = mean_squared_error(V_region, np.dot(F_region, region_B))
        p_value = 0.0
        return region_B, region_MSE, p_value, [p_min, p_max]

    p_candidates = np.logspace(np.log10(p_min), np.log10(p_max), num=100)
    region_B = np.ones(F_region.shape[1])
    sum = np.sum(region_B)
    first_iter = True

    while sum > 1.0:
        if sum != F_region.shape[1] and first_iter:
            first_iter = False
            print(f"Sum is greater than 1 ({sum})!")
        lasso = LassoCV(alphas=p_candidates, max_iter=int(2e2), tol=1e-8,
                    copy_X=True, cv=k_fold, n_jobs=amount_CPUs, positive=True, random_state=22).fit(F_region, V_region)

        region_B = lasso.coef_

        sum = np.sum(region_B)
        p_value = lasso.alpha_
        region_MSE = mean_squared_error(V_region, lasso.predict(F_region))

        p_candidates = np.setdiff1d(p_candidates, np.array([p_value]))

        if len(p_candidates) == 0:
            print("All candidates are not valid!")
            region_B = np.zeros(F_region.shape[1])
            break

    return region_B, region_MSE, p_value, [p_min, p_max]