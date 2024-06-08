from enum import Enum
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from matplotlib import pyplot

class CoolingSchedule(Enum):
    LINEAR = 1
    GEOMETRIC = 2
    LOGARITHMIC = 3

def sMAPE(pred, true):
    return 100/len(true) * np.sum(2 * np.abs(pred - true) / (np.abs(true) + np.abs(pred)))

def generate_candidate(prev_candidate, p_min, p_max, all_p, T):
    # Standard deviation dynamically adjusts with temperature
    # Lower T should correspond with more refined, smaller changes
    std = T * (p_max - p_min) / 5  # Reduced the influence and scaled by T

    candidate = np.random.normal(prev_candidate, std)

    while candidate < p_min or candidate > p_max or candidate in all_p:
        candidate = np.random.normal(prev_candidate, std)
    
    # print(f"candidate: {candidate}")

    return candidate

def is_accepted(E_new, E_old, T):
    delta_E = E_new - E_old

    # print(f"E_new: {E_new}, E_old: {E_old}")
    # print(f"delta_E: {delta_E}, acceptance_probability: {acceptance_probability}")

    if delta_E < 0:
        return True
    else:
        acceptance_probability = np.exp(-delta_E / T)
        return np.random.rand() < acceptance_probability
    
def generate_initial_values(p_min, p_max, F_region_train, V_region_train, F_region_test, V_region_test):
    avg = (p_min + p_max) / 2
    std = (p_max - p_min) / 6
    p0 = np.random.normal(avg, std)
    p_values = [p0]

    lasso = Lasso(alpha=p0, max_iter=100000).fit(F_region_train, V_region_train)
    B = lasso.coef_
    yk = lasso.predict(F_region_test)

    sMape_values = [sMAPE(yk, V_region_test)]
    MAPE = mean_absolute_percentage_error(V_region_test, yk) 
    MAPE_values = [MAPE]
    MSE_values = [mean_squared_error(V_region_test, yk)]
    B_values = [B]

    return p_values, sMape_values, B_values
    

def inference(F_region, V_region, iterations=100, T_init=100, T_final=0.000000001, cooling_schedule=CoolingSchedule.GEOMETRIC, alpha=0.8, amount_CPUs=-1):        
    F_transposed = np.transpose(F_region)
    Ft_V = np.dot(F_transposed, V_region)

    p_max = 2 * np.linalg.norm(Ft_V, np.inf)
    p_min = 0.0001 * p_max
    V_region = V_region.reshape((-1,))

    F_region_train, F_region_test, V_region_train, V_region_test = train_test_split(F_region, V_region, test_size=0.3)

    if p_max == 0:
        region_B = np.zeros(F_region.shape[1])
        region_MSE = mean_squared_error(V_region, np.dot(F_region, region_B))
        p_value = 0.0
        return region_B, region_MSE, p_value, [p_min, p_max]

    p_values, sMAPE_values, B_values = generate_initial_values(p_min, p_max, F_region_train, V_region_train, F_region_test, V_region_test)
    best_p = [p_values[-1]]
    best_sMAPE = [sMAPE_values[-1]]
    # best_MAPE = [MAPE_values[-1]]
    # best_MSE = [MSE_values[-1]]
    best_B = [B_values[-1]]
    T_init = 2.0
    T = T_init
    # print(T_init)

    for k in range(iterations):
        # print(f"> {k}")
        p_old = p_values[-1]
        candidates = generate_candidate(p_old, p_min, p_max, best_p, T)

        # print(f"p_old: {p_old} p_new: {p_new}")
        # print(f"Temperature: {T}")
        candidate_MAPE = []
        candidate_MSE = []
        candidate_sMAPE = []
        candidate_B = []
        
        for candidate in candidates:
            lasso = Lasso(alpha=candidate, max_iter=100000, positive=True).fit(F_region_train, V_region_train)
            B = lasso.coef_
            # print(f"B: {B}")

            if np.sum(B) > 1.0:
                print("SUM GREATER THAN 1.0!!\n")
                continue

            yk = lasso.predict(F_region_test)
            MAPE = mean_absolute_percentage_error(V_region_test, yk) 
            sMape = sMAPE(yk, V_region_test)
            MSE = mean_squared_error(V_region_test, yk)
            # print(f"MAPE: {MAPE}, MSE: {MSE}")

            score = lasso.score(F_region_test, V_region_test)

            candidate_MAPE.append(MAPE)
            candidate_MSE.append(MSE)
            candidate_sMAPE.append(sMape)
            candidate_B.append(B)

        best_MAPE_idx = candidate_MAPE.index(min(candidate_MAPE))
        best_sMAPE_idx = candidate_sMAPE.index(min(candidate_sMAPE))
        best_MSE_idx = candidate_MSE.index(min(candidate_MSE))

        best_p_candidate = candidates[best_MSE_idx]
        best_sMAPE_candidate = candidate_sMAPE[best_sMAPE_idx]
        best_MAPE_candidate = candidate_MAPE[best_MSE_idx]
        best_MSE_candidate = candidate_MSE[best_MSE_idx]

        if is_accepted(best_sMAPE_candidate, best_sMAPE[-1], T):
            # print(f"Accepted: {score} {MSE_values[-1]}")
            # print(MSE_values)
            best_p.append(best_p_candidate)
            best_sMAPE.append(best_sMAPE_candidate)
            # best_MSE.append(best_MAPE_candidate)
            # best_MSE.append(best_MSE_candidate)
            best_B.append(B)

        if T < T_final:
            # print("T lower than T_final")
            break

        if cooling_schedule == CoolingSchedule.LINEAR:
            T = T_init * (1 - k*alpha)
        elif cooling_schedule == CoolingSchedule.GEOMETRIC:
            T = alpha * T
        elif cooling_schedule == CoolingSchedule.LOGARITHMIC:
            T = T / np.log(1+k)

    p_value = best_p[-1]
    print(best_p)

    lasso = Lasso(alpha=p_value, max_iter=100000, positive=True).fit(F_region, V_region)
    B = lasso.coef_
    yk = lasso.predict(F_region_test)
    MAPE = mean_absolute_percentage_error(V_region_test, yk) 
    MSE = mean_squared_error(V_region_test, yk)

    # line plot of best scores
    # print(f"Length of MAPE: {len(MAPE_values)}")
    # pyplot.plot(all_MAPE, '.-')
    # pyplot.xlabel('Improvement Number')
    # pyplot.ylabel('Evaluation f(x)')
    # pyplot.show()

    return B, MSE, p_value, [p_min, p_max]