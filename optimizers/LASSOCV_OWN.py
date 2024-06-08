import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def inference(F_region, V_region, k_fold=3, test_size=0.3):
    F_transposed = np.transpose(F_region)
    Ft_V = np.dot(F_transposed, V_region)

    # print(F_region)
    # print(V_region)
    # print(Ft_V)

    p_max = 2 * np.linalg.norm(Ft_V, np.inf)
    p_min = 0.0001 * p_max

    V_region = V_region.reshape((-1,))

    if p_max == 0:
        B_region = np.zeros(F_region.shape[1])
        region_MSE = mean_squared_error(V_region, np.dot(F_region, B_region))
        p_value = 0.0
        return B_region, region_MSE, p_value, [p_min, p_max]

    p_candidates = np.logspace(np.log10(p_min), np.log10(p_max), num=100)
    
    B_region = np.ones(F_region.shape[1])
    sum = np.sum(B_region)

    F_region_train, F_region_test, V_region_train, V_region_test = train_test_split(F_region, V_region, test_size=test_size)

    kf = KFold(n_splits=k_fold)

    while True:
        MSE_list = [100000 for i in range(kf.get_n_splits())]
        p_list = [-1 for i in range(kf.get_n_splits())]
        B_list = [-1 for i in range(kf.get_n_splits())]
        folds = kf.split(F_region_train, V_region_train)

        for i, (train_index, test_index) in enumerate(folds):
            F_region_train_fold = [F_region_train[idx] for idx in train_index]
            F_region_test_fold = [F_region_train[idx] for idx in test_index]
            V_region_train_fold = [V_region_train[idx] for idx in train_index]
            V_region_test_fold = [V_region_train[idx] for idx in test_index]

            for p in p_candidates:
                lasso = Lasso(alpha=p, max_iter=int(200), tol=1e-8, positive=True, random_state=22)
                lasso.fit(F_region_train_fold, V_region_train_fold)
                B_region = lasso.coef_

                if np.sum(B_region) > 1.0:
                    continue

                MSE_region = mean_squared_error(V_region_test_fold, lasso.predict(F_region_test_fold))

                if MSE_region < MSE_list[i]:
                    MSE_list[i] = MSE_region
                    p_list[i] = p
                    B_list[i] = B_region
                
        idx_min_MSE = MSE_list.index(min(MSE_list))
        B_region = B_list[idx_min_MSE]
        region_MSE = MSE_list[idx_min_MSE]
        p_opt = p_list[idx_min_MSE]

        lasso = Lasso(alpha=p_opt, max_iter=int(2e2), tol=1e-8, positive=True, random_state=22)
        lasso.fit(F_region, V_region)
        B_region = lasso.coef_

        sum = np.sum(B_region)
        
        if sum > 1.0:
            print(f"Sum is greater than 1 ({sum})!")
            p_candidates = np.setdiff1d(p_candidates, np.array([p_opt]))
        else:
            break

        if len(p_candidates) == 0:
            print("All candidates are not valid!")
            region_B = np.zeros(F_region.shape[1])
            break

    return B_region, region_MSE, p_opt, [p_min, p_max]