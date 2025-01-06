import polars as pl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import time
from datetime import datetime as dt
from datetime import timedelta
from os.path import join

from utils.Optimizers import Optimizers
from utils.io import get_train_data_path, save_results_data

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from scipy.optimize import dual_annealing
import utils.evaluation as evaluation

from NIPA import NIPA

class NIPA(NIPA):

    def train(self, train_days, I_df_real):
        """
        Trains the model using the provided training days.

        Parameters
        ----------
        train_days : list of datetime.date
            The training days.
        I_df_real : Dataframe
            The real infected percentage of the population.
        
        Returns
        -------
        DataFrame
            A DataFrame (B) containing the infection probabilities.
        """

        print(f"Training from {train_days[0]} to {train_days[-1]}")
        save_date = train_days[-1]
        B = pl.DataFrame({
            "regions": [self.regions[i] for i in self.regions.keys()]
        })
        r_parameters = pl.DataFrame({
            "parameter": ["regularization", "min", "max"]
        })
        k_list = self.convert_dates_to_k(train_days)
        
        real_I = I_df_real.with_columns(pl.Series("regions", [str(i+1) for i in range(len(self.regions))]))
        real_I = I_df_real.transpose(include_header=True, header_name="k", column_names="regions")

        for i in self.regions.keys():
            I_region = self.I[k_list[0]-1:k_list[-1], int(i)].to_list()
            real_I_region = real_I[k_list[0]-1:k_list[-1], int(i)].to_list()

            ts = time.time()
            print(f"Training Region {self.regions[i]} ({i}/{len(self.regions.keys())})...")
            
            # print(f"Training for curing probability {curing_prob} at time {time.time()-ts:.2f}")
            I_df = self.I[k_list[0]-1:k_list[-1], 1:]
            real_I_df = real_I[k_list[0]-1:k_list[-1], 1:]

            B_region, evaluation_region, p_value, curing_prob, p_min_max = self.inference(I_region, I_df, real_I_region, real_I_df)
            
            self.curing_probs = self.curing_probs.with_columns(pl.Series(str(i), [curing_prob]))
            r_parameters = r_parameters.with_columns(pl.Series(self.regions[i], [p_value, p_min_max[0], p_min_max[1]]))

            B = B.with_columns(pl.Series(self.regions[i], B_region))

            te = time.time()
            print(f"Region {self.regions[i]} took {te-ts:.4f} seconds to train \n")
            print(f"The estimate values for region {self.regions[i]}:")
            print(f'\t- Curing probability: {curing_prob}')
            print(f'\t- Regularization value: {p_value}  (min: {p_min_max[0]}; max: {p_min_max[1]})')
            print(f'\t- Evaluation (MSE): {evaluation_region}')
            print(f'\t- Infection probability matrix (B): {B_region}')
            print()
        
        save_results_data(B, self.country, self.type, self.parameter_optimizer, save_date, "B")
        save_results_data(r_parameters, self.country, self.type, self.parameter_optimizer, save_date, "parameters")
        save_results_data(self.curing_probs, self.country, self.type, self.parameter_optimizer, save_date, "curing_probs")
        return B
    
    def cost(self, parameters, I_region, I_df, low_p_min, high_p_max, all_p, all_MSE, all_curing_probs, real_I_region, real_I_df, random=False):
        """
        Calculates the cost for a given set of parameters using Lasso regression.

        Parameters
        ----------
        parameters : list of float
            The parameters [regularization parameter, curing probability].
        I_region : list of float
            The infection fractions for the region.
        I_df : DataFrame
            The DataFrame containing infection data.
        all_p : list of float
            List to store all regularization parameter values.
        all_MSE : list of float
            List to store all MSE values.
        random : bool, optional
            Whether to use a random seed. Default is False.

        Returns
        -------
        float
            The mean squared error (MSE) for the given parameters.
        """        
        p = parameters[0]
        curing_prob = parameters[1]

        R_region = self.calc_R(I_region, curing_prob)
        S_region = self.calc_S(R_region, I_region)
        V_region = self.get_region_V(I_region, curing_prob)
        F_region = self.get_region_F(S_region, I_df)
        
        F_transposed = np.transpose(F_region)
        Ft_V = np.dot(F_transposed, V_region)
        
        p_max = 2 * np.linalg.norm(Ft_V, np.inf)
        p_min = 0.0001 * p_max
        
        if p_min > p or p_max < p:
            perc = (p-low_p_min)/(high_p_max-low_p_min)
            
            range_p = p_max-p_min
            new_p = range_p * perc
            p = p_min + new_p
        
        lasso = Lasso(alpha=p, max_iter=int(2e2), tol=1e-8, positive=True, random_state=42 if not random else None).fit(F_region, V_region)
        
        
######### PREDICTING ACCORDING TO NIPA
        real_I = [[I] for I in real_I_region]
        I_pred = []
        for k, value in enumerate(I_region):
            interaction_sum = np.sum([infection_prob*self.I.get_column(self.I.columns[idx+1]).to_list()[k-1] for idx, infection_prob in enumerate(lasso.coef_)])
            # B.get_column(self.regions[i]).to_list()[j-1] * self.I.get_column(str(j)).to_list()[k-1]
            I_pred.append([(1-curing_prob)*I_region[k-1] + (1 - I_region[k-1] - R_region[k-1]) * interaction_sum])
        I_pred = np.array(I_pred)
        I_region = np.array([[I] for I in I_region])
        
        MSE = evaluation.MSE(real_I_region, I_pred)
        sMAPE = evaluation.sMAPE(real_I_region, I_pred)
        
        sum_coefs = np.sum(lasso.coef_)
        
        if sum_coefs >= 1.0:
            return 25
        
        all_p.append(p)
        all_MSE.append(MSE)
        all_curing_probs.append(curing_prob)

        return MSE
    
    def calc_initial_temp(self, q_a, initial_acceptance_prob, annealing_args, I_df, I_region, real_I_region, real_I_df, p_min, p_max, until=250):
        annealing_args['maxiter'] = until
        MSE_until = []
        p_until = []
        curing_probs_until = []
        found_zero = []
        
        ret = dual_annealing(self.cost, args=(I_region, I_df, p_min, p_max, p_until, MSE_until, curing_probs_until, real_I_region, real_I_df), seed=42 if not self.random else None, **annealing_args)
        
        MSE_change = [np.absolute(MSE_until[idx+1] - MSE) for idx, MSE in enumerate(MSE_until) if (idx < len(MSE_until)-1)]
        
        np_avg_E = np.average(MSE_change) if len(MSE_change) > 0 else 0
        avg_E = self.calc_avg_E(MSE_until) if len(MSE_change) > 0 else 0
        
        max_E_change = max(MSE_change) if len(MSE_change) > 0 else np.random.uniform(0.75, 2)

        v = 1 - (q_a)
        prob_chance = initial_acceptance_prob ** v

        v *= avg_E
        prob_chance -= 1
        T0 = (-v)/prob_chance
        
        print(f"Highest change: {max_E_change}; Average change: {avg_E}; T0: {T0}")
        
        return T0, MSE_until, max_E_change
    
    def find_p_min_max(self, I_region, I_df):
        """
        Finds the minimum and maximum regularization parameters.

        Parameters
        ----------
        I_region : list of float
            The infection fractions for the region.
        I_df : DataFrame
            The DataFrame containing infection data.

        Returns
        -------
        tuple
            A tuple containing the minimum (p_min) and maximum (p_max) regularization parameters.
        """
        all_curing_probs = np.arange(0.01, 1.01, 0.01)

        all_p = []

        for curing_prob in all_curing_probs:
            V_region = self.get_region_V(I_region, curing_prob)
            F_region = self.get_region_F(self.calc_S(self.calc_R(I_region, curing_prob), I_region), I_df)
            F_transposed = np.transpose(F_region)
            Ft_V = np.dot(F_transposed, V_region)

            p_max = 2 * np.linalg.norm(Ft_V, np.inf)
            p_min = 0.0001 * p_max

            all_p.append(p_min)
            all_p.append(p_max)

        p_min = min(all_p)
        p_max = max(all_p)
        curing_prob_min = all_curing_probs[all_p.index(p_min)]
        
        print(f"P_min is {p_min} with curing probability {curing_prob_min}")

        return p_min, p_max, curing_prob_min


    def calc_avg_E(self, all_MSE):
        """
        Calculates the average energy difference for the MSE values.

        Parameters
        ----------
        all_MSE : list of float
            List of MSE values.
        until : int, optional
            Number of values to consider for the average calculation. Default is 100.
        """
        if len(all_MSE) == 0:
            return 0
        
        sum = 0        

        for idx, MSE in enumerate(all_MSE):
            if idx <= 0 or idx == len(all_MSE)-1:
                continue
            
            change = all_MSE[idx+1] - MSE
            sum += np.absolute(change)
        
        avg_E = sum / (len(all_MSE))
        
        return avg_E


    def calc_std_E(self, MSE_until):
        """
        Calculates the standard deviation of the energy difference for the MSE values.

        Parameters
        ----------
        MSE_until : list of float
            List of MSE values.
        T0 : float
            The initial temperature.Returns
        -------
        float
            The estimated standard deviation of the Energy (E) for the given parameters.
        """
        if len(MSE_until) == 0 or len(MSE_until) == 1:
            return 0
        
        sum_E = 0
        sum_E2 = 0
        
        MSE_length = len(MSE_until)

        first_part = (1/(MSE_length-1)) * np.sum(MSE_until)
        second_part = (1/(MSE_length*(MSE_length-1))) * np.sum([MSE ** 2 for MSE in MSE_until])

        std_E = first_part - second_part

        return std_E
    
    def calc_k(self, T0, initial_acceptance_prob, MSE_until):
        """
        Calculates the estimation of the optimization problem for the initial temperature.

        Parameters
        ----------
        MSE_until : list of float
            List of MSE values.
        T0 : float
            The initial temperature.Returns
        -------
        float
            The estimated standard deviation of the Energy (E) for the given parameters.
        """
        std_E = self.calc_std_E(MSE_until)
        avg_E = self.calc_avg_E(MSE_until)
        
        max_E = max(std_E, avg_E)
        
        k = -(T0 * np.log(initial_acceptance_prob)) / max_E
        
        return k
        
    def inference(self, I_region, I_df, real_I_region, real_I_df):
        """
        Performs inference to estimate model parameters according to the regularization optimization.

        Parameters
        ----------
        viral_state : list of numpy.ndarray
            The viral state [S_region, I_region, R_region].
        I_df : DataFrame
            The DataFrame containing infection data.
        curing_prob : float
            The curing probability.

        Returns
        -------
        tuple
            A tuple containing the coefficients of the Lasso regression model (region_B),
            the mean squared error of the model predictions (region_evaluation), the selected alpha value
            for the Lasso model (p_value), and the range of candidate alpha values (p_min, p_max).
        """
        p_min, p_max, curing_prob_min = self.find_p_min_max(I_region, I_df)
        
        # Start with random values for the regularization parameter and the curing probability
        start_p = np.random.uniform(p_min, 0.0005 * p_max)
        start_curing_prob = np.random.uniform(0.25, 0.75)
        
        annealing_args = {
            'bounds': [(p_min, p_max), (0.01, 0.99)],
            'maxiter': int(1e3),
            'visit': 2.62,
            'accept': -5,
            'no_local_search': False,
            'initial_temp': 1000,
            'x0': [start_p, start_curing_prob]
        }
        
        print(f"Start position: ({start_p}, {start_curing_prob})")

        all_p = []
        all_MSE = []
        all_curing_probs = []

        q_a = -5
        initial_acceptance_prob = 0.8
        
        # Calculation of initial temperature works only with sMAPE evaluation as MSE evaluation produces too small values...
        T_qa, MSE_until, max_E_change = self.calc_initial_temp(q_a, initial_acceptance_prob, annealing_args, I_df, I_region, real_I_region, real_I_df, p_min, p_max, until=500)
        k = self.calc_k(T_qa, initial_acceptance_prob, MSE_until)
        print(f"k: {k}; T_qa: {T_qa}; k*T_qa: {k*T_qa}")
        MSE_until = [MSE for MSE in MSE_until if MSE < 1.0]
        
        # High temperature (100) is needed when MSE evaluation is used to make sure it does not 'freeze' at a local minimum
        # Otherwise use k * T_qa
        temp = 100
        print(f"Initial temperature: {temp}")
        
        # Changing annealing arguments to the correct version
        annealing_args['initial_temp'] = temp
        annealing_args['maxiter'] = int(1e3)
        
        ret = dual_annealing(self.cost, args=(I_region, I_df, p_min, p_max, all_p, all_MSE, all_curing_probs, real_I_region, real_I_df), seed=42 if not self.random else None, **annealing_args)
        
        if len(all_p) == 0:
            print("No suitable values for p and delta!")
            all_p.append(p_min)
            all_curing_probs.append(start_curing_prob)
        
        print(f"Information:")
        print(f"\tCause of termination: {ret.message[0]}")
        print(f"\tRegularization parameter went from {min(all_p)} to {max(all_p)}")
        print(f"\tCuring probability went from {min(all_curing_probs)} to {max(all_curing_probs)}")
        
        best_curing_prob = ret.x[1]
        if best_curing_prob in all_curing_probs:
            best_p = all_p[all_curing_probs.index(best_curing_prob)]
        else:
            print("Curing prob not in array")
            best_p = ret.x[0]
            
        if len(all_MSE) > 0:
            best_eval = min(all_MSE)
        else:
            best_eval = -1        

        R_region = self.calc_R(I_region, best_curing_prob)
        S_region = self.calc_S(R_region, I_region)
        
        V_region = self.get_region_V(I_region, best_curing_prob)
        F_region = self.get_region_F(S_region, I_df)

        lasso = Lasso(alpha=best_p, max_iter=int(2e2), tol=1e-8, positive=True, random_state=42 if not self.random else None).fit(F_region, V_region)
        region_B = lasso.coef_
        region_MSE = evaluation.MSE(V_region, lasso.predict(F_region)) 

        return region_B, region_MSE, best_p, best_curing_prob, [p_min, p_max]
