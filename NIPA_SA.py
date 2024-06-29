import polars as pl
import numpy as np
import time
from datetime import datetime as dt
from datetime import timedelta

from utils.Optimizers import Optimizers
from utils.io import get_train_data_path, save_results_data

import numpy as np
from sklearn.linear_model import Lasso
from scipy.optimize import dual_annealing
import utils.evaluation as evaluation

class NIPA:
    def __init__(self, data, regions, country, type, dates, parameter_optimizer=Optimizers.CV, random=False):
        """
        Initializes the NIPA model.

        Parameters
        ----------
        data : DataFrame
            The initial data.
        regions : list of str
            The region names.
        country : Country
            The country being modeled.
        type : Type
            The type of NIPA model.
        dates : dict
            A dictionary mapping indices to dates.
        parameter_optimizer : Optimizers, optional
            The optimizer used for parameter optimization. Default is Optimizers.CrossValidation.
        """
        
        self.save_path = "data/saves/"

        self.start_data = data
        self.regions = self.define_regions_i(regions)
        self.country = country
        self.type = type
        self.dates = dates
        self.parameter_optimizer = parameter_optimizer
        self.curing_probs = pl.DataFrame({})
        self.I = self.get_I()
        self.random = random

    def get_I(self):
        """
        Loads and processes the initial infection data.

        Returns
        -------
        DataFrame
            A DataFrame containing the fraction of infected individuals.
        """

        I = pl.read_csv(get_train_data_path(self.country) + "I.csv")
        # Change the column names to be the index of region i
        I = I.with_columns(pl.Series("regions", [str(i+1) for i in range(len(self.regions))]))
        return I.transpose(include_header=True, header_name="k", column_names="regions")

    def define_regions_i(self, regions):
        """
        Defines region indices.

        Parameters
        ----------
        regions : list of str
            The region names.

        Returns
        -------
        dict
            A dictionary mapping region indices to region names.
        """

        return {i+1: regions[i] for i in range(len(regions))}

    def train(self, train_days):
        """
        Trains the model using the provided training days.

        Parameters
        ----------
        train_days : list of datetime.date
            The training days.

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

        curing_probs = self.generate_curing_probs(50)

        for i in self.regions.keys():
            I_region = self.I[k_list[0]-1:k_list[-1], int(i)].to_list()

            B_region_list = []
            evaluation_region_list = []
            curing_prob_region_list = []
            r_parameters_list = []
            p_min_max_list = []

            ts = time.time()
            print(f"Training Region {self.regions[i]} ({i}/{len(self.regions.keys())})...")
            
            # print(f"Training for curing probability {curing_prob} at time {time.time()-ts:.2f}")
            I_df = self.I[k_list[0]-1:k_list[-1], 1:]

            B_region, evaluation_region, p_value, curing_prob, p_min_max = self.inference(I_region, I_df)

            B_region_list.append(B_region)
            evaluation_region_list.append(evaluation_region)
            curing_prob_region_list.append(curing_prob)
            r_parameters_list.append([p_value, p_min_max[0], p_min_max[1]])
            p_min_max_list.append(p_min_max)
            
            min_index = evaluation_region_list.index(min(evaluation_region_list))
            self.curing_probs = self.curing_probs.with_columns(pl.Series(str(i), [curing_prob_region_list[min_index]]))
            r_parameters = r_parameters.with_columns(pl.Series(self.regions[i], r_parameters_list[min_index]))

            B = B.with_columns(pl.Series(self.regions[i], B_region_list[min_index]))

            te = time.time()
            print(f"Region {self.regions[i]} took {te-ts:.4f} seconds to train")
            print(f'Curing probability: {curing_probs[min_index]}')
            print(f'P min: {p_min_max_list[min_index][0]} P max: {p_min_max_list[min_index][1]}')
            print(f'Regularization value: {r_parameters_list[min_index][0]}')
            print(f'evaluation: {evaluation_region_list[min_index]}')
            print(f'B: {B_region_list[min_index]}')
            print()
        
        save_results_data(B, self.country, self.type, self.parameter_optimizer, save_date, "B")
        save_results_data(r_parameters, self.country, self.type, self.parameter_optimizer, save_date, "parameters")
        save_results_data(self.curing_probs, self.country, self.type, self.parameter_optimizer, save_date, "curing_probs")
        return B

    def predict(self, B, pred_days, curing_probs):
        """
        Predicts future infection fractions using the trained model.

        Parameters
        ----------
        B : DataFrame
            The trained model parameters.
        pred_days : list of datetime.date
            The prediction days.

        Returns
        -------
        tuple
            A tuple containing DataFrames for predicted infection fractions (I_pred) and predicted recovered fractions (R_pred).
        """

        k_list = self.convert_dates_to_k(pred_days)
        R_pred = pl.DataFrame({       
            "regions": [self.regions[i] for i in self.regions.keys()]
        })
        I_pred = pl.DataFrame({
            "regions": [self.regions[i] for i in self.regions.keys()]
        })
        all_Ri = []

        for i in self.regions.keys():
            curing_prob_region = curing_probs.get_column(str(i)).to_list()[0]
            I_region = self.I.get_column(str(i))[:(k_list[-1])].to_list()

            Ri = self.calc_R(I_region, curing_prob_region)

            all_Ri.append(Ri)

        for k in range(1, len(pred_days)+1):
            Ii_pred = []
            Ri_pred = []
            for i in self.regions.keys():
                curing_prob_region = curing_probs.get_column(str(i)).to_list()[0]
                if k == 1:
                    Iik = self.I.get_column(str(i)).to_list()[k_list[0]-1]
                    Rik = Ri[-1]
                else:
                    Iik = I_pred.get_column(str(k_list[k-2])).to_list()[i-1]
                    Rik = R_pred.get_column(str(k_list[k-2])).to_list()[i-1]
                
                Iik_pred = (1-curing_prob_region)*Iik + (1 - Iik - Rik) * self.calc_interaction_sum(B, i, k_list[k-1]) 
                Rik_pred = Rik + (curing_prob_region * Iik)            
                Ii_pred.append(Iik_pred)
                Ri_pred.append(Rik_pred)
            
            I_pred = I_pred.with_columns(pl.Series(str(k_list[k-1]), Ii_pred))
            R_pred = R_pred.with_columns(pl.Series(str(k_list[k-1]), Ri_pred))

        return I_pred, R_pred

    def calc_interaction_sum(self, B, i, k):
        """
        Calculates the interaction sum for region i at time k.

        Parameters
        ----------
        B : DataFrame
            The trained model parameters.
        i : int
            The region index.
        k : int
            The time index.

        Returns
        -------
        float
            The interaction sum for the specified region and time.
        """

        sum = 0

        # Get the interaction sum for region i, j at time k
        # Where j is the region index, so for the index in the list we need to do j-1 (same with k)
        for j in self.regions.keys():
            if j == i:
                continue
            sum += B.get_column(self.regions[i]).to_list()[j-1] * self.I.get_column(str(j)).to_list()[k-1]

        return sum

    def convert_dates_to_k(self, days):
        """
        Converts dates to indices.

        Parameters
        ----------
        days : list of datetime.date
            The dates to convert.

        Returns
        -------
        list of int
            A list of indices corresponding to the dates.
        """

        k_list = []
        for i, k in enumerate(self.dates.keys()):
            date = self.dates[k]
            if date >= days[0] and date <= days[-1]:
                k_list.append(k)
        return k_list

    def calc_R(self, I_region, curing_prob):
        """
        Calculates the recovered individuals for a region.

        Parameters
        ----------
        I_region : list of float
            The infection fractions for the region.
        curing_prob : float
            The curing probability.

        Returns
        -------
        numpy.ndarray
            An array containing the recovered individuals for the region.
        """

        R_region = np.zeros_like(I_region)

        for i in range(len(I_region) -1):
            R_region[i+1] = R_region[i] + (curing_prob * I_region[i])

        return R_region

    def calc_S(self, R_region, I_region):
        """
        Calculates the susceptible individuals for a region.

        Parameters
        ----------
        R_region : numpy.ndarray
            The recovered individuals for the region.
        I_region : list of float
            The infection fractions for the region.

        Returns
        -------
        numpy.ndarray
            An array containing the susceptible individuals for the region.
        """

        return (1 - R_region - I_region)
    
    def generate_curing_probs(self, amount=50):
        """
        Generates a range of curing probabilities.

        Parameters
        ----------
        amount : int, optional
            The number of curing probabilities to generate. Default is 50.

        Returns
        -------
        numpy.ndarray
            An array of curing probabilities.
        """
        curing_probs = np.linspace(0.01, 1.0, amount)

        return curing_probs

    def get_region_V(self, I_region, curing_prob):
        """
        Calculates the change in infection fractions between each time step for a region.

        Parameters
        ----------
        I_region : list of float
            The infection fractions for the region.
        curing_prob : float
            The curing probability.

        Returns
        -------
        numpy.ndarray
            The change in infection fractions for the region.
        """

        Vi = []

        for k in range(1, len(I_region)):
            Vik = I_region[k] - ((1-curing_prob)*I_region[k-1])
            Vi.append(Vik)

        V_np = np.asarray(Vi)
        V_np = V_np.reshape(-1, 1)
        return V_np
    
    def get_region_F(self, S_region, I_df):
        """
        Calculates the maximum number of interactions between susceptible and infected individuals for a region.

        Parameters
        ----------
        S_region : numpy.ndarray
            The susceptible individuals for the region.
        I_df : DataFrame
            The DataFrame containing infection data between a certain time.

        Returns
        -------
        numpy.ndarray
            The maximum number of interactions between susceptible and infected individuals for the region.
        """

        F = []

        for k in range(len(S_region)-1):
            row = []
            for i, region in enumerate(self.regions):
                elem = S_region[k] * I_df[k, i]
                row.append(elem)
            F.append(row)

        F_np = np.asarray(F)
        return F_np
    
    def cost(self, parameters, I_region, I_df, all_p, all_MSE, random=False):
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

        if random:
            lasso = Lasso(alpha=p, max_iter=int(2e2), tol=1e-8, positive=True).fit(F_region, V_region)
        else:
            lasso = Lasso(alpha=p, max_iter=int(2e2), tol=1e-8, positive=True, random_state=42).fit(F_region, V_region)
        MSE = evaluation.MSE(V_region, lasso.predict(F_region))

        if np.sum(lasso.coef_) > 1.0:
            MSE = 1

        if MSE not in all_MSE:
            all_p.append(p)
            all_MSE.append(MSE)

        return MSE
    
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

        return p_min, p_max


    def calc_avg_E(self, all_MSE, until=100):
        """
        Calculates the average energy difference for the MSE values.

        Parameters
        ----------
        all_MSE : list of float
            List of MSE values.
        until : int, optional
            Number of values to consider for the average calculation. Default is 100.
        """
        sum = 0

        for idx, MSE in enumerate(all_MSE[:until]):
            if idx == 0:
                continue

            sum += all_MSE[idx-1] - MSE
        
        avg_E = sum / (len(all_MSE) - 1)

        with open('data/avg_E_all.txt', 'a+') as f:
            f.write(f"{avg_E}\n")


    def calc_delta_E(self, all_MSE, T0, until=1500):
        """
        Calculates the standard deviation of the energy difference for the MSE values.

        Parameters
        ----------
        all_MSE : list of float
            List of MSE values.
        T0 : float
            The initial temperature.
        until : int, optional
            Number of values to consider for the standard deviation calculation. Default is 1500.
        """
        sum_E = 0
        sum_E2 = 0

        first_part = (1/(until-1)) * np.sum(all_MSE[:until])
        second_part = (1/(until*(until-1))) * np.sum([MSE ** 2 for MSE in all_MSE[:until]])

        std_E = first_part - second_part

        with open('data/std_E.txt', 'a+') as f:
            f.write(f"{std_E}\n")

        k = -(T0 * np.log(0.8)) / std_E

        with open('data/K_DSA.txt', 'a+') as f:
            f.write(f"{k}\n")

        print(f"Calculated K: {k}")

    def inference(self, I_region, I_df):
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


        p_min, p_max = self.find_p_min_max(I_region, I_df)

        print(f"p_min: {p_min}, p_max: {p_max}")

        all_p = []
        all_MSE = []

        # BEST: 4.491 * 0.013417252146760345
        T0 = 0.04578861788617888
        temp = 15.326148949995753 * T0

        print(temp)

        annealing_args = {
            'bounds': [(p_min, p_max), (0.01, 1.0)],
            'maxiter': int(2e2),
            'visit': 2.62,
            'accept': -5,
            'no_local_search': False,
            'initial_temp': temp,
            'x0': [p_min, 0.5]
        }

        ret = dual_annealing(self.cost, args=(I_region, I_df, all_p, all_MSE), seed=42 if not self.random else None, **annealing_args)

        self.calc_avg_E(all_MSE)
        self.calc_delta_E(all_MSE, T0)

        best_p = ret.x[0]
        best_curing_prob = ret.x[1]

        R_region = self.calc_R(I_region, best_curing_prob)
        S_region = self.calc_S(R_region, I_region)
        
        V_region = self.get_region_V(I_region, best_curing_prob)
        F_region = self.get_region_F(S_region, I_df)

        lasso = Lasso(alpha=best_p, max_iter=int(2e2), tol=1e-8, positive=True, random_state=42).fit(F_region, V_region)
        region_B = lasso.coef_
        region_MSE = evaluation.MSE(V_region, lasso.predict(F_region)) 

        return region_B, region_MSE, best_p, best_curing_prob, [p_min, p_max]