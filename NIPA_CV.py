import polars as pl
import numpy as np
import time

from utils.Optimizers import Optimizers
from optimizers import LASSOCV
from utils.io import get_train_data_path, save_results_data

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
            [NOT USED in Cross-Validation] The real infected percentage of the population.

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

        # Train for each region
        for i in self.regions.keys():
            I_region = self.I[k_list[0]-1:k_list[-1], int(i)].to_list()

            B_region_list = []
            evaluation_region_list = []
            curing_prob_region_list = []
            r_parameters_list = []
            p_min_max_list = []

            ts = time.time()
            print(f"Training Region {self.regions[i]} ({i}/{len(self.regions.keys())})...")
            for curing_prob in curing_probs: 
                R_region = self.calc_R(I_region, curing_prob)
                S_region = self.calc_S(R_region, I_region)
                        
                anomaly_R = [R for R in R_region if R > 1]
                if len(anomaly_R) > 0:
                    print(f"Anomaly R values: {anomaly_R}")
                    continue
                
                viral_state_region = [S_region, I_region, R_region]
                I_df = self.I[k_list[0]-1:k_list[-1], 1:]

                B_region, evaluation_region, p_value, p_min_max = self.inference(viral_state_region, I_df, curing_prob)

                B_region_list.append(B_region)
                evaluation_region_list.append(evaluation_region)
                curing_prob_region_list.append(curing_prob)
                r_parameters_list.append([p_value, p_min_max[0], p_min_max[1]])
                p_min_max_list.append(p_min_max)
            
            # Get the index of the best evaluation and save the best results
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

    def inference(self, viral_state, I_df, curing_prob):
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

        S_region, I_region, R_region = viral_state
        V_region = self.get_region_V(I_region, curing_prob)
        F_region = self.get_region_F(S_region, I_df)

        match self.parameter_optimizer:
            case Optimizers.CV:
                return LASSOCV.inference(F_region, V_region, random=self.random)
