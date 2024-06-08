import polars as pl
import numpy as np
import time
from datetime import datetime as dt
from datetime import timedelta

from utils.Optimizers import Optimizers
from optimizers import LASSOCV, LASSOCV_OWN, SimulatedAnnealing
import utils.show_data as show_data
from utils.io import save_train_data, get_train_data_path, save_results_data, get_results_data_path

class NIPA:
    def __init__(self, data, regions, country, type, dates, parameter_optimizer=Optimizers.LASSOCV):
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
        I = I.with_columns(pl.Series("i", [str(i+1) for i in range(len(self.regions))]))
        return I.transpose(include_header=True, header_name="k", column_names="i")

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
        save_date = train_days[0]
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
            MSE_region_list = []
            curing_prob_region_list = []
            r_parameters_list = []
            p_min_max_list = []

            ts = time.time()
            print(f"Training Region {self.regions[i]} ({i}/{len(self.regions.keys())})...")
            for curing_prob in curing_probs: 
                # print(f"Training for curing probability {curing_prob} at time {time.time()-ts:.2f}")
                R_region = self.calc_R(I_region, curing_prob)
                S_region = self.calc_S(R_region, I_region)
                        
                anomaly_R = [R for R in R_region if R > 1]

                if len(anomaly_R) > 0:
                    print(f"Anomaly R values: {anomaly_R}")
                    continue
                
                viral_state_region = [S_region, I_region, R_region]
                I_df = self.I[k_list[0]-1:k_list[-1], 1:]

                B_region, MSE_region, p_value, p_min_max = self.inference(viral_state_region, I_df, curing_prob)

                B_region_list.append(B_region)
                MSE_region_list.append(MSE_region)
                curing_prob_region_list.append(curing_prob)
                r_parameters_list.append([p_value, p_min_max[0], p_min_max[1]])
                p_min_max_list.append(p_min_max)
            
            min_index = MSE_region_list.index(min(MSE_region_list))
            self.curing_probs = self.curing_probs.with_columns(pl.Series(str(i), [curing_prob_region_list[min_index]]))
            r_parameters = r_parameters.with_columns(pl.Series(self.regions[i], r_parameters_list[min_index]))

            B = B.with_columns(pl.Series(self.regions[i], B_region_list[min_index]))

            te = time.time()
            print(f"Region {self.regions[i]} took {te-ts:.4f} seconds to train")
            print(f'Curing probability: {curing_probs[min_index]}')
            print(f'P min: {p_min_max_list[min_index][0]} P max: {p_min_max_list[min_index][1]}')
            print(f'Regularization value: {r_parameters_list[min_index][0]}')
            print(f'MSE: {MSE_region_list[min_index]}')
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
            the mean squared error of the model predictions (region_MSE), the selected alpha value
            for the Lasso model (p_value), and the range of candidate alpha values (p_min, p_max).
        """

        S_region, I_region, R_region = viral_state
        V_region = self.get_region_V(I_region, curing_prob)
        F_region = self.get_region_F(S_region, I_df)

        match self.parameter_optimizer:
            case Optimizers.LASSOCV:
                return LASSOCV.inference(F_region, V_region)
            case Optimizers.LASSOCV_OWN:
                return LASSOCV_OWN.inference(F_region, V_region)
            case Optimizers.SIMULATED_ANNEALING:
                return SimulatedAnnealing.inference(F_region, V_region, 1000)
    