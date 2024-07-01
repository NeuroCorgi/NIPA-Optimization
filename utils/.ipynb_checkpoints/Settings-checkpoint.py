from utils.Type import Type
from utils.Country import Country
from utils.Optimizers import Optimizers
from utils.Evaluations import Evaluation
from utils.visualisation import Visualizations

class Settings:
    def __init__(self, args):
        """
        Initialize the Settings object with command-line arguments.

        Parameters
        ----------
        args : Namespace
            The arguments parsed from the command line.
        """

        self.optimizers = self.set_optimizers(args.optimizers)
        self.evaluations = self.set_evaluations(args.evaluations)
        self.visualizations = self.set_visualizations(args.visuals)
        self.visual_days = args.visual_days
        self.country = self.set_country(args.country)
        self.type = self.set_type(args.type)
        self.n_days = args.n_days
        self.pred_days = self.set_pred_days(args.pred_days)
        self.compensate_fluctuations = args.compensate_fluctuations
        self.random = args.random

        if self.type == Type.ORIGINAL_NIPA and args.train_days is not None:
            raise Exception("The train_days argument is only available for the static and dynamic NIPA types.")
        else:
            self.train_days = args.train_days

        self.print()
        
    def set_pred_days(self, pred_days):
        """
        Set the prediction days from a comma-separated string.

        Parameters
        ----------
        pred_days : str
            Comma-separated string of prediction days.

        Returns
        -------
        list of int
            List of prediction days.
        """
        try:
            return [int(day) for day in pred_days.split(',')]
        except ValueError as e:
            raise ValueError(f"{pred_days} contains non-integer values!") from e

    def set_evaluations(self, evaluations):
        """
        Set the evaluation metrics from a comma-separated string.

        Parameters
        ----------
        evaluations : str
            Comma-separated string of evaluation metrics.

        Returns
        -------
        list of Evaluation
            List of evaluation metrics.
        """
        evaluation_map = {
            'mse': Evaluation.MSE,
            'mape': Evaluation.MAPE,
            'smape': Evaluation.sMAPE
        }

        try:
            return [evaluation_map[eval] for eval in evaluations.split(',')]
        except KeyError as e:
            raise ValueError(f"Invalid evaluation metric: {e.args[0]}")
    
    def set_visualizations(self, visualizations):
        """
        Set the visualizations from a comma-separated string.

        Parameters
        ----------
        visualizations : str
            Comma-separated string of visualizations.

        Returns
        -------
        list of Visualizations
            List of visualizations.
        """
        visualization_map = {
            'all': Visualizations.ALL,
            'all_pred': Visualizations.ALL_PREDICTIONS,
            'all_eval': Visualizations.ALL_EVALUATIONS,
            'heatmap': Visualizations.HEATMAP,
            'iter_pred': Visualizations.ITER_PREDICTIONS,
            'iter_eval': Visualizations.ITER_EVALUATIONS
        }
    
        try:
            return [visualization_map[vis] for vis in visualizations.split(',')]
        except KeyError as e:
            raise ValueError(f"Invalid visualization: {e.args[0]}")

    def set_optimizers(self, optimizers):
        """
        Set the optimizers from a comma-separated string.

        Parameters
        ----------
        optimizers : str
            Comma-separated string of optimizers.

        Returns
        -------
        list of Optimizers
            List of optimizers.
        """
        optimizer_map = {
            'cv': Optimizers.CV,
            'gsa': Optimizers.GENERALIZED_SIMULATED_ANNEALING,
            'dsa': Optimizers.DUAL_SIMULATED_ANNEALING,
            'full_dsa': Optimizers.DUAL_SIMULATED_ANNEALING_ALL
        }
        try:
            return [optimizer_map[opt] for opt in optimizers.split(',')]
        except KeyError as e:
            raise ValueError(f"Invalid optimizer: {e.args[0]}")
        
    def set_country(self, country):
        """
        Set the country based on a string input.

        Parameters
        ----------
        country : str
            The country name.

        Returns
        -------
        Country
            The corresponding Country enum value.
        """
        country_map = {
            'hubei': Country.Hubei,
            'mexico': Country.Mexico,
            'netherlands': Country.Netherlands
        }
        try:
            return country_map[country]
        except KeyError as e:
            raise ValueError(f"Invalid country: {e.args[0]}")
        
    def set_type(self, type):
        """
        Set the type of NIPA model based on a string input.

        Parameters
        ----------
        type : str
            The type of NIPA model.

        Returns
        -------
        Type
            The corresponding Type enum value.
        """
        type_map = {
            'original': Type.ORIGINAL_NIPA,
            'static': Type.STATIC_NIPA,
            'dynamic': Type.DYNAMIC_NIPA
        }
        try:
            return type_map[type]
        except KeyError as e:
            raise ValueError(f"Invalid NIPA type: {e.args[0]}")
        
    def set_n_days(self, n_days):
        """
        Set the number of days for the NIPA model.

        Parameters
        ----------
        n_days : int
            The number of days to set.
        """
        self.n_days = n_days
        
    def print(self):
        """
        Print the settings for the current run.
        """

        print("Settings for this run:")
        print(f"\t Country: {self.country} \t Optimizers: {self.optimizers} \t Type: {self.type} \t Compensate fluctuations: {self.compensate_fluctuations}")
        if self.type == Type.DYNAMIC_NIPA or self.type == Type.STATIC_NIPA:
            print(f"\t Days of training: {self.train_days} \t Days to predict: {self.pred_days}")
        else:
            print(f"\t Days to predict: {self.pred_days}")