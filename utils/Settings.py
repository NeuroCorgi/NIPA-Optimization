from utils.Type import Type
from utils.Country import Country
from utils.Optimizers import Optimizers

class Settings:
    def __init__(self, args):
        self.optimizer = self.set_optimizer(args.optimizer)
        self.country = self.set_country(args.country)
        self.type = self.set_type(args.type)
        self.n_days = args.n_days
        self.pred_days = args.pred_days
        self.compensate_fluctuations = args.compensate_fluctuations

        if self.type == Type.ORIGINAL_NIPA and args.train_days != None:
            raise Exception("The train_days argument is only available for the static and dynamic NIPA types.")
        else:
            self.train_days = args.train_days

        print("Settings for this run:")
        print(f"\t Country: {self.country} \t Optimizer: {self.optimizer} \t Type: {self.type} \t Compensate fluctuations: {self.compensate_fluctuations}")
        if self.type == Type.DYNAMIC_NIPA or self.type == Type.STATIC_NIPA:
            print(f"\t Days of training: {self.train_days} \t Days to predict: {self.pred_days}")
        else:
            print(f"\t Days to predict: {self.pred_days}")
        


    def set_optimizer(self, optimizer):
        if optimizer == 'lassocv':
            return Optimizers.LASSOCV
        elif optimizer == 'lasso':
            return Optimizers.LASSOCV_OWN
        elif optimizer == 'sa':
            return Optimizers.SIMULATED_ANNEALING
        
    def set_country(self, country):
        if country == 'hubei':
            return Country.Hubei
        elif country == 'mexico':
            return Country.Mexico
        elif country == 'netherlands':
            return Country.Netherlands
        
    def set_type(self, type):
        if type == 'original':
            return Type.ORIGINAL_NIPA
        elif type == 'static':
            return Type.STATIC_NIPA
        elif type == 'dynamic':
            return Type.DYNAMIC_NIPA