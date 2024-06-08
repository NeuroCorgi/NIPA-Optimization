from Optimizers import Optimizer
from Country import Country
from main import Type


class Options:
    def __init__(self, optimizer=Optimizer.LASSOCV, country=Country.Hubei, type=Type.ORIGINAL_NIPA, random_state=22, k_fold=3, max_iter=2e2, tol=1e-8):
        self.optimizer = optimizer
        self.country = country
        self.type = type
        self.random_state = random_state
        self.k_fold = k_fold
        self.max_iter = int(max_iter)
        self.tol = tol