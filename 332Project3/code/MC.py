# MC.py
import numpy as np
import pandas as pd

class MonteCarloSimulation:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.epsilon = np.array([0.01, np.sqrt(np.log(k/n)), 100])
        self.results = []

    def run(self, environment, algorithm):
        for epsilon in self.epsilon:
            algorithm.epsilon = epsilon
            results = algorithm.run(environment)
            self.results.append(results)
        return self.results