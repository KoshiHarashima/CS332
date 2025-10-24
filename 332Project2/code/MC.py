# implement Monte Carlo simulation for each environment
# \begin{frame}{Basic Setting; MC Simulation}
#     we implemented MC Simulation as follow;
#     \begin{itemize}
#         \item fix k, n
#         \item for each times, 
#         \begin{enumerate}
#             \item set $\epsilon$ to {$0.0001, \sqrt{log\frac{k}{n}}, 10000$}
#             \item simulate it through implemented algorithms in each setting.
#             \item calculate Regret (total Payoff, and his set of choices)
#         \end{enumerate}
#         \item then aggregate these results, and calculate mean of Regret and their confident intervals.
#     \end{itemize}
# \end{frame}


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