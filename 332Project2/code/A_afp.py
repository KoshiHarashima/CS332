# Adversarial Fair Payoffs

# In each round i:
# \begin{itemize}
#     \item Draw a payoff $x \sim U[0,1]$ (i.e., from the uniform distribution on interval [0,1])
#     \item Assign this payoff to the action $j^*$ that has the smallest total payoff so far,\\
#     i.e., $j^* = \arg\min_j V^{i-1}_{j} \quad \text{where} \quad V^{i}_{j} = \sum_{r=1}^{i} v^{r}_{j}$
#     \item (All other actions get 0 payoff in round i.)
# \end{itemize} 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AdversarialFairPayoffs:
    def __init__(self, k):
        self.k = k
        self.cumulative_payoffs = np.zeros(k)

    def generate_payoffs(self, round_num):
        payoff = np.random.uniform(0, 1)
        min_action = np.argmin(self.cumulative_payoffs)
        payoffs = np.zeros(self.k)
        payoffs[min_action] = payoff
        self.cumulative_payoffs += payoffs
        return payoffs
