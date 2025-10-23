# Bernoulli Payoffs

# Fix a probability for each action $p_{1},...,p_{k}$ with each $p_{k}$ in [0,1/2].\\
# In each round i,
# \begin{itemize}
#     \item draw the payoff of each action j as $v^{i}_{j} \sim B(p_{j})$ (i.e, from the Bernoulli distribution with probability $p_j$ of being 1 and probability $1-p_{j}$ of being 0).
# \end{itemize}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BernoulliPayoffs:
    def __init__(self, k):
        self.k = k
        self.probabilities = np.random.uniform(0, 0.5, k)
        print(f"Bernoulli probabilities for each action: {self.probabilities}")

    def generate_payoffs(self, round_num):
        return np.random.binomial(1, self.probabilities)    # returns an array of payoffs for each action   