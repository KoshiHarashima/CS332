# Adversarial Fair Payoffs

# In each round i:
# \begin{itemize}
#     \item Draw a payoff $x \sim U[0,1]$
#     \item If there are several actions whose total payoff are zero, payoff is randomly assigned to these actions.
#     \item Assign this payoff to the action $j^*$ that has the smallest total payoff so far,\\
#     i.e., $j^* = \arg\min_j V^{i-1}_{j} \quad \text{where} \quad V^{i}_{j} = \sum_{r=1}^{i} v^{r}_{j}$
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
        
        # Find all actions with minimum cumulative payoff
        min_payoff = np.min(self.cumulative_payoffs)
        min_actions = np.where(self.cumulative_payoffs == min_payoff)[0]
        
        # Randomly select one of the actions with minimum payoff
        min_action = np.random.choice(min_actions)
        
        payoffs = np.zeros(self.k)
        payoffs[min_action] = payoff
        self.cumulative_payoffs += payoffs
        return payoffs
