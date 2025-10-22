# Exponential Weights Algorithm

# \begin{frame}{Basic Setting : EW Algorithm}
#     \textbf{Algorithm}\\
#     Exponential Weight Algorithm is implemented as follow; 
#     \begin{itemize}
#         \item learning rate $\epsilon$
#         \item let $V_j^i = \sum_{r = 1}^i v_j^i$
#         \item in round i choose j with probability $\pi_j^i$ proportional to $(1+\epsilon)^{\frac{V_j^{i-1}}{h}}$
#         \item implemented as follow;
#         \[
#             \pi_j^i = \frac{(1+\epsilon)^{\frac{V_j^{i-1}}{h}}}{\sum_{j'}(1+\epsilon)^{\frac{V_{j'}^{i-1}}{h}}}
#         \]
#     \end{itemize}
#     \textbf{learning rates}\\
#     \begin{itemize}
#       \item No learning: \(\epsilon = 0\) .
#       \item Theoretical: \(\epsilon = \sqrt{\ln k / n}\).
#       \item FTL:  \(\epsilon \approx \infty\).
#     \end{itemize}
# \end{frame}

import numpy as np


class ExponentialWeights:
    def __init__(self, k, epsilon, n):
        self.k = k
        self.epsilon = epsilon
        self.n = n
        self.log_weights = np.zeros(k)
        self.cumulative_payoffs = np.zeros(k)
        self.regret_history = []
        self.total_payoff = 0    # Track our algorithm's total payoff
        
    def select_action(self):
        if self.epsilon == 0:
            action = np.random.randint(0, self.k)
        else:
            max_log = np.max(self.log_weights)
            exps = np.exp(self.log_weights - max_log)
            sum_exps = np.sum(exps)
            if not np.isfinite(sum_exps) or sum_exps <= 0:
                action = np.random.randint(0, self.k)
            else:
                probabilities = exps / sum_exps
                if not np.all(np.isfinite(probabilities)):
                    action = np.random.randint(0, self.k)
                else:
                    action = np.random.choice(self.k, p=probabilities)
        return action
        
    def update_weights(self, payoffs, action):
        self.cumulative_payoffs += payoffs
        self.total_payoff += payoffs[action]
        self.log_weights += self.epsilon * payoffs
        self.regret_history.append(np.max(self.cumulative_payoffs) - self.total_payoff)
    def run_algorithm(self, payoff_generator):
        for round_num in range(self.n):
            action = self.select_action()
            payoffs = payoff_generator(round_num)
            self.update_weights(payoffs, action)
        return self.regret_history, self.total_payoff, self.cumulative_payoffs 