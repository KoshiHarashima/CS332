# Exponential Weights Algorithm

import numpy as np


class ExponentialWeights:
    def __init__(self, k, epsilon, n, h=1):
        self.k = k
        self.epsilon = epsilon
        self.n = n
        self.h = h  # Scaling parameter
        self.cumulative_payoffs = np.zeros(k)
        self.regret_history = []
        self.total_payoff = 0    # Track our algorithm's total payoff
        self.payoff_history = []  # Track cumulative payoff over rounds
        self.optimal_cumulative_payoff = 0  # Track optimal strategy's cumulative payoff
        
    def select_action(self):
        if self.epsilon == 0:
            action = np.random.randint(0, self.k)
        else:
            # π_j^i = (1+ε)^(V_j^{i-1}/h) / Σ_j'(1+ε)^(V_j'^{i-1}/h)
            # where V_j^{i-1} = cumulative_payoffs[j] (previous cumulative payoffs)
            powers = (1 + self.epsilon) ** (self.cumulative_payoffs / self.h)
            sum_powers = np.sum(powers)
            
            if not np.isfinite(sum_powers) or sum_powers <= 0:
                action = np.random.randint(0, self.k)
            else:
                probabilities = powers / sum_powers
                if not np.all(np.isfinite(probabilities)):
                    action = np.random.randint(0, self.k)
                else:
                    action = np.random.choice(self.k, p=probabilities)
        return action
        
    def update_weights(self, payoffs, action):
        # 1. 最適戦略の累積ペイオフを更新（各ラウンドで最高ペイオフを選択）
        optimal_payoff = np.max(payoffs)
        self.optimal_cumulative_payoff += optimal_payoff
        
        # 2. アルゴリズムの累積ペイオフを更新
        self.total_payoff += payoffs[action]
        
        # 3. 全店舗の累積ペイオフを更新（重み計算用）
        self.cumulative_payoffs += payoffs
        
        # 4. リグレットを計算（最適戦略 - アルゴリズム）
        regret = self.optimal_cumulative_payoff - self.total_payoff
        self.regret_history.append(regret)
        
        # 5. ペイオフ履歴を更新
        self.payoff_history.append(self.total_payoff)
    def run_algorithm(self, payoff_generator):
        for round_num in range(self.n):
            action = self.select_action()
            payoffs = payoff_generator(round_num)
            self.update_weights(payoffs, action)
        return self.regret_history, self.total_payoff, self.cumulative_payoffs, self.payoff_history 