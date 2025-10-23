# Exponentiated Gradient Algorithm for Research Payoffs

import numpy as np
from typing import Callable, Tuple

class ExponentiatedGradientFI:
    """
    Full-information Exponentiated Gradient (Hedge for gains).
    更新: w_{t+1} ∝ w_t ⊙ exp(ε · α_t)
    後悔: max_j (∑_τ α_{τ,j}) − ∑_τ α_τ^T w_τ  （最良固定アクション基準）
    """

    def __init__(self, k: int, epsilon: float, n: int):
        self.k = int(k)
        self.epsilon = float(epsilon)
        self.n = int(n)

        self.weights = np.ones(self.k) / self.k
        self.cumulative_alpha = np.zeros(self.k)

        self.total_payoff = 0.0
        self.regret_history = []
        self.payoff_history = []
        self.weight_history = []

    def select_weights(self) -> np.ndarray:
        return self.weights.copy()

    def update(self, alpha: np.ndarray) -> None:
        """
        alpha: shape (k,) の利得ベクトル（full-information）
        """
        alpha = np.asarray(alpha, dtype=float)
        assert alpha.shape == (self.k,)

        # 現期利得と累積の更新
        current_payoff = float(np.dot(alpha, self.weights))
        self.total_payoff += current_payoff
        self.cumulative_alpha += alpha

        # EG更新（数値安定のための平行移動は不要だが、必要なら alpha -= alpha.max() を入れる）
        unnorm = self.weights * np.exp(self.epsilon * alpha)
        self.weights = unnorm / unnorm.sum()

        # 後悔（最良固定アクション）
        best_fixed = float(np.max(self.cumulative_alpha))
        regret = best_fixed - self.total_payoff

        # 記録
        self.regret_history.append(regret)
        self.payoff_history.append(self.total_payoff)
        self.weight_history.append(self.weights.copy())

    def run(self, payoff_generator: Callable[[int], np.ndarray]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        payoff_generator(t) -> alpha_t（shape (k,)）
        収益計算は update 内のみで行う（重複集計なし）
        """
        cumulative_payoffs = np.zeros(self.k, dtype=float)
        for t in range(self.n):
            alpha_t = payoff_generator(t)
            self.update(alpha_t)
            cumulative_payoffs += alpha_t

        return (
            np.array(self.regret_history),
            float(self.total_payoff),
            cumulative_payoffs,
            np.array(self.payoff_history, dtype=float),
        )

    def get_current_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_regret_history(self) -> np.ndarray:
        return np.array(self.regret_history, dtype=float)

    def get_payoff_history(self) -> np.ndarray:
        return np.array(self.payoff_history, dtype=float)

    def get_weight_history(self) -> np.ndarray:
        return np.array(self.weight_history, dtype=float)

    def reset(self) -> None:
        self.weights = np.ones(self.k) / self.k
        self.cumulative_alpha = np.zeros(self.k)
        self.total_payoff = 0.0
        self.regret_history = []
        self.payoff_history = []
        self.weight_history = []
