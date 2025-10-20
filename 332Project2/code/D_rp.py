# Research Payoffs - 研究における不確実性とAI利用のモデル

 # 私は, 研究をしているときの不確実性とAIの利用に関してモデル化しました.
# これに関して,私たちがよく直面することとして無駄なことが研究に役立つという直感と、簡単に作業をこなすAIがその機会を奪うという直感があります。

# これは, ps3のProblem 1を現実の問題に拡張したものです。
# テクニカルには,関数fを特徴づけし, 現実に適合させるようにしました.

# より正確に言うと, 研究に際して,私たちが不確実性に直面しているということを、次のように特徴づけました。
# テクニカルには, 研究には, Conceptualな作業とMechanicalが伴い, これらの作業のうちどれが行われるかを,研究者はこれを行うまで知らないとします.
# これの直感は, 例えばコードを書いていたときに数式についての洞察を思いついたり、論文の調べ物をしていたときに全く違う領域との接点を閃いたりすることです.

# each round i;
# [0,1]か[1,0]を選びます。それぞれ、AIの利用と自分で研究をすることを表します。

# そして, AIはMechanicalな作業を得意としますが, Conceptualな作業を得意としません。
# 反対に、研究者はconceptualな作業を得意としますが, Mechanicalな作業を得意としません。

# 研究者がAIを利用するのは、Mechanicalなものを自動的にこなすことの他にも、完成度を追求するという理由もあります。

# これをモデルに含めるために、以下のような工夫をしました。
# まず, インプットである選択から、直接payoffとせず, Knowledge gain と　draft progressの二つの指標が得らるようにした.
# AIはdraft progressを得意としますが, Knowledge gainを得意としません。
# 反対に、研究者はKnowledge gainを得意としますが, draft progressを得意としません。
# そして, Knowledge gain と　draft progressから, その時のpayoffを二つの線形結合で表現します。
# 具体的には、0から、lambda_1までの値を取るlambda_tを用いて、以下のように表現します。
# payoff = lambda_t * Knowledge gain + (1 - lambda_t) * draft progress
# ここで, lambda_tは0から1までの値を取るパラメータです。
# lambda_tは時間の経過とともに、とても滑らかに、最初のroundで0.3をとり, 最後のroundで0.8をとるように、指数関数等用いて、変化させます。
# これの直感は、前半は理解を重視し、後半は完成度を重視するというものです。

# フルインフォメーションで、研究者は毎回別の行動に関する、利得について計算できるとします.

# 研究の不確実性の他にも研究者がAIを使う理由はあります.例えば、研究の進捗の不安等です.

# 最終的なグラフの出力は、payoffの合計、理解度の合計と、完成度の合計です。それぞれの合計は、roundごとに計算されます。

# Mechanicalの時, 
# [0,1]を選ぶと, Knowledge gain = 0.2, draft progress = 0.9
# [1,0]を選ぶと, Knowledge gain = 0.5, draft progress = 0.6

# Conceptualの時, 
# [0,1]を選ぶと, Knowledge gain = 0.1, draft progress = 0.4
# [1,0]を選ぶと, Knowledge gain = 0.8, draft progress = 0.3

# ここでの, 不等式制約をかく.
# その中では, 成り立つということを示す.
# conceptualな作業の場合, human_knowledge > AI_knowledge, human_progress > AI_progress,
# mechanicalな作業の場合, human_knowledge < AI_knowledge, human_progress < AI_progress,
# 厳密に、conceptual_knowledge > mechanical_knowledge (human)
# 厳密に, conceptual_progress < mechanical_progress (human)
# この中で、自由に動ける

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ResearchPayoffs:
    """
    研究における不確実性とAI利用のモデル（知識蓄積版）
    
    各ラウンドで研究者は以下の選択を行う:
    - [0,1]: AIを利用する
    - [1,0]: 自分で研究する
    
    研究には2つのタイプがある:
    - Mechanical: 機械的な作業（AIが得意）
    - Conceptual: 概念的作業（研究者が得意）
    
    各選択から2つの指標が得られる:
    - Knowledge gain: 理解度の向上（そのラウンドでの基本値）
    - Draft progress: 完成度の向上（総知識に依存）
    
    知識蓄積効果:
    - 蓄積された知識は将来の進捗向上に寄与
    - knowledge_bonus = tanh(cumulative_knowledge * 0.1)
    - progress = base_progress * (1 + knowledge_bonus)
    
    最終的なpayoffは時間とともに重みが変わる線形結合:
    payoff = lambda_t * Knowledge_gain + (1 - lambda_t) * Draft_progress
    """
    
    def __init__(self, k=2, n=1000):
        self.k = k  # 2つの選択肢: [0,1] (AI利用) と [1,0] (自分で研究)
        self.n = n  # 総ラウンド数
        self.cumulative_knowledge = 0
        self.cumulative_progress = 0
        self.cumulative_payoff = 0
        
        # 研究タイプの状態遷移（粘着性あり）
        self.current_research_type = None  # 現在の研究タイプ
        self.stay_prob = 0.8  # 同じ状態を維持する確率
        self.switch_prob = 0.2  # 別の状態に移る確率
        
        # mechanicalタスク（作業寄り）
        ai_k_m = np.random.uniform(0.1, 0.4)
        ai_p_m = np.random.uniform(0.8, 1.0)
        hu_k_m = np.random.uniform(ai_k_m + 0.05, 0.7)
        hu_p_m = np.random.uniform(0.5, ai_p_m - 0.05)

        # conceptualタスク（概念寄り）
        ai_k_c = np.random.uniform(0.05, ai_k_m - 0.02)
        ai_p_c = np.random.uniform(0.3, hu_p_m - 0.05)
        hu_k_c = np.random.uniform(hu_k_m + 0.05, 1.0)
        hu_p_c = np.random.uniform(0.2, ai_p_c - 0.05)

        # 不等式チェック（安全側で保証）
        assert hu_k_c > ai_k_c
        assert hu_p_c > ai_p_c
        assert ai_k_m < hu_k_m
        assert ai_p_m > hu_p_m
        assert hu_k_c > hu_k_m          # conceptual > mechanical（knowledge）
        assert hu_p_c < hu_p_m          # conceptual < mechanical（progress）

        # 各選択肢の効果（Knowledge gain, Draft progress）
        self.effects = {
            'mechanical': {
                'ai': (ai_k_m, ai_p_m),
                'human': (hu_k_m, hu_p_m)
            },
            'conceptual': {
                'ai': (ai_k_c, ai_p_c),
                'human': (hu_k_c, hu_p_c)
            }
        }
        
        # 時間とともに変化する重みパラメータ
        # 最初は理解重視(0.3)、最後は完成度重視(0.8)
        self.lambda_0 = 0.3
        self.lambda_1 = 0.8
        
    def _get_lambda(self, round_num):
        """時間とともに変化する重みパラメータを計算"""
        # 指数関数で滑らかに変化
        t = round_num / (self.n - 1) if self.n > 1 else 0
        return self.lambda_0 + (self.lambda_1 - self.lambda_0) * (1 - np.exp(-3 * t))
    
    def _determine_research_type(self):
        """研究タイプを決定（粘着性のある状態遷移）"""
        if self.current_research_type is None:
            # 最初のラウンド：ランダムに決定
            self.current_research_type = 'mechanical' if np.random.random() < 0.5 else 'conceptual'
        else:
            # 前の状態から遷移
            if np.random.random() < self.stay_prob:
                # 0.8の確率で同じ状態を維持
                pass  # self.current_research_type をそのまま使用
            else:
                # 0.2の確率で別の状態に移る
                self.current_research_type = 'conceptual' if self.current_research_type == 'mechanical' else 'mechanical'
        
        return self.current_research_type
    
    def _get_action_choice(self, action):
        """選択されたactionを文字列に変換"""
        if action == 0:
            return 'ai'      # [0,1] = AI利用
        else:
            return 'human'   # [1,0] = 自分で研究
    
    def generate_payoffs(self, round_num):
        """
        指定されたラウンドでpayoffを生成
        Knowledge gainを蓄積し、Draft progressが総知識に依存するモデル
        
        Args:
            round_num: 現在のラウンド番号
            
        Returns:
            np.array: 各選択肢のpayoff [AI利用のpayoff, 自分で研究のpayoff]
        """
        # 研究タイプを決定
        research_type = self._determine_research_type()
        
        # 各選択肢の基本効果を取得
        ai_effects = self.effects[research_type]['ai']
        human_effects = self.effects[research_type]['human']
        
        # 現在の重みパラメータを計算
        lambda_t = self._get_lambda(round_num)
        
        # Knowledge gain（基本値 - そのラウンドでの理解度向上）
        ai_knowledge = ai_effects[0]
        human_knowledge = human_effects[0]
        
        # Draft progress（総知識に依存）
        # 知識の蓄積による進捗向上（非線形、上限付き）
        knowledge_bonus = np.tanh(self.cumulative_knowledge * 0.1)  # 0-1の範囲
        
        ai_progress = ai_effects[1] * (1 + knowledge_bonus)
        human_progress = human_effects[1] * (1 + knowledge_bonus)
        
        # Payoff計算
        ai_payoff = lambda_t * ai_knowledge + (1 - lambda_t) * ai_progress
        human_payoff = lambda_t * human_knowledge + (1 - lambda_t) * human_progress
        
        return np.array([ai_payoff, human_payoff])
    
    def update_cumulative_stats(self, action, round_num):
        """
        選択されたactionに基づいて累積統計を更新
        知識蓄積による進捗向上モデル
        
        Args:
            action: 選択されたaction (0=AI利用, 1=自分で研究)
            round_num: 現在のラウンド番号
        """
        # 研究タイプを再取得（generate_payoffsで既に決定済み）
        research_type = self.current_research_type
        
        # 選択されたactionの基本効果を取得
        if action == 0:  # AI利用
            base_knowledge, base_progress = self.effects[research_type]['ai']
        else:  # 自分で研究
            base_knowledge, base_progress = self.effects[research_type]['human']
        
        # Knowledge gain（そのラウンドでの理解度向上）
        knowledge_gain = base_knowledge
        
        # Draft progress（総知識に依存）
        # 現在の総知識に基づく進捗向上
        knowledge_bonus = np.tanh(self.cumulative_knowledge * 0.1)
        progress_gain = base_progress * (1 + knowledge_bonus)
        
        # 累積値を更新
        self.cumulative_knowledge += knowledge_gain
        self.cumulative_progress += progress_gain
        
        # payoffも計算して累積
        lambda_t = self._get_lambda(round_num)
        payoff = lambda_t * knowledge_gain + (1 - lambda_t) * progress_gain
        self.cumulative_payoff += payoff
    
    def get_cumulative_stats(self):
        """累積統計を取得"""
        return {
            'cumulative_knowledge': self.cumulative_knowledge,
            'cumulative_progress': self.cumulative_progress,
            'cumulative_payoff': self.cumulative_payoff
        }
    
    def reset(self):
        """状態をリセット"""
        self.cumulative_knowledge = 0
        self.cumulative_progress = 0
        self.cumulative_payoff = 0
        self.current_research_type = None  # 研究タイプもリセット