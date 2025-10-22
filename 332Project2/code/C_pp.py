# Here, we consider the Pachinko Payoffs based on ROI (Return on Investment).
# From Espace datasets, we collect the daily data of 5 stores, which seems quite unique. (海外の大学のWifiから最も欲しいデータセットは得ることができなかった.)
# Shops open the data about the number of balls in and out each day.
# We calculate ROI as (balls_out - balls_in) / investment_amount, then normalize it to [0,1] range.
# So, instead of applying the EW algorithm to each gumbuling machines, 
# we apply the EW algorithm to each store, and the payoff is the normalized ROI for each day.

# this setting seems to be unrealistic, 
# but it's common in Japan for gumbling specialists to make money by predicting the number of balls in and out for each stores and to betting it by playing collectively with a group of people.
# so considering that, we can apply the EW algorithm to each store.
# and in addtion, most common and normal strategy is to bet on the store with the highest number of balls in and out yesterday (which is FTL).

# each round, or we can say each day, we generate the ROI-based payoff for each store.
# there's five stores.
# each day, we choose stores to play gumbuling machines.
# then we calculate ROI for each store (extracted from the data) and normalize it to [0,1]
# then we apply the EW algorithm to each store
# That's all.

# Note; Stores strategically change settings so that store can gain more profit by inviting more customers.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class EspacePayoffs:
    """エスパス5店舗の日次データを使用したPayoffGenerator（ROIベース）"""
    
    def __init__(self, k):
        """
        エスパス5店舗のデータを読み込み、ROIベースのpayoff生成器を初期化
        
        Args:
            k: 店舗数（A_afp.pyとの互換性のため）
        """
        # データファイルのパスを設定
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / 'data'
        data_file_path = data_dir / 'espace_5stores_daily_2024_2025.csv'
        
        # データの読み込み
        self.data = pd.read_csv(data_file_path)
        
        # 店舗名の正規化（BOM文字を除去）
        self.data['store'] = self.data['store'].str.replace('\ufeff', '')
        
        # 店舗一覧を取得
        self.stores = self.data['store'].unique()
        self.k = len(self.stores)  # 店舗数
        
        # avg_diffの統計情報を計算
        self.min_avg_diff = self.data['avg_diff'].min()
        self.max_avg_diff = self.data['avg_diff'].max()
        
        # ROIの統計情報を事前計算（投資額1000を仮定）
        self.base_investment = 1000
        all_rois = self.data['avg_diff'] / self.base_investment
        self.min_roi = all_rois.min()
        self.max_roi = all_rois.max()
        
        # 各店舗の平均avg_diffを事前計算
        self.store_means = {}
        for store in self.stores:
            store_data = self.data[self.data['store'] == store]
            self.store_means[store] = store_data['avg_diff'].mean()
        
        # 状態変数
        self.current_day = 0
        self.total_days = len(self.data)

        for store, mean_val in self.store_means.items():
            print(f"  {store}: {mean_val:.2f}")
        
    def calculate_roi(self, avg_diff, base_investment=1000):
        """avg_diffをROI（投資収益率）として計算"""
        # avg_diffを利益として扱い、ROI = 利益 / 投資額 で計算
        # 負の値の場合は損失として扱う
        roi = avg_diff / base_investment
        return roi
    
    def normalize_roi(self, roi):
        """ROIを[0,1]の範囲に正規化"""
        # 事前計算されたROIの範囲を使用
        normalized_roi = (roi - self.min_roi) / (self.max_roi - self.min_roi)
        return max(0, min(1, normalized_roi))  # 0-1の範囲にクリップ
    
    def generate_payoffs(self, round_num):
        """指定されたラウンドで各店舗のpayoffを生成（ROIベース）"""
        if self.current_day >= self.total_days:
            # データが尽きた場合は最後の日を繰り返し
            day_data = self.data.iloc[-1]
        else:
            day_data = self.data.iloc[self.current_day]
        
        # 各店舗のpayoffを計算（ROIベース）
        payoffs = np.zeros(self.k)
        for i, store in enumerate(self.stores):
            if day_data['store'] == store:
                # その日の実際の店舗のavg_diffを使用
                avg_diff = day_data['avg_diff']
            else:
                # 他の店舗の場合は、その店舗の過去の平均を使用
                avg_diff = self.store_means[store]
            
            # ROIを計算して正規化
            roi = self.calculate_roi(avg_diff)
            normalized_payoff = self.normalize_roi(roi)
            payoffs[i] = normalized_payoff
        
        self.current_day += 1
        return payoffs
    
    def reset(self):
        """状態をリセット"""
        self.current_day = 0




