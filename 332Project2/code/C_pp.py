# Here, we consider the Pachinko Payoffs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class EspacePayoffs:
    """エスパス5店舗の日次データを使用したPayoffGenerator"""
    
    def __init__(self, k):
        """
        エスパス5店舗のデータを読み込み、前処理を行う
        
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
        
    def normalize_payoff(self, avg_diff):
        """avg_diffを[0,1]の範囲に正規化"""
        return (avg_diff - self.min_avg_diff) / (self.max_avg_diff - self.min_avg_diff)
    
    def generate_payoffs(self, round_num):
        """指定されたラウンドで各店舗のpayoffを生成"""
        if self.current_day >= self.total_days:
            # データが尽きた場合は最後の日を繰り返し
            day_data = self.data.iloc[-1]
        else:
            day_data = self.data.iloc[self.current_day]
        
        # 各店舗のpayoffを計算
        payoffs = np.zeros(self.k)
        for i, store in enumerate(self.stores):
            if day_data['store'] == store:
                # その日の実際の店舗のavg_diffを使用
                avg_diff = day_data['avg_diff']
            else:
                # 他の店舗の場合は、その店舗の過去の平均を使用
                avg_diff = self.store_means[store]
            
            normalized_payoff = self.normalize_payoff(avg_diff)
            payoffs[i] = normalized_payoff
        
        self.current_day += 1
        return payoffs
    
    def reset(self):
        """状態をリセット"""
        self.current_day = 0




