"""
量子インスパイアード 株価予測モデル

擬似量子ビットを活用した金融予測システム
- 量子並列市場分析
- 量子ウォークによる価格変動
- 量子干渉でトレンド予測
- ポートフォリオ最適化
- リスク評価
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
from pseudo_qubit import PseudoQubit


# ============================================================
# 株式データ
# ============================================================

@dataclass
class StockData:
    """株価データ"""
    symbol: str
    prices: List[float]
    volumes: List[float]
    dates: List[str]
    
    @property
    def returns(self) -> np.ndarray:
        """リターン（収益率）"""
        prices = np.array(self.prices)
        return np.diff(prices) / prices[:-1]
    
    @property
    def volatility(self) -> float:
        """ボラティリティ（変動率）"""
        return np.std(self.returns)
    
    @property
    def trend(self) -> float:
        """トレンド（傾向）: -1〜1"""
        if len(self.prices) < 2:
            return 0.0
        recent = np.mean(self.prices[-5:])
        older = np.mean(self.prices[-20:-5]) if len(self.prices) >= 20 else self.prices[0]
        change = (recent - older) / (older + 1e-10)
        return np.clip(change * 10, -1, 1)


# ============================================================
# 量子市場状態
# ============================================================

class QuantumMarketState:
    """
    市場の量子状態
    
    上昇/下降を重ね合わせで表現
    """
    
    def __init__(self, n_scenarios: int = 8):
        self.n_scenarios = n_scenarios
        self.n_qubits = int(np.ceil(np.log2(n_scenarios)))
        
        # 各シナリオの確率振幅
        self.amplitudes = np.ones(2 ** self.n_qubits, dtype=complex) / np.sqrt(2 ** self.n_qubits)
        
        # シナリオラベル
        self.scenarios = ['急騰', '上昇', '緩やか上昇', '横ばい', 
                         '緩やか下落', '下落', '急落', '暴落'][:n_scenarios]
    
    def _normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes /= norm
    
    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes[:self.n_scenarios]) ** 2
    
    def apply_market_data(self, trend: float, volatility: float, sentiment: float):
        """
        市場データで量子状態を更新
        
        trend: トレンド (-1=下落, 1=上昇)
        volatility: ボラティリティ
        sentiment: 市場センチメント (-1=悲観, 1=楽観)
        """
        # 各指標から量子ビットを生成
        trend_qubit = PseudoQubit(correlation=trend)
        vol_qubit = PseudoQubit(correlation=-volatility * 2)  # 高ボラ=不確実
        sent_qubit = PseudoQubit(correlation=sentiment)
        
        # 量子干渉で確率を調整
        for i in range(self.n_scenarios):
            # シナリオごとのスコア
            scenario_score = 1.0 - (i / (self.n_scenarios - 1)) * 2  # 1 to -1
            
            # 干渉計算
            interference = (
                trend_qubit.probabilities[0] * (1 + scenario_score) +
                sent_qubit.probabilities[0] * (1 + scenario_score) * 0.5 +
                vol_qubit.probabilities[1] * abs(scenario_score) * 0.3
            )
            
            self.amplitudes[i] *= np.sqrt(interference + 0.1)
        
        self._normalize()
    
    def measure(self) -> Tuple[int, str]:
        """測定（シナリオを確定）"""
        probs = self.probabilities
        probs /= probs.sum()
        idx = np.random.choice(self.n_scenarios, p=probs)
        return idx, self.scenarios[idx]
    
    def top_scenarios(self, k: int = 3) -> List[Tuple[str, float]]:
        """上位k個のシナリオ"""
        probs = self.probabilities
        indices = np.argsort(probs)[::-1][:k]
        return [(self.scenarios[i], probs[i]) for i in indices]


# ============================================================
# 量子ウォーク価格モデル
# ============================================================

class QuantumWalkPriceModel:
    """
    量子ウォークによる価格変動モデル
    
    量子的な重ね合わせで複数の価格パスを同時に探索
    """
    
    def __init__(self, initial_price: float, volatility: float = 0.02):
        self.initial_price = initial_price
        self.volatility = volatility
        self.price_paths: List[List[float]] = []
    
    def quantum_step(self, current_price: float, trend: float = 0.0) -> List[float]:
        """
        1ステップの量子ウォーク
        
        複数の可能な価格を重ね合わせで生成
        """
        # 量子ビットでステップを決定
        qubit = PseudoQubit(correlation=trend)
        
        # 上昇/下降の振幅
        up_amplitude = abs(qubit.state.alpha)
        down_amplitude = abs(qubit.state.beta)
        
        # 価格変動
        up_change = self.volatility * np.random.exponential(1)
        down_change = self.volatility * np.random.exponential(1)
        
        # 可能な価格（重ね合わせ）
        prices = [
            current_price * (1 + up_change),   # 上昇
            current_price * (1 - down_change), # 下降
            current_price * (1 + up_change * 0.5),  # 緩やか上昇
            current_price * (1 - down_change * 0.5), # 緩やか下落
        ]
        
        # 確率で重み付け
        weights = [
            up_amplitude ** 2,
            down_amplitude ** 2,
            up_amplitude ** 2 * 0.5,
            down_amplitude ** 2 * 0.5,
        ]
        weights = np.array(weights)
        weights /= weights.sum()
        
        return prices, weights
    
    def simulate(self, n_steps: int = 30, n_paths: int = 100, trend: float = 0.0) -> np.ndarray:
        """
        複数パスをシミュレート
        """
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.initial_price
        
        for path_idx in range(n_paths):
            current_price = self.initial_price
            
            for step in range(n_steps):
                # 量子ウォーク
                prices, weights = self.quantum_step(current_price, trend)
                
                # 量子測定
                qubit = PseudoQubit(correlation=trend)
                if qubit.measure() == 0:
                    current_price = prices[0] if np.random.random() < 0.7 else prices[2]
                else:
                    current_price = prices[1] if np.random.random() < 0.7 else prices[3]
                
                paths[path_idx, step + 1] = current_price
        
        self.price_paths = paths
        return paths
    
    def predict(self, confidence: float = 0.95) -> Dict:
        """予測結果を返す"""
        if len(self.price_paths) == 0:
            return {}
        
        final_prices = self.price_paths[:, -1]
        
        lower = np.percentile(final_prices, (1 - confidence) / 2 * 100)
        upper = np.percentile(final_prices, (1 + confidence) / 2 * 100)
        mean = np.mean(final_prices)
        median = np.median(final_prices)
        
        return {
            'mean': mean,
            'median': median,
            'lower': lower,
            'upper': upper,
            'volatility': np.std(final_prices) / mean,
            'up_probability': np.mean(final_prices > self.initial_price),
        }


# ============================================================
# 量子ポートフォリオ最適化
# ============================================================

class QuantumPortfolioOptimizer:
    """
    量子アニーリング風ポートフォリオ最適化
    
    リスク・リターンの最適バランスを量子的に探索
    """
    
    def __init__(self, stocks: List[StockData]):
        self.stocks = stocks
        self.n_stocks = len(stocks)
        
        # 期待リターン
        self.expected_returns = np.array([np.mean(s.returns) for s in stocks])
        
        # 共分散行列
        returns_matrix = np.array([s.returns for s in stocks])
        min_len = min(len(r) for r in returns_matrix)
        returns_matrix = np.array([r[:min_len] for r in returns_matrix])
        self.covariance = np.cov(returns_matrix)
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """ポートフォリオリターン"""
        return np.dot(weights, self.expected_returns)
    
    def portfolio_risk(self, weights: np.ndarray) -> float:
        """ポートフォリオリスク（標準偏差）"""
        return np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights)))
    
    def sharpe_ratio(self, weights: np.ndarray, risk_free_rate: float = 0.01) -> float:
        """シャープレシオ"""
        ret = self.portfolio_return(weights)
        risk = self.portfolio_risk(weights)
        return (ret - risk_free_rate / 252) / (risk + 1e-10)
    
    def optimize(self, n_iterations: int = 1000, 
                 risk_tolerance: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """
        量子アニーリング風最適化
        """
        # 初期ウェイト（均等配分）
        best_weights = np.ones(self.n_stocks) / self.n_stocks
        best_score = self._objective(best_weights, risk_tolerance)
        
        current_weights = best_weights.copy()
        current_score = best_score
        
        # 温度スケジュール
        initial_temp = 1.0
        final_temp = 0.01
        
        for i in range(n_iterations):
            temp = initial_temp * (final_temp / initial_temp) ** (i / n_iterations)
            
            # 量子トンネリング：ランダムな重み変更
            new_weights = current_weights.copy()
            
            # 擬似量子ビットで変更する銘柄を選択
            for j in range(self.n_stocks):
                qubit = PseudoQubit(correlation=0.0)
                if qubit.measure() == 0:
                    change = np.random.normal(0, 0.1)
                    new_weights[j] += change
            
            # 正規化（合計1、非負）
            new_weights = np.maximum(new_weights, 0)
            new_weights /= new_weights.sum()
            
            new_score = self._objective(new_weights, risk_tolerance)
            
            # メトロポリス基準
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_weights = new_weights
                current_score = new_score
                
                if current_score > best_score:
                    best_weights = current_weights.copy()
                    best_score = current_score
        
        # 結果
        result = {
            'weights': best_weights,
            'expected_return': self.portfolio_return(best_weights) * 252,  # 年率
            'risk': self.portfolio_risk(best_weights) * np.sqrt(252),  # 年率
            'sharpe_ratio': self.sharpe_ratio(best_weights),
        }
        
        return best_weights, result
    
    def _objective(self, weights: np.ndarray, risk_tolerance: float) -> float:
        """目的関数（最大化）"""
        ret = self.portfolio_return(weights)
        risk = self.portfolio_risk(weights)
        return ret - risk_tolerance * risk


# ============================================================
# 量子センチメント分析
# ============================================================

class QuantumSentimentAnalyzer:
    """
    量子的センチメント分析
    
    複数の指標を重ね合わせで統合
    """
    
    def __init__(self):
        self.indicators = {}
    
    def add_indicator(self, name: str, value: float, weight: float = 1.0):
        """指標を追加 (value: -1〜1)"""
        self.indicators[name] = {
            'value': np.clip(value, -1, 1),
            'weight': weight
        }
    
    def analyze(self) -> Dict:
        """量子的に分析"""
        if not self.indicators:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        # 各指標を量子ビットに変換
        qubits = []
        weights = []
        
        for name, data in self.indicators.items():
            qubit = PseudoQubit(correlation=data['value'])
            qubits.append(qubit)
            weights.append(data['weight'])
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        # 量子干渉で統合センチメント計算
        bullish_prob = 0.0
        bearish_prob = 0.0
        
        for qubit, weight in zip(qubits, weights):
            bullish_prob += qubit.probabilities[0] * weight
            bearish_prob += qubit.probabilities[1] * weight
        
        # センチメントスコア
        sentiment = bullish_prob - bearish_prob
        confidence = abs(sentiment)
        
        # 測定
        measurements = []
        for _ in range(100):
            votes = sum(q.measure() == 0 for q in qubits)
            measurements.append(votes / len(qubits))
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'bullish_probability': bullish_prob,
            'bearish_probability': bearish_prob,
            'measurement_mean': np.mean(measurements),
            'measurement_std': np.std(measurements),
        }


# ============================================================
# 株価予測システム
# ============================================================

class QuantumStockPredictor:
    """
    量子インスパイアード株価予測システム
    """
    
    def __init__(self):
        self.market_state = QuantumMarketState()
        self.sentiment_analyzer = QuantumSentimentAnalyzer()
    
    def predict(self, stock: StockData, days: int = 30) -> Dict:
        """
        株価を予測
        """
        # 市場分析
        trend = stock.trend
        volatility = stock.volatility
        
        # テクニカル指標
        prices = np.array(stock.prices)
        
        # 移動平均
        ma5 = np.mean(prices[-5:])
        ma20 = np.mean(prices[-20:]) if len(prices) >= 20 else ma5
        ma_signal = (ma5 - ma20) / (ma20 + 1e-10)
        
        # RSI
        returns = stock.returns[-14:] if len(stock.returns) >= 14 else stock.returns
        gains = np.maximum(returns, 0)
        losses = np.maximum(-returns, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        rsi_signal = (rsi - 50) / 50  # -1〜1に正規化
        
        # センチメント設定
        self.sentiment_analyzer.indicators = {}
        self.sentiment_analyzer.add_indicator('trend', trend, 1.0)
        self.sentiment_analyzer.add_indicator('ma_signal', ma_signal * 5, 0.8)
        self.sentiment_analyzer.add_indicator('rsi', rsi_signal, 0.6)
        self.sentiment_analyzer.add_indicator('volatility', -volatility * 5, 0.4)
        
        sentiment_result = self.sentiment_analyzer.analyze()
        
        # 市場状態を更新
        self.market_state = QuantumMarketState()
        self.market_state.apply_market_data(
            trend=trend,
            volatility=volatility,
            sentiment=sentiment_result['sentiment']
        )
        
        # 量子ウォークでシミュレーション
        price_model = QuantumWalkPriceModel(
            initial_price=prices[-1],
            volatility=volatility
        )
        
        paths = price_model.simulate(
            n_steps=days,
            n_paths=1000,
            trend=sentiment_result['sentiment']
        )
        
        prediction = price_model.predict()
        
        # シナリオ予測
        scenarios = self.market_state.top_scenarios(5)
        
        return {
            'current_price': prices[-1],
            'predicted_price': prediction['mean'],
            'price_range': (prediction['lower'], prediction['upper']),
            'up_probability': prediction['up_probability'],
            'volatility': prediction['volatility'],
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'scenarios': scenarios,
            'technical': {
                'trend': trend,
                'rsi': rsi,
                'ma5': ma5,
                'ma20': ma20,
            },
            'paths': paths,
        }


# ============================================================
# 可視化
# ============================================================

class StockVisualizer:
    """株価可視化"""
    
    @staticmethod
    def plot_prediction(stock: StockData, prediction: Dict, days: int = 30):
        """予測結果をプロット"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#0a0a0f')
        
        for ax in axes.flat:
            ax.set_facecolor('#0a0a0f')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        # 1. 価格チャートと予測
        ax1 = axes[0, 0]
        
        # 過去の価格
        ax1.plot(stock.prices, color='#00d4ff', linewidth=2, label='実績')
        
        # 予測パス（サンプル）
        paths = prediction['paths']
        n_hist = len(stock.prices)
        
        for i in range(min(50, len(paths))):
            x = np.arange(n_hist - 1, n_hist + len(paths[i]) - 1)
            ax1.plot(x, paths[i], color='#8b5cf6', alpha=0.1, linewidth=0.5)
        
        # 予測範囲
        mean_path = np.mean(paths, axis=0)
        lower_path = np.percentile(paths, 5, axis=0)
        upper_path = np.percentile(paths, 95, axis=0)
        
        x_pred = np.arange(n_hist - 1, n_hist + days)
        ax1.plot(x_pred, mean_path, color='#ec4899', linewidth=2, label='予測（平均）')
        ax1.fill_between(x_pred, lower_path, upper_path, color='#ec4899', alpha=0.2, label='90%信頼区間')
        
        ax1.axvline(n_hist - 1, color='white', linestyle='--', alpha=0.5)
        ax1.set_title(f'{stock.symbol} 価格予測', color='#00d4ff', fontsize=14)
        ax1.set_xlabel('日数', color='white')
        ax1.set_ylabel('価格', color='white')
        ax1.legend(loc='upper left')
        
        # 2. シナリオ確率
        ax2 = axes[0, 1]
        
        scenarios = prediction['scenarios']
        names = [s[0] for s in scenarios]
        probs = [s[1] for s in scenarios]
        colors = ['#10b981', '#00d4ff', '#8b5cf6', '#ec4899', '#f59e0b'][:len(scenarios)]
        
        bars = ax2.barh(names, probs, color=colors)
        ax2.set_xlim(0, max(probs) * 1.2)
        ax2.set_title('シナリオ確率', color='#00d4ff', fontsize=14)
        ax2.set_xlabel('確率', color='white')
        
        for bar, prob in zip(bars, probs):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1%}', color='white', va='center')
        
        # 3. センチメント
        ax3 = axes[1, 0]
        
        sentiment = prediction['sentiment']
        confidence = prediction['confidence']
        
        # ゲージ表示
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        ax3.plot(r * np.cos(theta), r * np.sin(theta), color='white', linewidth=2)
        
        # センチメント針
        needle_angle = np.pi / 2 - sentiment * np.pi / 2
        ax3.annotate('', xy=(0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#ec4899', lw=3))
        
        ax3.text(0, -0.3, f'センチメント: {sentiment:.2f}', 
                color='white', ha='center', fontsize=14)
        ax3.text(0, -0.5, f'信頼度: {confidence:.1%}', 
                color='#8b5cf6', ha='center', fontsize=12)
        
        ax3.text(-1, 0, '弱気', color='#ec4899', ha='center', fontsize=10)
        ax3.text(1, 0, '強気', color='#10b981', ha='center', fontsize=10)
        ax3.text(0, 1.1, '中立', color='white', ha='center', fontsize=10)
        
        ax3.set_xlim(-1.3, 1.3)
        ax3.set_ylim(-0.7, 1.3)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title('市場センチメント', color='#00d4ff', fontsize=14)
        
        # 4. テクニカル指標
        ax4 = axes[1, 1]
        
        tech = prediction['technical']
        
        indicators = [
            ('トレンド', tech['trend'], -1, 1),
            ('RSI', (tech['rsi'] - 50) / 50, -1, 1),
            ('MA乖離', (tech['ma5'] - tech['ma20']) / tech['ma20'] * 10, -1, 1),
        ]
        
        y_pos = np.arange(len(indicators))
        
        for i, (name, value, min_val, max_val) in enumerate(indicators):
            # バー背景
            ax4.barh(i, 2, left=-1, color='white', alpha=0.1, height=0.5)
            # 値
            color = '#10b981' if value > 0 else '#ec4899'
            ax4.barh(i, value, left=0, color=color, height=0.5)
            ax4.text(-1.3, i, name, color='white', va='center', ha='right')
            ax4.text(1.1, i, f'{value:.2f}', color='white', va='center')
        
        ax4.axvline(0, color='white', linestyle='-', alpha=0.5)
        ax4.set_xlim(-1.5, 1.5)
        ax4.set_ylim(-0.5, len(indicators) - 0.5)
        ax4.set_yticks([])
        ax4.set_title('テクニカル指標', color='#00d4ff', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'stock_prediction_{stock.symbol}.png', facecolor='#0a0a0f', dpi=150)
        plt.show()
    
    @staticmethod
    def plot_portfolio(optimizer: QuantumPortfolioOptimizer, weights: np.ndarray, result: Dict):
        """ポートフォリオをプロット"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#0a0a0f')
        
        for ax in axes:
            ax.set_facecolor('#0a0a0f')
            ax.tick_params(colors='white')
        
        # 1. 配分
        ax1 = axes[0]
        
        symbols = [s.symbol for s in optimizer.stocks]
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(symbols)))
        
        wedges, texts, autotexts = ax1.pie(
            weights, labels=symbols, colors=colors,
            autopct='%1.1f%%', pctdistance=0.8
        )
        
        for text in texts + autotexts:
            text.set_color('white')
        
        ax1.set_title('最適ポートフォリオ配分', color='#00d4ff', fontsize=14)
        
        # 2. リスク・リターン
        ax2 = axes[1]
        
        # 個別銘柄
        for stock in optimizer.stocks:
            ret = np.mean(stock.returns) * 252
            risk = stock.volatility * np.sqrt(252)
            ax2.scatter(risk, ret, s=100, alpha=0.7)
            ax2.annotate(stock.symbol, (risk, ret), color='white', fontsize=10)
        
        # ポートフォリオ
        ax2.scatter(result['risk'], result['expected_return'], 
                   s=200, color='#ec4899', marker='*', label='最適ポートフォリオ')
        
        ax2.set_xlabel('リスク（年率標準偏差）', color='white')
        ax2.set_ylabel('期待リターン（年率）', color='white')
        ax2.set_title('リスク・リターン分析', color='#00d4ff', fontsize=14)
        ax2.legend()
        ax2.axhline(0, color='white', linestyle='--', alpha=0.3)
        
        for spine in ax2.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
        plt.savefig('portfolio_optimization.png', facecolor='#0a0a0f', dpi=150)
        plt.show()


# ============================================================
# サンプルデータ生成
# ============================================================

def generate_sample_stock(symbol: str, days: int = 100, 
                         initial_price: float = 100,
                         trend: float = 0.0,
                         volatility: float = 0.02) -> StockData:
    """サンプル株価データを生成"""
    prices = [initial_price]
    volumes = []
    dates = []
    
    for i in range(days):
        # 量子ウォークで価格生成
        qubit = PseudoQubit(correlation=trend)
        
        if qubit.measure() == 0:
            change = 1 + volatility * np.random.exponential(1)
        else:
            change = 1 - volatility * np.random.exponential(1)
        
        prices.append(prices[-1] * change)
        volumes.append(np.random.randint(100000, 1000000))
        dates.append(f'2024-{(i//30)+1:02d}-{(i%30)+1:02d}')
    
    return StockData(symbol=symbol, prices=prices, volumes=volumes, dates=dates)


# ============================================================
# デモ
# ============================================================

def demo():
    """株価予測デモ"""
    print("=" * 70)
    print("  量子インスパイアード 株価予測システム")
    print("  Quantum-Inspired Stock Prediction System")
    print("=" * 70)
    
    # ============================================================
    # 1. サンプルデータ生成
    # ============================================================
    print("\n" + "─" * 70)
    print("  1. サンプル株価データ生成")
    print("─" * 70)
    
    stocks = [
        generate_sample_stock('AAPL', 100, 150, 0.3, 0.02),
        generate_sample_stock('GOOGL', 100, 140, 0.1, 0.025),
        generate_sample_stock('MSFT', 100, 350, 0.2, 0.018),
        generate_sample_stock('AMZN', 100, 180, -0.1, 0.03),
        generate_sample_stock('TSLA', 100, 250, 0.0, 0.04),
    ]
    
    for stock in stocks:
        print(f"\n  {stock.symbol}:")
        print(f"    現在価格: ${stock.prices[-1]:.2f}")
        print(f"    トレンド: {stock.trend:+.2f}")
        print(f"    ボラティリティ: {stock.volatility:.2%}")
    
    # ============================================================
    # 2. 株価予測
    # ============================================================
    print("\n" + "─" * 70)
    print("  2. 量子株価予測")
    print("─" * 70)
    
    predictor = QuantumStockPredictor()
    
    for stock in stocks[:3]:
        print(f"\n  {stock.symbol} の30日予測:")
        
        prediction = predictor.predict(stock, days=30)
        
        print(f"    現在価格: ${prediction['current_price']:.2f}")
        print(f"    予測価格: ${prediction['predicted_price']:.2f}")
        print(f"    予測範囲: ${prediction['price_range'][0]:.2f} - ${prediction['price_range'][1]:.2f}")
        print(f"    上昇確率: {prediction['up_probability']:.1%}")
        print(f"    センチメント: {prediction['sentiment']:+.2f}")
        
        print(f"    シナリオ:")
        for scenario, prob in prediction['scenarios'][:3]:
            print(f"      {scenario}: {prob:.1%}")
    
    # 可視化
    print("\n  [予測を可視化中...]")
    StockVisualizer.plot_prediction(stocks[0], predictor.predict(stocks[0], 30))
    
    # ============================================================
    # 3. ポートフォリオ最適化
    # ============================================================
    print("\n" + "─" * 70)
    print("  3. 量子ポートフォリオ最適化")
    print("─" * 70)
    
    optimizer = QuantumPortfolioOptimizer(stocks)
    weights, result = optimizer.optimize(n_iterations=1000)
    
    print("\n  最適配分:")
    for stock, weight in zip(stocks, weights):
        print(f"    {stock.symbol}: {weight:.1%}")
    
    print(f"\n  ポートフォリオ指標:")
    print(f"    期待リターン（年率）: {result['expected_return']:.1%}")
    print(f"    リスク（年率）: {result['risk']:.1%}")
    print(f"    シャープレシオ: {result['sharpe_ratio']:.2f}")
    
    # 可視化
    print("\n  [ポートフォリオを可視化中...]")
    StockVisualizer.plot_portfolio(optimizer, weights, result)
    
    # ============================================================
    # 4. センチメント分析
    # ============================================================
    print("\n" + "─" * 70)
    print("  4. 量子センチメント分析")
    print("─" * 70)
    
    analyzer = QuantumSentimentAnalyzer()
    analyzer.add_indicator('市場トレンド', 0.3, 1.0)
    analyzer.add_indicator('出来高', 0.2, 0.8)
    analyzer.add_indicator('ニュース', 0.5, 0.6)
    analyzer.add_indicator('VIX', -0.2, 0.5)
    
    result = analyzer.analyze()
    
    print(f"\n  センチメント分析結果:")
    print(f"    総合センチメント: {result['sentiment']:+.2f}")
    print(f"    信頼度: {result['confidence']:.1%}")
    print(f"    強気確率: {result['bullish_probability']:.1%}")
    print(f"    弱気確率: {result['bearish_probability']:.1%}")
    
    print("\n" + "=" * 70)
    print("  デモ完了！")
    print("=" * 70)


def interactive_mode():
    """対話モード"""
    print("\n" + "=" * 70)
    print("  量子株価予測システム - 対話モード")
    print("=" * 70)
    
    # サンプル株を生成
    stocks = {
        'AAPL': generate_sample_stock('AAPL', 100, 150, 0.3, 0.02),
        'GOOGL': generate_sample_stock('GOOGL', 100, 140, 0.1, 0.025),
        'MSFT': generate_sample_stock('MSFT', 100, 350, 0.2, 0.018),
        'AMZN': generate_sample_stock('AMZN', 100, 180, -0.1, 0.03),
        'TSLA': generate_sample_stock('TSLA', 100, 250, 0.0, 0.04),
    }
    
    predictor = QuantumStockPredictor()
    
    print("""
  コマンド:
    predict [銘柄] [日数]  - 株価予測 (例: predict AAPL 30)
    portfolio             - ポートフォリオ最適化
    sentiment             - センチメント分析
    list                  - 銘柄リスト
    help                  - ヘルプ
    quit                  - 終了
    """)
    
    while True:
        try:
            user_input = input("\n  株価予測> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', '終了']:
                print("  さようなら！")
                break
            
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd == 'list':
                print("\n  利用可能な銘柄:")
                for symbol, stock in stocks.items():
                    print(f"    {symbol}: ${stock.prices[-1]:.2f} (トレンド: {stock.trend:+.2f})")
            
            elif cmd == 'predict':
                symbol = parts[1].upper() if len(parts) > 1 else 'AAPL'
                days = int(parts[2]) if len(parts) > 2 else 30
                
                if symbol not in stocks:
                    print(f"  銘柄 {symbol} が見つかりません。'list' で確認してください。")
                    continue
                
                stock = stocks[symbol]
                prediction = predictor.predict(stock, days)
                
                print(f"\n  ┌{'─'*50}")
                print(f"  │ {symbol} {days}日予測")
                print(f"  ├{'─'*50}")
                print(f"  │ 現在価格:   ${prediction['current_price']:.2f}")
                print(f"  │ 予測価格:   ${prediction['predicted_price']:.2f}")
                print(f"  │ 予測範囲:   ${prediction['price_range'][0]:.2f} - ${prediction['price_range'][1]:.2f}")
                print(f"  │ 上昇確率:   {prediction['up_probability']:.1%}")
                print(f"  │ センチメント: {prediction['sentiment']:+.2f}")
                print(f"  ├{'─'*50}")
                print(f"  │ シナリオ:")
                for scenario, prob in prediction['scenarios'][:3]:
                    print(f"  │   {scenario}: {prob:.1%}")
                print(f"  └{'─'*50}")
                
                # 可視化
                StockVisualizer.plot_prediction(stock, prediction, days)
            
            elif cmd == 'portfolio':
                optimizer = QuantumPortfolioOptimizer(list(stocks.values()))
                weights, result = optimizer.optimize()
                
                print(f"\n  ┌{'─'*50}")
                print(f"  │ 最適ポートフォリオ")
                print(f"  ├{'─'*50}")
                for stock, weight in zip(stocks.values(), weights):
                    bar = '█' * int(weight * 20)
                    print(f"  │ {stock.symbol}: {weight:5.1%} {bar}")
                print(f"  ├{'─'*50}")
                print(f"  │ 期待リターン: {result['expected_return']:.1%}")
                print(f"  │ リスク:       {result['risk']:.1%}")
                print(f"  │ シャープ比:   {result['sharpe_ratio']:.2f}")
                print(f"  └{'─'*50}")
                
                StockVisualizer.plot_portfolio(optimizer, weights, result)
            
            elif cmd == 'sentiment':
                analyzer = QuantumSentimentAnalyzer()
                
                # 全銘柄のトレンドから計算
                avg_trend = np.mean([s.trend for s in stocks.values()])
                avg_vol = np.mean([s.volatility for s in stocks.values()])
                
                analyzer.add_indicator('市場トレンド', avg_trend, 1.0)
                analyzer.add_indicator('ボラティリティ', -avg_vol * 10, 0.5)
                
                result = analyzer.analyze()
                
                sentiment_emoji = '📈' if result['sentiment'] > 0.2 else '📉' if result['sentiment'] < -0.2 else '➡️'
                
                print(f"\n  {sentiment_emoji} 市場センチメント: {result['sentiment']:+.2f}")
                print(f"  信頼度: {result['confidence']:.1%}")
                print(f"  強気: {result['bullish_probability']:.1%} / 弱気: {result['bearish_probability']:.1%}")
            
            elif cmd == 'help':
                print("""
  ┌─────────────────────────────────────────────────────────┐
  │ コマンド一覧                                            │
  ├─────────────────────────────────────────────────────────┤
  │ predict [銘柄] [日数]  株価を量子予測                   │
  │   例: predict AAPL 30                                  │
  │                                                         │
  │ portfolio             量子ポートフォリオ最適化          │
  │                                                         │
  │ sentiment             市場センチメント分析              │
  │                                                         │
  │ list                  利用可能な銘柄一覧                │
  └─────────────────────────────────────────────────────────┘
                """)
            
            else:
                print("  不明なコマンド。'help' でヘルプを表示")
            
        except Exception as e:
            print(f"  エラー: {e}")
        except KeyboardInterrupt:
            print("\n  さようなら！")
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        demo()

