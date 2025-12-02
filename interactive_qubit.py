"""
インタラクティブな擬似量子ビットビジュアライザー

スライダーで相関係数をリアルタイム調整し、
ブロッホ球と確率分布が即座に更新される
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from pseudo_qubit import PseudoQubit


class InteractiveQubitVisualizer:
    def __init__(self):
        # 初期相関係数
        self.correlation = 0.0
        self.qubit = PseudoQubit(correlation=self.correlation)
        
        # フィギュアのセットアップ
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # ブロッホ球 (3D)
        self.ax_bloch = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_bloch.set_facecolor('#16213e')
        
        # 確率分布
        self.ax_prob = self.fig.add_subplot(1, 2, 2)
        self.ax_prob.set_facecolor('#16213e')
        
        # スライダー用のスペース
        plt.subplots_adjust(bottom=0.25)
        
        # スライダー
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='#0f3460')
        self.slider = Slider(
            ax_slider, 
            'r', 
            -1.0, 1.0, 
            valinit=self.correlation,
            valstep=0.01,
            color='#e94560'
        )
        self.slider.label.set_color('white')
        self.slider.valtext.set_color('white')
        
        # リセットボタン
        ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset', color='#0f3460', hovercolor='#e94560')
        self.button_reset.label.set_color('white')
        
        # 測定ボタン
        ax_measure = plt.axes([0.65, 0.025, 0.12, 0.04])
        self.button_measure = Button(ax_measure, 'Measure', color='#0f3460', hovercolor='#e94560')
        self.button_measure.label.set_color('white')
        
        # 測定結果表示用テキスト
        self.measure_text = self.fig.text(0.5, 0.02, '', ha='center', fontsize=12, color='#00ff88')
        
        # イベント接続
        self.slider.on_changed(self.update)
        self.button_reset.on_clicked(self.reset)
        self.button_measure.on_clicked(self.measure)
        
        # 初期描画
        self.draw_bloch_sphere()
        self.draw_probability_bars()
        
        plt.show()
    
    def draw_bloch_sphere(self):
        """ブロッホ球を描画"""
        self.ax_bloch.clear()
        self.ax_bloch.set_facecolor('#16213e')
        
        # ブロッホ球のワイヤーフレーム
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax_bloch.plot_wireframe(x, y, z, alpha=0.1, color='#4a9fff')
        
        # 赤道円
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax_bloch.plot(np.cos(theta), np.sin(theta), 0, 'c-', alpha=0.3)
        
        # 軸を描画
        axis_len = 1.3
        self.ax_bloch.plot([-axis_len, axis_len], [0, 0], [0, 0], 'w-', alpha=0.3, linewidth=1)
        self.ax_bloch.plot([0, 0], [-axis_len, axis_len], [0, 0], 'w-', alpha=0.3, linewidth=1)
        self.ax_bloch.plot([0, 0], [0, 0], [-axis_len, axis_len], 'w-', alpha=0.3, linewidth=1)
        
        # |0⟩ と |1⟩ のラベル
        self.ax_bloch.text(0, 0, 1.4, '|0⟩', fontsize=14, ha='center', color='#00ff88', fontweight='bold')
        self.ax_bloch.text(0, 0, -1.4, '|1⟩', fontsize=14, ha='center', color='#ff6b6b', fontweight='bold')
        
        # 量子状態ベクトル
        coords = self.qubit.to_bloch_coordinates()
        
        # ベクトルを描画
        self.ax_bloch.quiver(0, 0, 0, coords[0], coords[1], coords[2],
                           color='#e94560', arrow_length_ratio=0.15, linewidth=3)
        
        # 状態点
        self.ax_bloch.scatter([coords[0]], [coords[1]], [coords[2]], 
                             color='#e94560', s=150, edgecolors='white', linewidths=2, zorder=5)
        
        # 設定
        self.ax_bloch.set_xlim([-1.5, 1.5])
        self.ax_bloch.set_ylim([-1.5, 1.5])
        self.ax_bloch.set_zlim([-1.5, 1.5])
        self.ax_bloch.set_xlabel('X', color='white')
        self.ax_bloch.set_ylabel('Y', color='white')
        self.ax_bloch.set_zlabel('Z', color='white')
        
        # タイトル
        title = f'Bloch Sphere\nr = {self.correlation:.2f}, θ = {np.degrees(self.qubit.theta):.1f}°'
        self.ax_bloch.set_title(title, color='white', fontsize=12, pad=10)
        
        # グリッドと軸の色
        self.ax_bloch.tick_params(colors='white')
        self.ax_bloch.xaxis.pane.fill = False
        self.ax_bloch.yaxis.pane.fill = False
        self.ax_bloch.zaxis.pane.fill = False
        
    def draw_probability_bars(self):
        """確率分布を棒グラフで描画"""
        self.ax_prob.clear()
        self.ax_prob.set_facecolor('#16213e')
        
        probs = self.qubit.probabilities
        states = ['|0⟩', '|1⟩']
        colors = ['#00ff88', '#ff6b6b']
        
        bars = self.ax_prob.bar(states, probs, color=colors, edgecolor='white', linewidth=2, width=0.5)
        
        # 確率値を表示
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            self.ax_prob.text(bar.get_x() + bar.get_width()/2, height + 0.03,
                             f'{prob:.4f}', ha='center', fontsize=14, color='white', fontweight='bold')
        
        # 振幅も表示
        alpha_str = f'α = {self.qubit.state.alpha.real:.4f}'
        beta_str = f'β = {self.qubit.state.beta.real:.4f}'
        self.ax_prob.text(0, probs[0] - 0.1, alpha_str, ha='center', fontsize=10, color='white')
        self.ax_prob.text(1, probs[1] - 0.1, beta_str, ha='center', fontsize=10, color='white')
        
        # 設定
        self.ax_prob.set_ylim([0, 1.15])
        self.ax_prob.set_ylabel('Probability', color='white', fontsize=12)
        self.ax_prob.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='Equal superposition')
        
        # タイトル
        self.ax_prob.set_title(f'Measurement Probability\n|ψ⟩ = {self.qubit.state.alpha.real:.3f}|0⟩ + {self.qubit.state.beta.real:.3f}|1⟩', 
                              color='white', fontsize=12)
        
        # 軸の色
        self.ax_prob.tick_params(colors='white')
        self.ax_prob.spines['bottom'].set_color('white')
        self.ax_prob.spines['top'].set_color('#16213e')
        self.ax_prob.spines['left'].set_color('white')
        self.ax_prob.spines['right'].set_color('#16213e')
        
        # 相関係数の状態を表示
        if self.correlation > 0.9:
            state_desc = "Strong positive correlation → Almost |0⟩"
        elif self.correlation > 0.1:
            state_desc = "Positive correlation → Biased to |0⟩"
        elif self.correlation > -0.1:
            state_desc = "No correlation → Superposition"
        elif self.correlation > -0.9:
            state_desc = "Negative correlation → Biased to |1⟩"
        else:
            state_desc = "Strong negative correlation → Almost |1⟩"
            
        self.ax_prob.text(0.5, -0.15, state_desc, ha='center', transform=self.ax_prob.transAxes,
                         fontsize=11, color='#ffd93d', style='italic')
    
    def update(self, val):
        """スライダー更新時のコールバック"""
        self.correlation = self.slider.val
        self.qubit.correlation = self.correlation
        self.measure_text.set_text('')
        
        self.draw_bloch_sphere()
        self.draw_probability_bars()
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """リセットボタンのコールバック"""
        self.slider.reset()
        self.measure_text.set_text('')
    
    def measure(self, event):
        """測定ボタンのコールバック"""
        result = self.qubit.measure()
        state_str = '|0⟩' if result == 0 else '|1⟩'
        color = '#00ff88' if result == 0 else '#ff6b6b'
        self.measure_text.set_text(f'Measurement Result: {state_str}')
        self.measure_text.set_color(color)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    print("=" * 50)
    print("Interactive Pseudo-Qubit Visualizer")
    print("=" * 50)
    print("Controls:")
    print("  - Slider: Adjust correlation coefficient r")
    print("  - Measure: Simulate quantum measurement")
    print("  - Reset: Return to r = 0")
    print("=" * 50)
    
    viz = InteractiveQubitVisualizer()

