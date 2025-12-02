"""
量子ビット リアルタイムアニメーション

量子の重ね合わせ状態を視覚化：
- 0と1が確率的に揺らぐ
- ブロッホ球上で状態ベクトルが振動
- 測定まで状態が不確定
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
from pseudo_qubit import PseudoQubit
import random


class QuantumAnimation:
    def __init__(self):
        # 擬似量子ビット
        self.correlation = 0.0
        self.qubit = PseudoQubit(correlation=self.correlation)
        
        # 量子揺らぎのパラメータ
        self.fluctuation_amplitude = 0.08
        self.phase = 0
        self.measured = False
        self.measured_value = None
        self.measure_flash = 0
        
        # フィギュア設定
        self.fig = plt.figure(figsize=(16, 9), facecolor='#0a0a1a')
        self.fig.suptitle('PSEUDO-QUBIT: Quantum Superposition', 
                         fontsize=20, color='#00ffff', fontweight='bold', y=0.98)
        
        # レイアウト
        gs = self.fig.add_gridspec(3, 3, height_ratios=[3, 1, 0.5], 
                                   width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        # メイン表示エリア
        self.ax_qubit = self.fig.add_subplot(gs[0, 0])
        self.ax_wave = self.fig.add_subplot(gs[0, 1])
        self.ax_prob = self.fig.add_subplot(gs[0, 2])
        
        # 情報エリア
        self.ax_info = self.fig.add_subplot(gs[1, :])
        
        # コントロールエリア
        ax_slider = self.fig.add_axes([0.2, 0.08, 0.5, 0.02], facecolor='#1a1a3a')
        self.slider = Slider(ax_slider, 'r', -1.0, 1.0, valinit=0.0, 
                            valstep=0.01, color='#ff00ff')
        self.slider.label.set_color('white')
        self.slider.valtext.set_color('white')
        
        ax_measure = self.fig.add_axes([0.75, 0.06, 0.12, 0.04])
        self.btn_measure = Button(ax_measure, '⚡ MEASURE', 
                                  color='#1a1a3a', hovercolor='#ff00ff')
        self.btn_measure.label.set_color('#00ffff')
        self.btn_measure.label.set_fontweight('bold')
        
        ax_reset = self.fig.add_axes([0.88, 0.06, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'RESET', 
                               color='#1a1a3a', hovercolor='#333366')
        self.btn_reset.label.set_color('white')
        
        # イベント
        self.slider.on_changed(self.on_slider_change)
        self.btn_measure.on_clicked(self.on_measure)
        self.btn_reset.on_clicked(self.on_reset)
        
        # 波形データ
        self.wave_history = []
        self.max_history = 100
        
        # アニメーション開始
        self.ani = FuncAnimation(self.fig, self.update, frames=None,
                                interval=50, blit=False, cache_frame_data=False)
        
        plt.show()
    
    def on_slider_change(self, val):
        self.correlation = val
        self.qubit.correlation = val
        self.measured = False
        self.measured_value = None
    
    def on_measure(self, event):
        self.measured = True
        self.measured_value = self.qubit.measure()
        self.measure_flash = 10
    
    def on_reset(self, event):
        self.measured = False
        self.measured_value = None
        self.slider.set_val(0.0)
        self.wave_history = []
    
    def draw_quantum_state(self):
        """量子ビット状態の視覚化（0と1の重ね合わせ）"""
        ax = self.ax_qubit
        ax.clear()
        ax.set_facecolor('#0a0a1a')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 揺らぎを計算
        self.phase += 0.15
        fluctuation = np.sin(self.phase) * self.fluctuation_amplitude
        
        p0, p1 = self.qubit.probabilities
        
        if self.measured:
            # 測定後：確定状態
            if self.measured_value == 0:
                # |0⟩ に収縮
                color_0 = '#00ff88'
                color_1 = '#333333'
                alpha_0 = 1.0
                alpha_1 = 0.2
                size_0 = 0.8
                size_1 = 0.3
            else:
                # |1⟩ に収縮
                color_0 = '#333333'
                color_1 = '#ff6b6b'
                alpha_0 = 0.2
                alpha_1 = 1.0
                size_0 = 0.3
                size_1 = 0.8
                
            # フラッシュ効果
            if self.measure_flash > 0:
                flash_alpha = self.measure_flash / 10
                ax.add_patch(Circle((0, 0), 1.3, color='white', alpha=flash_alpha * 0.3))
                self.measure_flash -= 1
        else:
            # 重ね合わせ状態：揺らぐ
            color_0 = '#00ff88'
            color_1 = '#ff6b6b'
            
            # 確率に基づくサイズと透明度（揺らぎ付き）
            size_0 = 0.3 + p0 * 0.5 + fluctuation * p0
            size_1 = 0.3 + p1 * 0.5 - fluctuation * p1
            alpha_0 = 0.3 + p0 * 0.7 + fluctuation * 0.2
            alpha_1 = 0.3 + p1 * 0.7 - fluctuation * 0.2
        
        # |0⟩ 状態（上）
        circle_0 = Circle((0, 0.6), size_0, color=color_0, alpha=alpha_0)
        ax.add_patch(circle_0)
        ax.text(0, 0.6, '0', fontsize=40, ha='center', va='center', 
               color='white', fontweight='bold', alpha=alpha_0)
        
        # |1⟩ 状態（下）
        circle_1 = Circle((0, -0.6), size_1, color=color_1, alpha=alpha_1)
        ax.add_patch(circle_1)
        ax.text(0, -0.6, '1', fontsize=40, ha='center', va='center', 
               color='white', fontweight='bold', alpha=alpha_1)
        
        # 状態を結ぶ量子的な線（重ね合わせ表現）
        if not self.measured:
            for i in range(5):
                phase_offset = self.phase + i * 0.5
                x_offset = np.sin(phase_offset) * 0.1
                line_alpha = 0.1 + 0.1 * np.sin(phase_offset + i)
                ax.plot([x_offset, -x_offset], [0.6 - size_0, -0.6 + size_1], 
                       color='#ff00ff', alpha=line_alpha, linewidth=2)
        
        # タイトル
        if self.measured:
            state_text = f"COLLAPSED → |{self.measured_value}⟩"
            title_color = color_0 if self.measured_value == 0 else color_1
        else:
            state_text = "SUPERPOSITION"
            title_color = '#ff00ff'
        
        ax.set_title(state_text, fontsize=16, color=title_color, fontweight='bold', pad=10)
    
    def draw_wave_function(self):
        """波動関数の視覚化"""
        ax = self.ax_wave
        ax.clear()
        ax.set_facecolor('#0a0a1a')
        
        t = np.linspace(0, 4 * np.pi, 200)
        p0, p1 = self.qubit.probabilities
        
        if self.measured:
            # 測定後：波動関数の収縮
            if self.measured_value == 0:
                wave_0 = np.ones_like(t)
                wave_1 = np.zeros_like(t)
            else:
                wave_0 = np.zeros_like(t)
                wave_1 = np.ones_like(t)
        else:
            # 重ね合わせ：干渉パターン
            wave_0 = np.sqrt(p0) * np.cos(t + self.phase)
            wave_1 = np.sqrt(p1) * np.sin(t + self.phase * 1.3)
        
        # 波形を描画
        ax.fill_between(t, wave_0, alpha=0.3, color='#00ff88', label='|0⟩')
        ax.fill_between(t, wave_1, alpha=0.3, color='#ff6b6b', label='|1⟩')
        ax.plot(t, wave_0, color='#00ff88', linewidth=2)
        ax.plot(t, wave_1, color='#ff6b6b', linewidth=2)
        
        # 干渉パターン（重ね合わせ時のみ）
        if not self.measured:
            interference = wave_0 + wave_1
            ax.plot(t, interference, color='#ff00ff', linewidth=1, alpha=0.5, linestyle='--')
        
        ax.set_xlim(0, 4 * np.pi)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title('Wave Function |ψ⟩', fontsize=14, color='#00ffff', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#333366')
        ax.spines['left'].set_color('#333366')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', facecolor='#1a1a3a', edgecolor='#333366',
                 labelcolor='white')
    
    def draw_probability_bars(self):
        """確率分布バー"""
        ax = self.ax_prob
        ax.clear()
        ax.set_facecolor('#0a0a1a')
        
        p0, p1 = self.qubit.probabilities
        
        # 揺らぎ
        if not self.measured:
            fluctuation = np.sin(self.phase) * self.fluctuation_amplitude * 0.5
            display_p0 = p0 + fluctuation
            display_p1 = p1 - fluctuation
        else:
            display_p0 = 1.0 if self.measured_value == 0 else 0.0
            display_p1 = 1.0 if self.measured_value == 1 else 0.0
        
        # バー
        bars = ax.bar(['|0⟩', '|1⟩'], [display_p0, display_p1], 
                     color=['#00ff88', '#ff6b6b'], edgecolor='white', linewidth=2)
        
        # 確率値
        ax.text(0, display_p0 + 0.05, f'{display_p0:.3f}', ha='center', 
               fontsize=14, color='#00ff88', fontweight='bold')
        ax.text(1, display_p1 + 0.05, f'{display_p1:.3f}', ha='center', 
               fontsize=14, color='#ff6b6b', fontweight='bold')
        
        ax.set_ylim(0, 1.2)
        ax.axhline(y=0.5, color='#ffff00', linestyle='--', alpha=0.3)
        ax.set_title('Probability P(|ψ⟩)', fontsize=14, color='#00ffff', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#333366')
        ax.spines['left'].set_color('#333366')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def draw_info(self):
        """情報パネル"""
        ax = self.ax_info
        ax.clear()
        ax.set_facecolor('#0a0a1a')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        r = self.correlation
        theta = self.qubit.theta
        p0, p1 = self.qubit.probabilities
        alpha = self.qubit.state.alpha.real
        beta = self.qubit.state.beta.real
        
        # 量子状態の式
        state_eq = f"|ψ⟩ = {alpha:.4f}|0⟩ + {beta:.4f}|1⟩"
        
        # 情報テキスト
        info_text = (
            f"Correlation: r = {r:.2f}    |    "
            f"θ = {np.degrees(theta):.1f}°    |    "
            f"α = {alpha:.4f}    |    "
            f"β = {beta:.4f}    |    "
            f"P(0) = {p0:.4f}    |    "
            f"P(1) = {p1:.4f}"
        )
        
        ax.text(5, 0.7, state_eq, ha='center', fontsize=16, 
               color='#ff00ff', fontweight='bold', 
               fontfamily='monospace')
        
        ax.text(5, 0.3, info_text, ha='center', fontsize=11, 
               color='white', fontfamily='monospace')
        
        # 状態説明
        if self.measured:
            status = f"⚡ MEASURED: Collapsed to |{self.measured_value}⟩"
            status_color = '#00ff88' if self.measured_value == 0 else '#ff6b6b'
        else:
            if abs(r) > 0.9:
                status = "⬤ Almost pure state"
            elif abs(r) < 0.1:
                status = "◐ Maximum superposition"
            else:
                status = "◑ Partial superposition"
            status_color = '#ffff00'
        
        ax.text(5, 0.0, status, ha='center', fontsize=12, 
               color=status_color, fontweight='bold')
    
    def update(self, frame):
        """アニメーション更新"""
        self.draw_quantum_state()
        self.draw_wave_function()
        self.draw_probability_bars()
        self.draw_info()
        return []


if __name__ == "__main__":
    print("=" * 60)
    print("  PSEUDO-QUBIT: Quantum Superposition Animation")
    print("=" * 60)
    print("\n  Controls:")
    print("    • Slider: Adjust correlation coefficient r")
    print("    • MEASURE: Collapse the wave function")
    print("    • RESET: Return to superposition state")
    print("\n  Observation:")
    print("    • Watch 0 and 1 fluctuate in superposition")
    print("    • Click MEASURE to collapse to definite state")
    print("=" * 60)
    
    app = QuantumAnimation()

