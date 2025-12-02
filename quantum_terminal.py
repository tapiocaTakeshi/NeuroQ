"""
ターミナル版 量子ビット リアルタイムアニメーション

ASCII アートで量子の重ね合わせを表現
"""

import os
import sys
import time
import math
import random
from pseudo_qubit import PseudoQubit


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def create_progress_bar(value, width=30, fill_char='█', empty_char='░'):
    """プログレスバーを生成"""
    filled = int(width * value)
    return fill_char * filled + empty_char * (width - filled)


def quantum_animation(correlation=0.0, duration=30):
    """
    ターミナルで量子ビットのアニメーションを表示
    
    Args:
        correlation: 相関係数 r ∈ [-1, 1]
        duration: アニメーション時間（秒）
    """
    qubit = PseudoQubit(correlation=correlation)
    p0, p1 = qubit.probabilities
    
    phase = 0
    fluctuation_amp = 0.08
    measured = False
    measured_value = None
    
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("  PSEUDO-QUBIT: Quantum Superposition (Terminal)")
    print("=" * 60)
    print(f"  Correlation r = {correlation:.2f}")
    print(f"  Press Ctrl+C to measure!")
    print("=" * 60 + "\n")
    time.sleep(2)
    
    try:
        while time.time() - start_time < duration:
            clear_screen()
            
            # 揺らぎ計算
            phase += 0.3
            fluctuation = math.sin(phase) * fluctuation_amp
            
            # 表示用確率（揺らぎ付き）
            display_p0 = max(0, min(1, p0 + fluctuation))
            display_p1 = max(0, min(1, p1 - fluctuation))
            
            # ASCIIアート
            print("\n" + "═" * 60)
            print("  ╔═══════════════════════════════════════════════════════╗")
            print("  ║        PSEUDO-QUBIT: QUANTUM SUPERPOSITION            ║")
            print("  ╚═══════════════════════════════════════════════════════╝")
            print("═" * 60)
            
            # 量子状態表示
            alpha = qubit.state.alpha.real
            beta = qubit.state.beta.real
            print(f"\n  |ψ⟩ = {alpha:.4f}|0⟩ + {beta:.4f}|1⟩")
            print(f"  θ = {math.degrees(qubit.theta):.1f}°, r = {correlation:.2f}")
            
            # 確率バー
            print("\n  ┌─────────────────────────────────────────────────────┐")
            bar_0 = create_progress_bar(display_p0)
            bar_1 = create_progress_bar(display_p1)
            print(f"  │ |0⟩  {bar_0} {display_p0:.3f} │")
            print(f"  │ |1⟩  {bar_1} {display_p1:.3f} │")
            print("  └─────────────────────────────────────────────────────┘")
            
            # 視覚的な量子状態
            size_0 = int(5 + display_p0 * 10)
            size_1 = int(5 + display_p1 * 10)
            
            print("\n  SUPERPOSITION STATE:")
            print("  " + "─" * 40)
            
            # |0⟩ の円
            intensity_0 = int(display_p0 * 9)
            chars_0 = " ·∘○◯●◉⬤⬛"
            state_0_char = chars_0[min(intensity_0, len(chars_0)-1)]
            
            print(f"  │0⟩:  {state_0_char * size_0}")
            
            # 重ね合わせの線
            wave_char = "～" if int(phase * 2) % 2 == 0 else "∿"
            print(f"       {wave_char * 15}  ← superposition")
            
            # |1⟩ の円
            intensity_1 = int(display_p1 * 9)
            chars_1 = " ·∘○◯●◉⬤⬛"
            state_1_char = chars_1[min(intensity_1, len(chars_1)-1)]
            
            print(f"  │1⟩:  {state_1_char * size_1}")
            print("  " + "─" * 40)
            
            # 波動関数のASCII表現
            print("\n  WAVE FUNCTION:")
            wave_width = 50
            wave_height = 5
            
            for y in range(wave_height, -wave_height-1, -1):
                line = "  │"
                for x in range(wave_width):
                    t = x / 5 + phase
                    wave_0 = math.sqrt(p0) * math.cos(t) * wave_height
                    wave_1 = math.sqrt(p1) * math.sin(t * 1.3) * wave_height
                    
                    if abs(y - wave_0) < 0.5:
                        line += "○"
                    elif abs(y - wave_1) < 0.5:
                        line += "●"
                    elif abs(y - (wave_0 + wave_1)) < 0.5:
                        line += "◆"
                    elif y == 0:
                        line += "─"
                    else:
                        line += " "
                line += "│"
                print(line)
            
            print("  " + "═" * 54)
            print("  ○ = |0⟩ amplitude    ● = |1⟩ amplitude    ◆ = interference")
            
            # ステータス
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            print(f"\n  ⏱  Time: {elapsed:.1f}s / {duration}s")
            print("  ⚡ Press Ctrl+C to MEASURE and collapse the wave function!")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        # 測定！
        clear_screen()
        measured_value = qubit.measure()
        
        print("\n" + "═" * 60)
        print("  ╔═══════════════════════════════════════════════════════╗")
        print("  ║              ⚡ MEASUREMENT PERFORMED ⚡               ║")
        print("  ╚═══════════════════════════════════════════════════════╝")
        print("═" * 60)
        
        print("\n  WAVE FUNCTION COLLAPSED!")
        print("  " + "─" * 40)
        
        if measured_value == 0:
            print("\n  ████████████████████████████████████")
            print("  ██                                ██")
            print("  ██    RESULT:  │ 0 ⟩              ██")
            print("  ██                                ██")
            print("  ████████████████████████████████████")
            print(f"\n  Probability was: {p0:.4f}")
        else:
            print("\n  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
            print("  ▓▓                                ▓▓")
            print("  ▓▓    RESULT:  │ 1 ⟩              ▓▓")
            print("  ▓▓                                ▓▓")
            print("  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
            print(f"\n  Probability was: {p1:.4f}")
        
        print("\n  " + "═" * 40)
        print(f"  Initial state: r = {correlation:.2f}")
        print(f"  |ψ⟩ = {alpha:.4f}|0⟩ + {beta:.4f}|1⟩")
        print("  " + "═" * 40)


def main():
    print("\n" + "=" * 60)
    print("  PSEUDO-QUBIT: Quantum Superposition (Terminal)")
    print("=" * 60)
    
    # 相関係数の入力
    while True:
        try:
            r_input = input("\n  Enter correlation coefficient r [-1, 1] (default=0): ").strip()
            if r_input == "":
                r = 0.0
            else:
                r = float(r_input)
            
            if -1.0 <= r <= 1.0:
                break
            else:
                print("  ⚠ Please enter a value between -1 and 1")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    # アニメーション時間
    while True:
        try:
            t_input = input("  Enter animation duration in seconds (default=30): ").strip()
            if t_input == "":
                duration = 30
            else:
                duration = int(t_input)
            
            if duration > 0:
                break
            else:
                print("  ⚠ Please enter a positive number")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    print(f"\n  Starting animation with r = {r}, duration = {duration}s")
    print("  Get ready...")
    time.sleep(2)
    
    quantum_animation(correlation=r, duration=duration)


if __name__ == "__main__":
    main()

