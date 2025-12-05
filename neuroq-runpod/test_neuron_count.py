#!/usr/bin/env python3
"""
ニューロン数が実際に変わっているかを検証するテストスクリプト
"""

import torch
from neuroq_model import (
    NeuroQModel, 
    NeuroQConfig, 
    BrainQuantumLayer,
    create_neuroq_brain,
    create_neuroq_layered,
)


def test_brain_quantum_layer():
    """BrainQuantumLayerのニューロン数をテスト"""
    print("=" * 60)
    print("🧪 BrainQuantumLayer ニューロン数テスト")
    print("=" * 60)
    
    test_cases = [32, 64, 100, 256, 512]
    
    for num_neurons in test_cases:
        layer = BrainQuantumLayer(
            num_neurons=num_neurons,
            input_dim=128,
            output_dim=128,
            connection_density=0.25,
            lambda_entangle=0.35
        )
        
        # 各パラメータの形状を確認
        theta_shape = layer.theta.shape
        weights_shape = layer.weights.shape
        J_shape = layer.J.shape
        mask_shape = layer.connection_mask.shape
        
        # 接続数をカウント
        num_connections = layer.connection_mask.sum().item()
        expected_connections = num_neurons * num_neurons * 0.25  # 期待される接続数（概算）
        
        print(f"\n📊 num_neurons = {num_neurons}")
        print(f"   theta.shape:           {theta_shape} (期待: [{num_neurons}])")
        print(f"   weights.shape:         {weights_shape} (期待: [{num_neurons}, {num_neurons}])")
        print(f"   J.shape:               {J_shape} (期待: [{num_neurons}, {num_neurons}])")
        print(f"   connection_mask.shape: {mask_shape} (期待: [{num_neurons}, {num_neurons}])")
        print(f"   実際の接続数:          {int(num_connections)} (期待: ~{int(expected_connections)})")
        
        # 検証
        assert theta_shape[0] == num_neurons, f"theta shape mismatch: {theta_shape[0]} != {num_neurons}"
        assert weights_shape == (num_neurons, num_neurons), f"weights shape mismatch"
        assert J_shape == (num_neurons, num_neurons), f"J shape mismatch"
        
        print(f"   ✅ 検証OK!")


def test_brain_model_neurons():
    """Brainモデル全体のニューロン数をテスト"""
    print("\n" + "=" * 60)
    print("🧪 NeuroQ Brain Model ニューロン数テスト")
    print("=" * 60)
    
    test_cases = [32, 64, 128, 256]
    
    for num_neurons in test_cases:
        config = NeuroQConfig(
            mode='brain',
            vocab_size=1000,
            embed_dim=64,
            num_neurons=num_neurons,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
            dropout=0.1,
            connection_density=0.25,
            lambda_entangle=0.35,
        )
        
        model = NeuroQModel(config)
        
        print(f"\n📊 config.num_neurons = {num_neurons}")
        print(f"   モデルパラメータ数: {model.num_params:,}")
        
        # 各ブロックのFFNレイヤーを確認
        for i, block in enumerate(model.model.blocks):
            ffn_neurons = block.ffn.num_neurons
            attn_neurons = block.attention.brain_layer.num_neurons
            
            # FFNは num_neurons * 2 で作成されている
            expected_ffn = num_neurons * 2
            
            print(f"   Block {i}:")
            print(f"     - FFN neurons:  {ffn_neurons} (期待: {expected_ffn})")
            print(f"     - Attn neurons: {attn_neurons} (期待: {num_neurons})")
            
            assert ffn_neurons == expected_ffn, f"FFN neurons mismatch in block {i}"
            assert attn_neurons == num_neurons, f"Attention neurons mismatch in block {i}"
        
        print(f"   ✅ 検証OK!")


def test_parameter_count_changes():
    """ニューロン数によってパラメータ数が変わることを確認"""
    print("\n" + "=" * 60)
    print("🧪 パラメータ数変化テスト")
    print("=" * 60)
    
    param_counts = {}
    
    for num_neurons in [32, 64, 128, 256]:
        model = create_neuroq_brain(
            vocab_size=1000,
            embed_dim=64,
            num_neurons=num_neurons,
            num_layers=2,
        )
        param_counts[num_neurons] = model.num_params
    
    print("\n📊 ニューロン数とパラメータ数の関係:")
    prev_count = 0
    for neurons, count in sorted(param_counts.items()):
        diff = f"(+{count - prev_count:,})" if prev_count > 0 else ""
        print(f"   num_neurons={neurons:4d}: パラメータ数 = {count:>10,} {diff}")
        prev_count = count
    
    # パラメータ数が増加していることを確認
    counts = list(param_counts.values())
    for i in range(1, len(counts)):
        assert counts[i] > counts[i-1], "パラメータ数が増加していない！"
    
    print("\n   ✅ ニューロン数に応じてパラメータ数が正しく増加しています！")


def test_forward_pass():
    """異なるニューロン数での順伝播テスト"""
    print("\n" + "=" * 60)
    print("🧪 順伝播テスト（異なるニューロン数）")
    print("=" * 60)
    
    for num_neurons in [32, 64, 128]:
        model = create_neuroq_brain(
            vocab_size=1000,
            embed_dim=64,
            num_neurons=num_neurons,
            num_layers=2,
        )
        model.eval()
        
        # テスト入力
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"\n📊 num_neurons = {num_neurons}")
        print(f"   入力形状:  {input_ids.shape}")
        print(f"   出力形状:  {output.shape}")
        print(f"   期待形状:  ({batch_size}, {seq_len}, 1000)")
        
        assert output.shape == (batch_size, seq_len, 1000), "出力形状が不正"
        print(f"   ✅ 順伝播OK!")


def test_quantum_stats():
    """量子統計情報の確認"""
    print("\n" + "=" * 60)
    print("🧪 量子統計情報テスト")
    print("=" * 60)
    
    for num_neurons in [32, 100, 200]:
        model = create_neuroq_brain(
            vocab_size=1000,
            embed_dim=64,
            num_neurons=num_neurons,
            num_layers=2,
        )
        
        quantum_info = model.get_quantum_info()
        
        print(f"\n📊 num_neurons = {num_neurons}")
        for info in quantum_info:
            print(f"   Block {info['block']}:")
            print(f"     - Mode: {info['mode']}")
            print(f"     - Attn r (相関係数): {info['attn_r']:.4f}")
            print(f"     - Attn T (温度): {info['attn_T']:.4f}")
            print(f"     - Attn λ (もつれ強度): {info['attn_lambda']:.4f}")
            print(f"     - FFN connections: {int(info['connections'])}")
        
        # 接続数がニューロン数に依存していることを確認
        # FFNは num_neurons * 2 のニューロンを持つ
        expected_max_connections = (num_neurons * 2) ** 2
        actual_connections = quantum_info[0]['connections']
        
        print(f"   接続数: {int(actual_connections)} / 最大 {expected_max_connections}")
        assert actual_connections > 0, "接続数が0"
        assert actual_connections < expected_max_connections, "接続数が多すぎる"
        
        print(f"   ✅ 量子統計OK!")


if __name__ == "__main__":
    print("\n🔬 NeuroQ ニューロン数検証テスト\n")
    
    test_brain_quantum_layer()
    test_brain_model_neurons()
    test_parameter_count_changes()
    test_forward_pass()
    test_quantum_stats()
    
    print("\n" + "=" * 60)
    print("🎉 すべてのテストが成功しました！")
    print("   ニューロン数は正しく反映されています。")
    print("=" * 60)
