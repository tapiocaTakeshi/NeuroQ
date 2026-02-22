"""マルチモーダル学習後の生成テスト"""
from train_multimodal import MultiModalDataset
from neuroquantum_brain import NeuroQuantumBrain
from neuroq_pretrained import NeuroQTokenizer

def test_multimodal():
    tokenizer = NeuroQTokenizer.from_pretrained('neuroq_tokenizer_ja')
    model = NeuroQuantumBrain.from_pretrained('checkpoints/neuroq_multimodal_final.pt')
    
    prompts = {
        'llm': '生成AIの学習データについて',
        'code': 'Pythonで量子ニューラルネットワークを実装',
        'image': '量子回路の美しいビジュアライゼーション'
    }
    
    for dtype, prompt in prompts.items():
        inputs = tokenizer.encode(prompt, max_length=512)
        outputs = model.generate(inputs, max_new_tokens=100, temperature=0.7)
        response = tokenizer.decode(outputs)
        print(f"\n{dtype.upper()}生成:")
        print(response)

if __name__ == "__main__":
    test_multimodal()
