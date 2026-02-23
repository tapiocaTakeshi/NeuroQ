import modal
import sys

# Connect to the deployed app
stub = modal.App.lookup("neuroq-api")

def test_neuroq_api(prompt="What is quantum entanglement?", model_size="micro"):
    print("=" * 60)
    print(f"🚀 Testing NeuroQ API (Model: {model_size})")
    try:
        print(f"User: {prompt}\n")
        print("Assistant (NeuroQ): ", end="", flush=True)

        # Call the remote method via Cls.from_name
        inference_cls = modal.Cls.from_name("neuroq-api", "NeuroQInference")
        inference = inference_cls()
        
        # Call the generate method
        result = inference.generate.remote(
            prompt=prompt,
            max_length=150,
            temperature=0.7,
            model_size=model_size
        )
        
        print(f"{result.get('generated')}")
        print("-" * 60)
        
        if result.get("rag_used"):
            print("🔍 RAG was used for this response.")
            print("Sources:", result.get("rag_sources"))
            
    except Exception as e:
        print(f"\n❌ Error calling Modal API: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Can you explain what quantum computing is in simple terms?")
    parser.add_argument("--model", type=str, default="micro", choices=["micro", "small", "large"])
    args = parser.parse_args()
    
    test_neuroq_api(args.prompt, args.model)
