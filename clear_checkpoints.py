import modal

app = modal.App("clear-volume-app")
vol = modal.Volume.from_name("neuroq-checkpoints")

@app.function(volumes={"/mnt/checkpoints": vol})
def clear_volume():
    import os
    
    deleted = 0
    checkpoint_dir = "/mnt/checkpoints"
    
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, item)
            if os.path.isfile(path) and path.endswith(".pt"):
                print(f"Deleting: {item}")
                os.remove(path)
                deleted += 1
                
    return deleted

@app.local_entrypoint()
def main():
    print("Clearing old .pt checkpoint files from neuroq-checkpoints...")
    count = clear_volume.remote()
    print(f"Successfully deleted {count} checkpoint files.")
