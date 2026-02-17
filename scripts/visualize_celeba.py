
import os
import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

def main():
    # Path to the specific valid dataset directory containing arrow files and dataset_info.json
    dataset_path = r"e:\cmu-e\courses\diffusion\homeworks\cmu-10799-diffusion\data\celeba-subset\electronickale___cmu-10799-celeba64-subset\default\0.0.0\cea8d2312303971a09528db035498464cbb01e37"
    
    print(f"Loading dataset from: {dataset_path}")
    try:
        ds = datasets.load_from_disk(dataset_path)
    except Exception as e:
        print(f"Failed to load with load_from_disk: {e}")
        # Try finding the arrow file directly
        arrow_files = [f for f in os.listdir(dataset_path) if f.endswith('.arrow')]
        if arrow_files:
            arrow_file = os.path.join(dataset_path, arrow_files[0])
            print(f"Found arrow file: {arrow_file}. Loading with Dataset.from_file...")
            ds = datasets.Dataset.from_file(arrow_file)
        else:
            print("No arrow file found. Cannot load dataset.")
            return

    print(f"Dataset loaded. Info: {ds}")
    
    # Check if it is a Dataset or DatasetDict
    if isinstance(ds, datasets.DatasetDict):
        print("Dataset is a path dict, keys:", ds.keys())
        # Usually train is the key
        if 'train' in ds:
            data = ds['train']
        else:
            data = ds[list(ds.keys())[0]]
    else:
        data = ds

    print(f"Number of samples: {len(data)}")
    
    # Pick 16 random indices
    indices = random.sample(range(len(data)), 16)
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i, idx in enumerate(indices):
        item = data[idx]
        # The image column is usually 'image' or 'file_name' mapped to image. 
        # Let's verify features if we could, but 'image' is standard for generic image datasets.
        # If 'image' key exists and is a PIL object or bytes.
        if 'image' in item:
            img = item['image']
        else:
            print(f"Row {idx} keys: {item.keys()}")
            # Simple fallback if key is different, though unlikley for "celeba"
            img = list(item.values())[0] # Risky but okay for debugging
            
        row = i // 4
        col = i % 4
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        
    plt.tight_layout()
    output_path = os.path.join("outputs", "celeba_grid_4x4.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved grid to {output_path}")

if __name__ == "__main__":
    main()
