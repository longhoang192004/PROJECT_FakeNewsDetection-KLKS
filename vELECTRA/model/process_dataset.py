import pandas as pd
import os

# Define paths
base_dir = r"d:\KLKS\source\PROJECT_FakeNewsDetection-KLKS"
# Use the directory where the user pointed us to, assuming it has the source files
data_dir = os.path.join(base_dir, "vELECTRA", "data")
model_dir = os.path.join(base_dir, "model")

# Create model dir if not exists
os.makedirs(model_dir, exist_ok=True)

files = ["test.csv", "train.csv"]

for f in files:
    src_path = os.path.join(data_dir, f)
    dest_path = os.path.join(model_dir, f)
    
    print(f"Reading {src_path}...")
    try:
        # Read with utf-8 (standard for modern CSVs)
        # If this fails, we might need to try 'latin1' or 'cp1258', but 'utf-8' is most likely for generated datasets
        df = pd.read_csv(src_path, usecols=['post_message', 'label'], encoding='utf-8')
        
        # Check label content
        label_counts = df['label'].value_counts(dropna=False)
        print(f"Original Label Distribution in {f}:\n{label_counts}")
        
        print(f"Writing to {dest_path}...")
        # Save with utf-8-sig (BOM) so Excel opens it correctly with characters
        df.to_csv(dest_path, index=False, encoding='utf-8-sig')
        print(f"Finished {f}")
            
    except Exception as e:
        print(f"Error processing {f}: {e}")
