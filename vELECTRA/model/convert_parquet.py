import pandas as pd
import os

# Define paths
base_dir = r"d:\KLKS\source\PROJECT_FakeNewsDetection-KLKS"
velectra_dir = os.path.join(base_dir, "vELECTRA")
data_dir = os.path.join(base_dir, "data")

# Create data dir if not exists
os.makedirs(data_dir, exist_ok=True)

files = [
    "test-00000-of-00001.parquet",
    "train-00000-of-00001.parquet"
]

for f in files:
    src_path = os.path.join(velectra_dir, f)
    # New filename: separate name and extension, change ext to csv
    name, _ = os.path.splitext(f)
    dest_path = os.path.join(data_dir, name + ".csv")
    
    print(f"Reading {src_path}...")
    try:
        df = pd.read_parquet(src_path)
        print(f"Writing to {dest_path}...")
        df.to_csv(dest_path, index=False)
        print(f"Finished {f}")
    except Exception as e:
        print(f"Error converting {f}: {e}")
