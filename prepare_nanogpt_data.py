import os
import numpy as np
import tiktoken  # pip install tiktoken

# === Settings ===
enc = tiktoken.get_encoding("gpt2")
data_dir = "data_gpt2"
output_dir = "data_nanogpt"
os.makedirs(output_dir, exist_ok=True)

def encode_file(path, output_bin_path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
        print(f"\nLoaded {len(data)} characters from {path}")

    tokens = enc.encode(data)
    print(f"Tokenized to {len(tokens)} tokens.")

    # Save as .bin
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(output_bin_path)
    print(f"Saved binary to {output_bin_path}")

# === Encode both train and val ===
encode_file(os.path.join(data_dir, "train.txt"), os.path.join(output_dir, "train.bin"))
encode_file(os.path.join(data_dir, "val.txt"), os.path.join(output_dir, "val.bin"))
