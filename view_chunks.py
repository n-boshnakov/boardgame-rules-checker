
import pickle
import json
import pandas as pd
import sys
import os
from datetime import datetime

# Usage: python view_chunks.py [chunks_path] [num]
chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/chunks.pkl"
num = int(sys.argv[2]) if len(sys.argv) > 2 else 5

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks from {chunks_path}\n")

# Write all chunks to a new JSON file with the same base name and current date
base = os.path.splitext(os.path.basename(chunks_path))[0]
date_str = datetime.now().strftime("%Y-%m-%d")
json_name = f"{base}_{date_str}.json"
json_path = os.path.join(os.path.dirname(chunks_path), json_name)
with open(json_path, "w", encoding="utf-8") as jf:
    json.dump(chunks, jf, indent=2, ensure_ascii=False)
print(f"All chunks written to {json_path}\n")

# Print as table
try:
    df = pd.DataFrame(chunks)
    print(df.head(num).to_string(index=False))
except Exception as e:
    print("Could not display as table, printing as JSON:")
    print(json.dumps(chunks[:num], indent=2, ensure_ascii=False))
