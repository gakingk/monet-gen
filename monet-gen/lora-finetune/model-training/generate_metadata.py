import json

input_path = "dataset/captions.txt"
output_path = "dataset/metadata.json"

entries = []
with open(input_path, "r") as f:
    for line in f:
        filename, caption = line.strip().split("|", 1)
        entries.append({"file_name": filename, "text": caption})

with open(output_path, "w") as f:
    json.dump(entries, f, indent=2)
