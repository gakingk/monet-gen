import os
import json
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from sklearn.model_selection import train_test_split

dataset_dir = "dataset"  # ou o caminho correto para as imagens
metadata_path = os.path.join(dataset_dir, "metadata.json")

# Carrega o dicionário: {filename: caption}
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Constrói lista de exemplos
data = []
for filename, caption in metadata.items():
    image_path = os.path.join(dataset_dir, filename)
    if os.path.isfile(image_path):
        data.append({"image": image_path, "text": caption})

# Separa splits
train_val, test = train_test_split(data, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)

# Cria datasets Hugging Face
features = Features({"image": HFImage(), "text": Value("string")})
dataset = DatasetDict({
    "train": Dataset.from_list(train, features=features),
    "validation": Dataset.from_list(val, features=features),
    "test": Dataset.from_list(test, features=features),
})

# Salva o dataset processado
dataset.save_to_disk("processed_dataset")
