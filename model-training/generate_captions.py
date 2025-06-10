import os
from PIL import Image
from tqdm import tqdm
import re
from transformers import BlipProcessor, BlipForConditionalGeneration

image_dir = "dataset"
output_path = "dataset/captions.txt"

# Carregar modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Limpeza estruturada da legenda
def clean_caption(caption):
    caption = caption.strip()

    # Remove padrões comuns como "A painting of", "An illustration of", etc.
    pattern = re.compile(r"^(a|an)\s+(painting|drawing|artwork|illustration|sketch)\s+of\s+", re.IGNORECASE)
    cleaned = pattern.sub("", caption)

    # Capitaliza a primeira letra e remove espaços extras
    cleaned = cleaned.strip().capitalize()

    # Se sobrou muito pouco texto, volta ao original
    if len(cleaned.split()) < 3:
        return caption
    return cleaned

# Geração em lote
with open(output_path, "w") as f:
    for filename in tqdm(os.listdir(image_dir)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(image_dir, filename)
            try:
                raw_caption = generate_caption(path)
                cleaned_caption = clean_caption(raw_caption)
                f.write(f"{filename}|{cleaned_caption}\n")
            except Exception as e:
                print(f"Erro com {filename}: {e}")
