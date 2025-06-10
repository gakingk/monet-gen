import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
from datetime import datetime

# Carrega modelo base
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-4", 
    torch_dtype=torch.float16
).to("cuda")

# Diretórios
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Carrega prompts
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Geração
for i, prompt in enumerate(prompts):
    image = pipe(prompt, guidance_scale=8.5).images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image.save(output_dir / f"{i:02d}_monet_{timestamp}.png")
    print(f"✔ Imagem gerada para prompt {i+1}: {prompt}")
