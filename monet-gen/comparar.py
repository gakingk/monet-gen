import os
import torch
from diffusers import StableDiffusionPipeline
from torch import manual_seed
from pathlib import Path

# Lista de pares de prompts
prompts = [
    ("A calm lake at sunset", "A calm lake at sunset, in the style of Claude Monet, with soft brushstrokes and warm impressionist tones"),
    ("A field of blooming wildflowers under a cloudy sky", "A field of blooming wildflowers under a cloudy sky, painted in the impressionist style of Monet"),
    ("A small wooden bridge over a quiet river", "A small wooden bridge over a quiet river, Claude Monet style, with light reflections and vibrant pastels"),
    ("A seaside village with boats on the shore", "A seaside village with boats on the shore, impressionist painting, in Claude Monet's style"),
    ("A rainy street in a 19th-century European town", "A rainy 19th-century European street, painted with dabs of color and soft outlines, in Claude Monet’s style"),
    ("A lush garden with rose arches and green hedges", "A lush garden with rose arches and green hedges, in the style of Claude Monet, impressionist garden scene"),
    ("Snow-covered trees on a hill in winter", "Snow-covered hill with trees, impressionist painting, soft brushwork, Claude Monet winter style"),
    ("A steam train passing through a countryside station", "A steam train in a countryside station, impressionist style, inspired by Monet"),
    ("Sunset over calm ocean waves with orange glow", "Sunset over ocean, impressionist style with glowing pastels, Claude Monet inspired"),
    ("A forest path in early autumn", "A forest path in early autumn, impressionist colors and diffuse lighting, Monet style")
]

# Caminhos
BASE_MODEL = "CompVis/stable-diffusion-v1-4"
LORA_DIR = "lora-finetune/trained-model/model"
OUT_DIR = "comparacao_lora_vs_base"
SEED = 42
STEPS = 30
GUIDANCE = 7.5

def gerar_imagem(pipeline, prompt, nome, subdir):
    image = pipeline(prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE).images[0]
    path = os.path.join(OUT_DIR, subdir)
    os.makedirs(path, exist_ok=True)
    image.save(os.path.join(path, f"{nome}.png"))
    print(f"✅ {subdir}: {nome}")

def main():
    device = "cpu" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float32
    manual_seed(SEED)

    # PIPELINE BASE
    print("🔵 Carregando modelo base...")
    pipe_base = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)

    # PIPELINE COM LORA
    print("🟢 Carregando modelo fine-tuned com LoRA...")
    pipe_lora = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    pipe_lora.load_lora_weights(LORA_DIR)

    # Geração
    for i, (prompt_lora, prompt_base) in enumerate(prompts):
        nome = f"{i+1:02d}_{prompt_lora[:30].replace(' ', '_')}"
        gerar_imagem(pipe_base, prompt_base, nome, "modelo_base")
        gerar_imagem(pipe_lora, prompt_lora, nome, "modelo_lora")

if __name__ == "__main__":
    main()
