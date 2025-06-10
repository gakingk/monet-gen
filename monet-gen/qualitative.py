import os
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = "comparacao_lora_vs_base/modelo_base"
LORA_DIR = "comparacao_lora_vs_base/modelo_lora"

PROMPTS = [
    "A calm lake at sunset",
    "A field of blooming wildflowers under a cloudy sky",
    "A small wooden bridge over a quiet river",
    "A seaside village with boats on the shore",
    "A rainy street in a 19th-century European town",
    "A lush garden with rose arches and green hedges",
    "Snow-covered trees on a hill in winter",
    "A steam train passing through a countryside station",
    "Sunset over calm ocean waves with orange glow",
    "A forest path in early autumn"
]

# Parâmetros
cols = 3  # Prompt, Base, LoRA
rows = len(PROMPTS)
figsize = (12, 2.5 * rows)

fig, axes = plt.subplots(rows, cols, figsize=figsize)
fig.suptitle("Análise Qualitativa: Modelo Base vs Modelo LoRA", fontsize=16)

for i, prompt in enumerate(PROMPTS):
    nome = f"{i+1:02d}_{prompt[:30].replace(' ', '_')}".lower()
    path_base = os.path.join(BASE_DIR, nome + ".png")
    path_lora = os.path.join(LORA_DIR, nome + ".png")

    # Prompt
    axes[i, 0].axis('off')
    axes[i, 0].text(0.5, 0.5, prompt, wrap=True, ha='center', va='center', fontsize=10)

    # Imagem base
    img_base = Image.open(path_base).convert("RGB")
    axes[i, 1].imshow(img_base)
    axes[i, 1].axis('off')
    if i == 0:
        axes[i, 1].set_title("Base", fontsize=12)

    # Imagem LoRA
    img_lora = Image.open(path_lora).convert("RGB")
    axes[i, 2].imshow(img_lora)
    axes[i, 2].axis('off')
    if i == 0:
        axes[i, 2].set_title("LoRA", fontsize=12)

# Salvar e exibir
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("analise_qualitativa.png", dpi=300)
plt.show()
