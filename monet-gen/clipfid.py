import os
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from scipy import linalg
import torch
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights

# ========= Dispositivo =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Diretórios =========
BASE_DIR = "comparacao_lora_vs_base/modelo_base"
LORA_DIR = "comparacao_lora_vs_base/modelo_lora"

# ========= Prompts =========
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

# ========= Modelos =========

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Inception para FID
weights = Inception_V3_Weights.IMAGENET1K_V1
inception = inception_v3(weights=weights, transform_input=False).eval().to(device)
inception.fc = torch.nn.Identity()

# ========= Transforms =========
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ========= Funções =========
@torch.no_grad()
def get_clipscore(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    return outputs.logits_per_image.item()

@torch.no_grad()
def get_inception_embedding(image):
    image = transform(image).unsqueeze(0).to(device)
    features = inception(image)  # output: (1, 2048)
    return features.squeeze().cpu().numpy()

def compute_fid(embeddings1, embeddings2):
    mu1, sigma1 = np.mean(embeddings1, axis=0), np.cov(embeddings1, rowvar=False)
    mu2, sigma2 = np.mean(embeddings2, axis=0), np.cov(embeddings2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

# ========= Execução principal =========
def main():
    scores = {"base": [], "lora": []}
    feats_base, feats_lora = [], []

    for i, prompt in enumerate(tqdm(PROMPTS)):
        nome = f"{i+1:02d}_{prompt[:30].replace(' ', '_')}".lower()
        path_base = os.path.join(BASE_DIR, nome + ".png")
        path_lora = os.path.join(LORA_DIR, nome + ".png")

        img_base = Image.open(path_base).convert("RGB")
        img_lora = Image.open(path_lora).convert("RGB")

        # CLIPScore
        scores["base"].append(get_clipscore(img_base, prompt))
        scores["lora"].append(get_clipscore(img_lora, prompt))

        # FID
        feats_base.append(get_inception_embedding(img_base))
        feats_lora.append(get_inception_embedding(img_lora))

        # Libera memória GPU a cada passo
        torch.cuda.empty_cache()

    fid_score = compute_fid(np.array(feats_base), np.array(feats_lora))

# (cabeçalho e funções anteriores mantidos iguais até o final da função main)

def main():
    scores = {
        "input_base": [],
        "input_lora": [],
        "estilo_base": [],
        "estilo_lora": []
    }
    feats_base, feats_lora = [], []
    clipscore_rows = []

    # Prompt estilo Monet padrão
    estilo_template = "{descricao}, impressionist style, soft brushstrokes, light pastel colors, in the style of Claude Monet"

    for i, prompt in enumerate(tqdm(PROMPTS)):
        nome = f"{i+1:02d}_{prompt[:30].replace(' ', '_')}".lower()
        path_base = os.path.join(BASE_DIR, nome + ".png")
        path_lora = os.path.join(LORA_DIR, nome + ".png")

        img_base = Image.open(path_base).convert("RGB")
        img_lora = Image.open(path_lora).convert("RGB")

        # Prompt alternativo com estilo
        estilo_prompt = estilo_template.format(descricao=prompt)

        # CLIPScore com prompt original
        score_input_base = get_clipscore(img_base, prompt)
        score_input_lora = get_clipscore(img_lora, prompt)
        scores["input_base"].append(score_input_base)
        scores["input_lora"].append(score_input_lora)

        # CLIPScore com prompt estilizado
        score_estilo_base = get_clipscore(img_base, estilo_prompt)
        score_estilo_lora = get_clipscore(img_lora, estilo_prompt)
        scores["estilo_base"].append(score_estilo_base)
        scores["estilo_lora"].append(score_estilo_lora)

        # Armazenar linha
        clipscore_rows.append((
            prompt, 
            score_input_base, score_input_lora,
            score_estilo_base, score_estilo_lora
        ))

        # FID
        feats_base.append(get_inception_embedding(img_base))
        feats_lora.append(get_inception_embedding(img_lora))

        torch.cuda.empty_cache()

    fid_score = compute_fid(np.array(feats_base), np.array(feats_lora))

    # === Relatório ===
    print("\n=== MÉDIAS CLIPSCORE ===")
    print(f"Prompt original - Base: {np.mean(scores['input_base']):.4f} | LoRA: {np.mean(scores['input_lora']):.4f}")
    print(f"Prompt estilizado - Base: {np.mean(scores['estilo_base']):.4f} | LoRA: {np.mean(scores['estilo_lora']):.4f}")
    print("\n=== FID ENTRE OS DOIS MODELOS ===")
    print(f"FID: {fid_score:.2f}")

    # === Tabela LaTeX ===
    with open("resultados_clip_fid.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{CLIPScore com prompt original e com estilo Monet para os dois modelos}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{p{4.5cm}cccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Prompt} & \\textbf{Base (orig)} & \\textbf{LoRA (orig)} & \\textbf{Base (Monet)} & \\textbf{LoRA (Monet)} \\\\\n")
        f.write("\\midrule\n")
        for p, s1, s2, s3, s4 in clipscore_rows:
            p = p.replace("_", "\\_")
            f.write(f"{p} & {s1:.2f} & {s2:.2f} & {s3:.2f} & {s4:.2f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"\\textbf{{Média}} & {np.mean(scores['input_base']):.2f} & {np.mean(scores['input_lora']):.2f} & {np.mean(scores['estilo_base']):.2f} & {np.mean(scores['estilo_lora']):.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{{tab:clipscore_duplo}}\n")
        f.write("\\fonte{Autoria própria}\n")
        f.write("\\end{table}\n\n")

        f.write(f"\\textbf{{FID}} entre base e LoRA: {fid_score:.2f}\n")

        print("✅ Tabela LaTeX salva como 'resultados_clip_fid.tex'")

        import matplotlib.pyplot as plt

        # === Gráfico de comparação CLIPScore ===
        labels = [p[:25] + "..." if len(p) > 28 else p for p, *_ in clipscore_rows]
        x = np.arange(len(labels))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - 1.5*width, scores["input_base"], width, label='Base (orig)', color="#4C72B0")
        rects2 = ax.bar(x - 0.5*width, scores["input_lora"], width, label='LoRA (orig)', color="#55A868")
        rects3 = ax.bar(x + 0.5*width, scores["estilo_base"], width, label='Base (Monet)', color="#C44E52")
        rects4 = ax.bar(x + 1.5*width, scores["estilo_lora"], width, label='LoRA (Monet)', color="#8172B3")

        ax.set_ylabel('CLIPScore')
        ax.set_title('Comparação de CLIPScore por prompt')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        fig.tight_layout()
        plt.savefig("clipscore_comparacao.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()

