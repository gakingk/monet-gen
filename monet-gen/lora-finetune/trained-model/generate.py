import torch
from diffusers import StableDiffusionPipeline
from torch import manual_seed
import argparse
import os

def main(args):
    # Define dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Carrega pipeline base
    print("🔄 Carregando modelo base...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype
    ).to(device)

    # Aplica pesos LoRA
    print("📦 Carregando pesos LoRA de:", args.lora_path)
    pipe.load_lora_weights(args.lora_path)

    # Semente aleatória para reprodutibilidade
    if args.seed is not None:
        manual_seed(args.seed)

    # Gera imagem
    print(f"🎨 Gerando imagem para o prompt:\n\"{args.prompt}\"\n")
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance
    ).images[0]

    # Salva imagem
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "imagem_gerada.png")
    image.save(output_path)
    print(f"✅ Imagem salva em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Modelo base")
    parser.add_argument("--lora_path", type=str, required=True, help="Caminho para o diretório com os pesos LoRA")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt textual")
    parser.add_argument("--steps", type=int, default=30, help="Número de passos de inferência")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (quanto seguir o texto)")
    parser.add_argument("--output_dir", type=str, default="inferencias", help="Diretório para salvar a imagem")
    parser.add_argument("--seed", type=int, default=None, help="Seed aleatória")
    
    args = parser.parse_args()
    main(args)

