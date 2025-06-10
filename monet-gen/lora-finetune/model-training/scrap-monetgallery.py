import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image, ImageOps
from io import BytesIO
from tqdm import tqdm

# Parâmetros
BASE_URL = "https://claudemonetgallery.org"
TOTAL_PAGES = 170
OUTPUT_DIR = "monet_images_512"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Cria diretório se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_filename(name):
    name = name.lower().strip()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '-', name)
    return name + ".jpg"

def resize_with_padding(image, size=(512, 512), fill_color=(0, 0, 0)):
    return ImageOps.pad(image, size, method=Image.LANCZOS, color=fill_color)

def process_gallery_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.select("a.d-block > img")

        for img_tag in tqdm(img_tags, desc=f"Processando {url.split('/')[-1]}", leave=False):
            img_src = img_tag.get("src")
            alt_text = img_tag.get("alt")

            if not img_src or not alt_text:
                continue

            img_url = urljoin(BASE_URL, img_src.split("?")[0])
            filename = sanitize_filename(alt_text)
            filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(filepath):
                continue

            try:
                img_data = requests.get(img_url, headers=HEADERS).content
                img = Image.open(BytesIO(img_data)).convert("RGB")
                img_padded = resize_with_padding(img, size=(512, 512))
                img_padded.save(filepath, format="JPEG")
            except Exception as e:
                print(f"Erro ao processar imagem {img_url}: {e}")

    except Exception as e:
        print(f"✗ Erro ao acessar página {url}: {e}")

# Loop por todas as páginas com barra de progresso principal
for page in tqdm(range(1, TOTAL_PAGES + 1), desc="Baixando páginas"):
    if page == 1:
        page_url = f"{BASE_URL}/the-complete-works.html"
    else:
        page_url = f"{BASE_URL}/the-complete-works_pageno-{page}.html"

    process_gallery_page(page_url)
    time.sleep(1)  # respeitar servidor
