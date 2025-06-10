# monet-gen
Monet style image generation comparison with prompt engineering and LoRA Fine-Tuning

How to run:
Install requirements in your environment of preference
`pip install -r requirements.txt`

Generate images for comparison
`python comparar.py`

Images are ready, you can see them in the comparacao_lora_vs_base folder

To get CLIP scores and FID for the generated images, run:
`python clipfid.py`

If you want to generate a nice ready-to-paste .tex file for your article with a collage of the generated images, run:
`python qualitative.py`
