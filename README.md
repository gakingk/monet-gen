# monet-gen
Monet style image generation comparison with prompt engineering and LoRA Fine-Tuning

# How to run:
Install requirements in your environment of preference
`
pip install -r requirements.txt
`

Generate images for comparison
`python comparar.py`

Currently the above script is configured to use CPU because my GPU wasn't handling it, you should change this to generate faster if yours can handle

After running, images are ready, you can see them in the comparacao_lora_vs_base folder

To get CLIP scores and FID for the generated images, run:
`
python clipfid.py
`

If you want to generate a nice ready-to-paste .tex file for your article with a collage of the generated images, run:
`
python qualitative.py
`

# If you want to train the model
Go to monet-gen/lora-finetune/model-training and run the python files in this order:
`
scrap-monetgallery.py #data scraps the monet gallery
generate_captions.py #generates captions for them
generate_metadata.py #makes an image-caption dictionary
prepare_dataset.py #turn this dict into a json idk
train_test_to_image_lora.py #trains the model, originally from CompVis/stable-diffusion-v1-4 with changes to use local storage
`
