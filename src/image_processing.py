import requests
import torch
import io
from PIL import Image
from src.utils import CFG

# Read and process image from URL
def read_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes)

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        return image
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from {url}. Error: {e}")
        return None

# Image processing using Bunny-Llama model
def process_image(image_url, model, tokenizer, prompt=''):
    image = read_image(image_url)
    if not image:
        return "", None

    # Prepare prompt for the model
    sys_prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in sys_prompt.split('<image>')]
    
    input_ids = torch.tensor(
        text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long
    ).unsqueeze(0).to(model.device) # use [device = next(model.parameters()).device] (if causing error)

    image_tensor = model.process_images(
        [image],
        model.config
    ).to(dtype=model.dtype, device=model.device) # use [device = next(model.parameters()).device] (if causing error)
    
    # Run inference
    output_ids = model.generate(input_ids, images=image_tensor, max_new_tokens=CFG.max_new_tokens, use_cache=True)[0]
    response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    return response, None
