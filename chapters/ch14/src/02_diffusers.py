from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

pipe = pipe.to(device)

prompt = "a cute colorful cartoon cat"
image = pipe(prompt).images[0]

current_dir = Path(__file__).parent 
img_dir = current_dir.parent / "img"
image.save(img_dir / "cartoon_cat.png")