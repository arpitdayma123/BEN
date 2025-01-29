import spaces
import gradio as gr
import torch
import os
import sys
from loadimg import load_img
from ben_base import BEN_Base
import random
import huggingface_hub
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(9)
torch.set_float32_matmul_precision("high")

model = BEN_Base()
# Download the model file from Hugging Face Hub
model_path = huggingface_hub.hf_hub_download(
    repo_id="PramaLLC/BEN2",
    filename="BEN2_Base.pth"
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model.loadcheckpoints(model_path)
model.to(device)
model.eval()

output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    result_image = process(im)
    image_path = os.path.join(output_folder, "foreground.png")
    result_image.save(image_path)
    return result_image, image_path


@spaces.GPU
def process_video(video_path):
    output_path = "./foreground.mp4"
    
    # print(type(video_path))
    # print(video_path)
    
    model.segment_video(video_path)  # This will save to ./foreground.mp4
    return output_path

@spaces.GPU
def process(image):
    foreground = model.inference(image)
    print(type(foreground))
    return foreground

def process_file(f):
    name_path = f.rsplit(".",1)[0]+".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path

# Interface components
image = gr.Image(label="Upload an image")
video = gr.Video(label="Upload a video")

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "image.jpg")
examples = load_img(image_path, output_type="pil")

# Image processing tab
tab1 = gr.Interface(
    fn,
    inputs=image,
    outputs=[
        gr.Image(label="Result Foreground"),
        gr.File(label="Download PNG")
    ],
    examples=[examples],
    api_name="image"
)

# Video processing tab
tab2 = gr.Interface(
    process_video,
    inputs=video,
    outputs=gr.Video(label="Result Video"),
    api_name="video",
    title="Video Processing (experimental)",
    description="Note: For ZeroGPU timeout, videos are limited to processing the first 100 frames only."
)

# Combined interface 
demo = gr.TabbedInterface(
    [tab1, tab2],
    ["Image Processing", "Video Processing"],
    title="BEN2 for background removal. Download the image/video for higher quality foreground.",
    # description="Note: Video processing is limited to the first 100 frames for performance reasons."
)

if __name__ == "__main__":
    demo.launch(show_error=True)