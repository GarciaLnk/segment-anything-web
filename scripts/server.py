import base64
import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from typing_extensions import Annotated

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

curr_dir = os.path.dirname(__file__)
checkpoint = curr_dir + "/model/sam_vit_h.pth"
model_type = "vit_h"

# Download the model if it doesn't exist
if not os.path.exists(checkpoint):
    import requests
    from tqdm import tqdm

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(checkpoint, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


def set_device(dev: str):
    global device
    if dev:
        device = dev
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"


@app.post("/api/embedding")
async def create_upload_file(file: Annotated[bytes, File()]):
    # Read the image file
    image_data = Image.open(io.BytesIO(file))
    nparr = np.array(image_data)

    # Embedding generation
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    _ = sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(nparr)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    print(image_embedding.shape)

    embedding_base64 = base64.b64encode(image_embedding.tobytes()).decode("utf-8")

    return [embedding_base64]


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Server for embedding generation using SAM."
    )

    parser.add_argument("--device", type=str, help="The device to run the model on.")

    args = parser.parse_args()
    set_device(args.device)

    uvicorn.run(app, host="0.0.0.0", port=3000)
