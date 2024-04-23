# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import cv2  # type: ignore
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

parser = argparse.ArgumentParser(
    description=("Converts an input image into embeddings.")
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input image to extract into embeddings.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=("Path to the output folder to save the embeddings to."),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use.",
)

parser.add_argument(
    "--device", type=str, default="cuda", help="The device to run the model on."
)


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)

    print("Processing...")
    image = cv2.imread(args.input)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    out_file_name = args.input.split("/")[-1].split(".")[0] + "_embedding.npy"
    np.save(args.output + out_file_name, image_embedding)

    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
