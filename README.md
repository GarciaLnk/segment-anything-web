## Segment Anything Web demo

This **front-end only** React based web demo shows how to load a fixed image and corresponding `.npy` file of the SAM image embedding, and run the SAM ONNX model in the browser using Web Assembly with mulithreading enabled by `SharedArrayBuffer`, Web Worker, and SIMD128.

<img src="https://github.com/facebookresearch/segment-anything/raw/main/assets/minidemo.gif" width="500"/>

## Run the server

Create a conda environment with the dependencies in `scripts/environment.yml`:

```
mamba env create -f scripts/environment.yml
mamba activate sam-demo
```

Run the server:

```
python scripts/server.py
```

Wait for the model to be downloaded.

**NOTE:** You may need to run the vision model on the CPU if it doesn't fit in the VRAM. To do so, set the `--device cpu` flag when running the server.

```
python scripts/server.py --device cpu
```

## Export the ONNX model

You need to export the quantized ONNX model.

From the same conda environment, run:

```
python scripts/export_onnx_model.py --checkpoint scripts/model/sam_vit_h.pth --model-type vit_h --opset 17 --quantize-out model/sam_h_onnx_quantized.onnx --output model/sam_h_onnx.onnx
```

## Run the app

Install Yarn if needed:

```

npm install --g yarn

```

Copy .env.example to .env and set the `API_ENDPOINT` to the server endpoint (default is `http://localhost:3000/api`):

```
cp .env.example .env
```

Build and run:

```

yarn
yarn start

```

Navigate to [`http://localhost:8080/`](http://localhost:8080/)

Load an image and move your cursor around to see the mask prediction update in real time.

## ONNX multithreading with SharedArrayBuffer

To use multithreading, the appropriate headers need to be set to create a cross origin isolation state which will enable use of `SharedArrayBuffer` (see this [blog post](https://cloudblogs.microsoft.com/opensource/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/) for more details)

The headers below are set in `configs/webpack/dev.js`:

```js
headers: {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "credentialless",
}
```

## Structure of the app

**`App.tsx`**

- Initializes ONNX model
- Loads image embedding and image
- Runs the ONNX model based on input prompts

**`Stage.tsx`**

- Handles mouse move interaction to update the ONNX model prompt

**`Tool.tsx`**

- Renders the image and the mask prediction

**`helpers/maskUtils.tsx`**

- Conversion of ONNX model output from array to an HTMLImageElement

**`helpers/modelAPI.tsx`**

- Formats the inputs from the server for the ONNX model

**`helpers/scaleHelper.tsx`**

- Handles image scaling logic for SAM (longest size 1024)

**`hooks/`**

- Handle shared state for the app
