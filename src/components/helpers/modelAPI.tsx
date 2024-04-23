// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { Tensor } from "onnxruntime-web";
import {
  setParmsandQueryModelProps,
  queryModelReturnTensorsProps,
  modeDataProps,
  modelInputProps,
} from "./Interfaces";

const API_ENDPOINT = process.env.API_ENDPOINT;

const setParmsandQueryModel = ({
  width,
  height,
  uploadScale,
  imgData,
  handleSegModelResults,
  imgName,
}: setParmsandQueryModelProps) => {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * uploadScale);
  canvas.height = Math.round(height * uploadScale);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(imgData, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(
    (blob) => {
      blob &&
        queryModelReturnTensors({
          blob,
          handleSegModelResults,
          imgName,
        });
    },
    "image/jpeg",
    1.0,
  );
};

const queryModelReturnTensors = async ({
  blob,
  handleSegModelResults,
  imgName,
}: queryModelReturnTensorsProps) => {
  if (!API_ENDPOINT) return;
  const req_data = new FormData();
  req_data.append("file", blob, imgName);

  const segRequest = fetch(`${API_ENDPOINT}/embedding`, {
    method: "POST",
    body: req_data,
  });

  segRequest.then(async (segResponse) => {
    const segJSON = await segResponse.json();
    const embedArr = segJSON.map((arrStr: string) => {
      const binaryString = window.atob(arrStr);
      const uint8arr = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        uint8arr[i] = binaryString.charCodeAt(i);
      }
      const float32Arr = new Float32Array(uint8arr.buffer);
      return float32Arr;
    });
    const lowResTensor = new Tensor("float32", embedArr[0], [1, 256, 64, 64]);
    handleSegModelResults({
      tensor: lowResTensor,
    });
  });
};

const getPointsFromBox = (box: modelInputProps) => {
  if (box.width === null || box.height === null) return;
  const upperLeft = { x: box.x, y: box.y };
  const bottomRight = { x: box.width, y: box.height };
  return { upperLeft, bottomRight };
};

const isFirstClick = (clicks: Array<modelInputProps>) => {
  return (
    (clicks.length === 1 && clicks[0].clickType === 1) ||
    (clicks.length === 2 && clicks.every((c) => c.clickType === 2))
  );
};

const modelData = ({
  clicks,
  tensor,
  modelScale,
  last_pred_mask,
}: modeDataProps) => {
  const imageEmbedding = tensor;
  let pointCoords;
  let pointLabels;
  let pointCoordsTensor;
  let pointLabelsTensor;

  // Check there are input click prompts
  if (clicks) {
    let n = clicks.length;
    const clicksFromBox = clicks[0].clickType === 2 ? 2 : 0;

    // If there is no box input, a single padding point with
    // label -1 and coordinates (0.0, 0.0) should be concatenated
    // so initialize the array to support (n + 1) points.
    pointCoords = new Float32Array(2 * (n + 1));
    pointLabels = new Float32Array(n + 1);

    // Check if there is a box input
    if (clicksFromBox) {
      // For box model need to include the box clicks in the point
      // coordinates and also don't need to include the extra
      // negative point
      pointCoords = new Float32Array(2 * (n + clicksFromBox));
      pointLabels = new Float32Array(n + clicksFromBox);
      const {
        upperLeft,
        bottomRight,
      }: {
        upperLeft: { x: number; y: number };
        bottomRight: { x: number; y: number };
      } = getPointsFromBox(clicks[0])!;
      pointCoords = new Float32Array(2 * (n + clicksFromBox));
      pointLabels = new Float32Array(n + clicksFromBox);
      pointCoords[0] = upperLeft.x * modelScale.samScale;
      pointCoords[1] = upperLeft.y * modelScale.samScale;
      pointLabels[0] = 2.0; // UPPER_LEFT
      pointCoords[2] = bottomRight.x * modelScale.samScale;
      pointCoords[3] = bottomRight.y * modelScale.samScale;
      pointLabels[1] = 3.0; // BOTTOM_RIGHT

      last_pred_mask = null;
    }

    // Add regular clicks and scale to what SAM expects
    for (let i = 0; i < n; i++) {
      pointCoords[2 * (i + clicksFromBox)] = clicks[i].x * modelScale.samScale;
      pointCoords[2 * (i + clicksFromBox) + 1] =
        clicks[i].y * modelScale.samScale;
      pointLabels[i + clicksFromBox] = clicks[i].clickType;
    }

    // Add in the extra point/label when only clicks and no box
    // The extra point is at (0, 0) with label -1
    if (!clicksFromBox) {
      pointCoords[2 * n] = 0.0;
      pointCoords[2 * n + 1] = 0.0;
      pointLabels[n] = -1.0;
      // update n for creating the tensor
      n = n + 1;
    }

    // Create the tensor
    pointCoordsTensor = new Tensor("float32", pointCoords, [
      1,
      n + clicksFromBox,
      2,
    ]);
    pointLabelsTensor = new Tensor("float32", pointLabels, [
      1,
      n + clicksFromBox,
    ]);
  }
  const imageSizeTensor = new Tensor("float32", [
    modelScale.height,
    modelScale.width,
  ]);

  if (pointCoordsTensor === undefined || pointLabelsTensor === undefined)
    return;

  // If there is a previous tensor, use it, otherwise we default to an empty tensor
  const lastPredMaskTensor =
    last_pred_mask && clicks && !isFirstClick(clicks)
      ? last_pred_mask
      : new Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]);

  // +!! is javascript shorthand to convert truthy value to 1, falsey value to 0
  const hasLastPredTensor = new Tensor("float32", [
    +!!(last_pred_mask && clicks && !isFirstClick(clicks)),
  ]);

  return {
    image_embeddings: imageEmbedding,
    point_coords: pointCoordsTensor,
    point_labels: pointLabelsTensor,
    orig_im_size: imageSizeTensor,
    mask_input: lastPredMaskTensor,
    has_mask_input: hasLastPredTensor,
  };
};

export { setParmsandQueryModel, modelData };
