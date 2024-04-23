// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { Tensor } from "onnxruntime-web";

export interface modelScaleProps {
  samScale: number;
  height: number;
  width: number;
}

export interface setParmsandQueryModelProps {
  width: number;
  height: number;
  uploadScale: number;
  imgData: HTMLImageElement;
  handleSegModelResults: ({ tensor }: { tensor: Tensor }) => void;
  imgName: string;
}

export interface queryModelReturnTensorsProps {
  blob: Blob;
  handleSegModelResults: ({ tensor }: { tensor: Tensor }) => void;
  imgName: string;
}

export interface modelInputProps {
  x: number;
  y: number;
  width: null | number;
  height: null | number;
  clickType: number;
}

export interface modeDataProps {
  clicks?: Array<modelInputProps>;
  tensor: Tensor;
  modelScale: modelScaleProps;
  last_pred_mask: Tensor | null;
}

export interface ToolProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handleMouseMove: (e: any) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handleMouseUp: (e: any, forceHasClicked?: boolean) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handleMouseDown: (e: any) => void;
  hasClicked: boolean;
}

export interface StageProps {
  hasClicked: boolean;
  setHasClicked: (e: boolean) => void;
}
