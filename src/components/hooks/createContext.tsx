// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { createContext } from "react";
import { modelInputProps } from "../helpers/Interfaces";
import { Tensor } from "onnxruntime-web";

interface contextProps {
  click: [
    click: modelInputProps | null,
    setClick: (e: modelInputProps | null) => void,
  ];
  clicks: [
    clicks: modelInputProps[] | null,
    setClicks: (e: modelInputProps[] | null) => void,
  ];
  image: [
    image: HTMLImageElement | null,
    setImage: (e: HTMLImageElement | null) => void,
  ];
  segmentTypes: [
    segmentTypes: "Box" | "Click",
    setSegmentTypes: (e: "Box" | "Click") => void,
  ];
  maskImg: [
    maskImg: HTMLImageElement | null,
    setMaskImg: (e: HTMLImageElement | null) => void,
  ];
  showLoadingModal: [
    showLoadingModal: boolean,
    setShowLoadingModal: React.Dispatch<React.SetStateAction<boolean>>,
  ];
  predMask: [
    predMask: Tensor | null,
    setPredMask: React.Dispatch<React.SetStateAction<Tensor | null>>,
  ];
  predMasks: [
    predMasks: Tensor[] | null,
    setPredMasks: React.Dispatch<React.SetStateAction<Tensor[] | null>>,
  ];
  predMasksHistory: [
    predMasksHistory: Tensor[] | null,
    setPredMasksHistory: React.Dispatch<React.SetStateAction<Tensor[] | null>>,
  ];
}

const AppContext = createContext<contextProps | null>(null);

export default AppContext;
