// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import React, { useState } from "react";
import { modelInputProps } from "../helpers/Interfaces";
import AppContext from "./createContext";
import { Tensor } from "onnxruntime-web";

const AppContextProvider = (props: {
  children: React.ReactElement<
    unknown,
    string | React.JSXElementConstructor<unknown>
  >;
}) => {
  const [click, setClick] = useState<modelInputProps | null>(null);
  const [clicks, setClicks] = useState<Array<modelInputProps> | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [segmentTypes, setSegmentTypes] = useState<"Box" | "Click">("Click");
  const [maskImg, setMaskImg] = useState<HTMLImageElement | null>(null);
  const [predMask, setPredMask] = useState<Tensor | null>(null);
  const [predMasks, setPredMasks] = useState<Tensor[] | null>(null);
  const [predMasksHistory, setPredMasksHistory] = useState<Tensor[] | null>(
    null,
  );

  return (
    <AppContext.Provider
      value={{
        click: [click, setClick],
        clicks: [clicks, setClicks],
        image: [image, setImage],
        segmentTypes: [segmentTypes, setSegmentTypes],
        maskImg: [maskImg, setMaskImg],
        showLoadingModal: useState<boolean>(false),
        predMask: [predMask, setPredMask],
        predMasks: [predMasks, setPredMasks],
        predMasksHistory: [predMasksHistory, setPredMasksHistory],
      }}
    >
      {props.children}
    </AppContext.Provider>
  );
};

export default AppContextProvider;
