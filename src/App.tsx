// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { InferenceSession, Tensor } from "onnxruntime-web";
import React, { useContext, useEffect, useState } from "react";
import "./assets/scss/App.scss";
import { handleImageScale } from "./components/helpers/scaleHelper";
import { modelScaleProps } from "./components/helpers/Interfaces";
import { onnxMaskToImage } from "./components/helpers/maskUtils";
import {
  modelData,
  setParmsandQueryModel,
} from "./components/helpers/modelAPI";
import Stage from "./components/Stage";
import AppContext from "./components/hooks/createContext";
const ort = require("onnxruntime-web");
import npyjs from "npyjs";
import { useFilePicker } from "use-file-picker";
import {
  FileAmountLimitValidator,
  FileTypeValidator,
  FileSizeValidator,
} from "use-file-picker/validators";
import LoadingModal from "./components/LoadingModal";

// Define image, embedding and model paths
const IMAGE_PATH = "/assets/data/dogs.jpg";
const IMAGE_EMBEDDING = "/assets/data/dogs_embedding.npy";
const MODEL_DIR = process.env.MODEL_DIR;

const App = () => {
  const {
    click: [, setClick],
    clicks: [clicks, setClicks],
    image: [, setImage],
    segmentTypes: [segmentTypes, setSegmentTypes],
    maskImg: [, setMaskImg],
    showLoadingModal: [showLoadingModal, setShowLoadingModal],
    predMask: [predMask, setPredMask],
    predMasks: [predMasks, setPredMasks],
    predMasksHistory: [predMasksHistory],
  } = useContext(AppContext)!;
  const [model, setModel] = useState<InferenceSession | null>(null); // ONNX model
  const [tensor, setTensor] = useState<Tensor | null>(null); // Image embedding tensor
  const [hasClicked, setHasClicked] = useState<boolean>(false);

  // The ONNX model expects the input to be rescaled to 1024.
  // The modelScale state variable keeps track of the scale values.
  const [modelScale, setModelScale] = useState<modelScaleProps | null>(null);

  // File picker for image upload
  const { openFilePicker, filesContent } = useFilePicker({
    readAs: "DataURL",
    accept: "image/*",
    multiple: false,
    validators: [
      new FileAmountLimitValidator({ max: 1 }),
      new FileTypeValidator(["jpg", "png"]),
      new FileSizeValidator({ maxFileSize: 50 * 1024 * 1024 /* 50 MB */ }),
    ],
  });

  // Initialize the ONNX model. load the image, and load the SAM
  // pre-computed image embedding
  useEffect(() => {
    // Initialize the ONNX model
    const initModel = async () => {
      try {
        if (MODEL_DIR === undefined) return;
        const URL: string = MODEL_DIR;
        const model = await InferenceSession.create(URL);
        setModel(model);
      } catch (e) {
        console.log(e);
      }
    };
    initModel();

    // Load the image
    const url = new URL(IMAGE_PATH, location.origin);
    loadImage(url.href, url.pathname.split("/").pop()!, true);
  }, []);

  // Load the image from the file picker
  useEffect(() => {
    if (filesContent.length > 0) {
      const file = filesContent[0].content;
      const name = filesContent[0].name;
      handleResetState();
      loadImage(file, name, false);
    }
  }, [filesContent]);

  const loadImage = async (
    dataUrl: string,
    imageName: string,
    init: boolean,
  ) => {
    try {
      const img = new Image();
      img.src = dataUrl;
      img.onload = () => {
        const { height, width, samScale } = handleImageScale(img);
        setModelScale({
          height: height, // original image height
          width: width, // original image width
          samScale: samScale, // scaling factor for image which has been resized to longest side 1024
        });
        img.width = width;
        img.height = height;
        setImage(img);

        if (init) {
          // Load the Segment Anything pre-computed embedding
          Promise.resolve(loadNpyTensor(IMAGE_EMBEDDING, "float32"))
            .then((embedding) => setTensor(embedding))
            .catch((e) => {
              handleResetState();
              setImage(null);
              setTensor(null);
              console.log(e);
            });
        } else {
          setShowLoadingModal(true);
          setParmsandQueryModel({
            width: img.width,
            height: img.height,
            uploadScale: samScale,
            imgData: img,
            handleSegModelResults,
            imgName: imageName,
          });
        }
      };
    } catch (error) {
      console.log(error);
    }
  };

  const handleSegModelResults = ({ tensor }: { tensor: Tensor }) => {
    setTensor(tensor);
    setShowLoadingModal(false);
  };

  // Decode a Numpy file into a tensor.
  const loadNpyTensor = async (tensorFile: string, dType: string) => {
    const npLoader = new npyjs();
    const npArray = await npLoader.load(tensorFile);
    const tensor = new ort.Tensor(dType, npArray.data, npArray.shape);
    return tensor;
  };

  // Run the ONNX model every time clicks has changed
  useEffect(() => {
    runONNX();
  }, [clicks, hasClicked]);

  const runONNX = async () => {
    try {
      if (
        model === null ||
        clicks === null ||
        tensor === null ||
        modelScale === null
      )
        return;
      else {
        // Prepare the model input in the correct format for SAM.
        // The modelData function is from modelAPI.tsx.
        // console.log(predMask);
        const feeds = modelData({
          clicks,
          tensor,
          modelScale,
          last_pred_mask: predMask,
        });
        if (feeds === undefined) return;
        // Run the SAM ONNX model with the feeds returned from modelData()
        const results = await model.run(feeds);
        const output = results["masks"]; // model.outputNames[0] = "masks"
        if (hasClicked) {
          const pred_mask = results["low_res_masks"]; // model.outputNames[2] = "low_res_masks"

          setPredMask(pred_mask);
          if (!predMasksHistory) {
            setPredMasks([...(predMasks || []), pred_mask]);
          }
        }
        // The predicted mask returned from the ONNX model is an array which is
        // rendered as an HTML image using onnxMaskToImage() from maskUtils.tsx.
        setMaskImg(
          onnxMaskToImage(output.data, output.dims[2], output.dims[3]),
        );
      }
    } catch (e) {
      console.log(e);
    }
  };

  const handleResetState = () => {
    setMaskImg(null);
    setHasClicked(false);
    setClick(null);
    setClicks(null);
    setPredMask(null);
    setShowLoadingModal(false);
    setPredMasks(null);
  };

  return (
    <div className="container py-10 px-10 min-w-full flex flex-col items-center">
      <button
        className="bg-purple-900 text-white hover:bg-blue-400 font-bold py-2 px-4 rounded"
        onClick={() => openFilePicker()}
      >
        Select image{" "}
      </button>
      <div className="flex flex-row">
        <button
          className={
            "text-white hover:bg-blue-400 font-bold py-2 px-4 my-3 mx-3 rounded " +
            (segmentTypes === "Click" ? "bg-blue-400" : "bg-gray-400")
          }
          onClick={() => {
            segmentTypes !== "Click" && handleResetState();
            setSegmentTypes("Click");
          }}
        >
          Click
        </button>
        <button
          className={
            "text-white hover:bg-blue-400 font-bold py-2 px-4 my-3 mx-3 rounded " +
            (segmentTypes === "Box" ? "bg-blue-400" : "bg-gray-400")
          }
          onClick={() => {
            segmentTypes !== "Box" && handleResetState();
            setSegmentTypes("Box");
          }}
        >
          Box
        </button>
      </div>
      {showLoadingModal ? (
        <LoadingModal />
      ) : (
        <Stage hasClicked={hasClicked} setHasClicked={setHasClicked} />
      )}
      <button
        className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 my-3 rounded"
        onClick={() => handleResetState()}
      >
        Reset
      </button>
    </div>
  );
};

export default App;
