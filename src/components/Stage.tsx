// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import React, { useContext, useEffect, useState } from "react";
import * as _ from "underscore";
import Tool from "./Tool";
import { StageProps, modelInputProps } from "./helpers/Interfaces";
import AppContext from "./hooks/createContext";

type Points = { sx: number; sy: number; x: number; y: number };

const Stage = ({ hasClicked, setHasClicked }: StageProps) => {
  const {
    click: [, setClick],
    clicks: [clicks, setClicks],
    segmentTypes: [segmentTypes],
    image: [image],
  } = useContext(AppContext)!;
  const [, setInputs] = useState<Array<modelInputProps>>([]);
  const [newInput, setNewInput] = useState<Array<modelInputProps>>([]);
  const [hasInput, setHasInput] = useState<boolean>(false);
  const [numOfDragEvents, setNumOfDragEvents] = useState<number>(0);
  const [points, setPoints] = useState<Points>();
  const DRAG_THRESHOLD = 4;

  const getInput = ({ sx, sy, x, y }: Points): modelInputProps => {
    return {
      x: sx,
      y: sy,
      width: x - sx,
      height: y - sy,
      clickType: 2,
    };
  };

  const getClick = (x: number, y: number): modelInputProps => {
    const clickType = 1;
    return { x, y, width: null, height: null, clickType };
  };

  const handleSegmentByClick = (x: number, y: number) => {
    const click = getClick(x, y);
    if (!click) return;
    setClicks([...(clicks || []), click]);
  };

  // Get mouse position and scale the (x, y) coordinates back to the natural
  // scale of the image.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const getXY = (e: any) => {
    const el = e.nativeEvent.target;
    const rect = el.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    const imageScale = image ? image.width / el?.offsetWidth : 1;
    x *= imageScale;
    y *= imageScale;
    return { x, y };
  };

  const handleMouseMove = (e: MouseEvent) => {
    const { x, y } = getXY(e);
    if (segmentTypes === "Click" && !hasClicked) {
      handleMoveToMask(e, x, y);
    } else if (newInput.length === 1) {
      const sx = newInput[0].x;
      const sy = newInput[0].y;
      setNewInput([getInput({ sx, sy, x, y })]);
      setInputs([]);
      setPoints({ sx, sy, x, y });
      setNumOfDragEvents((prevValue) => prevValue + 1);
    }
  };

  useEffect(() => {
    if (numOfDragEvents === DRAG_THRESHOLD && points) {
      setNumOfDragEvents(0);
      handleSegmentByBox(points);
    }
  }, [numOfDragEvents, points]);

  // Update the state of clicks with setClicks to trigger the ONNX model to run
  // and generate a new mask via a useEffect in App.tsx
  const handleMoveToMask = _.throttle((e: unknown, x: number, y: number) => {
    const click = getClick(x, y);
    if (click) setClicks([click]);
  }, 15);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleMouseUp = (e: any, shouldSetClick?: boolean) => {
    setHasClicked(true);
    const { x, y } = getXY(e);
    switch (segmentTypes) {
      case "Click":
        if (hasClicked || shouldSetClick) {
          if (shouldSetClick) {
            const newClick = getClick(x, y) || null;
            setClick(newClick);
          } else {
            handleSegmentByClick(x, y);
          }
        }
        break;
      case "Box": {
        if (!hasInput) return;
        const sx = newInput[0].x;
        const sy = newInput[0].y;
        const width = x - sx;
        const height = y - sy;
        const isClick = width === 0 && height === 0;
        setNewInput([]);
        setHasInput(false);
        if (isClick) {
          // A box must exist before a click is accepted
          if (clicks?.length && clicks[0].width && clicks[0].height) {
            const newClick = getClick(x, y);
            const boxPoints = {
              sx: clicks[0].x,
              sy: clicks[0].y,
              x: clicks[0].width,
              y: clicks[0].height,
            };
            adjustPointsToRange(boxPoints, newClick);
            setInputs([getInput(boxPoints)]);
            handleSegmentByBox(boxPoints, newClick);
          } else {
            setHasClicked(false);
          }
        } else {
          const points = { sx, sy, x, y };
          setPoints(points);
          adjustPointsToRange(points);
          setInputs([getInput(points)]);
          handleSegmentByBox(points);
        }
        break;
      }
      default:
        null;
    }
  };

  const handleMouseDown = (e: React.ChangeEvent<MouseEvent>) => {
    if (segmentTypes === "Box") {
      const { x, y } = getXY(e);
      setNumOfDragEvents(0);
      if (newInput.length === 0) {
        setNewInput([{ x, y, width: 0, height: 0, clickType: -1 }]);
        setHasInput(true);
      }
    }
  };

  const handleSegmentByBox = (
    { sx, sy, x, y }: Points,
    extraClick?: modelInputProps,
    newerClicks?: modelInputProps[],
  ) => {
    const newClick = {
      x: Math.min(sx, x),
      y: Math.min(sy, y),
      width: Math.max(sx, x),
      height: Math.max(sy, y),
      clickType: 2,
    };
    const newClicks = newerClicks || [...(clicks || [])];
    if (extraClick) {
      newClicks.push(extraClick);
    }
    if (newClicks[0] && !newClicks[0].width) {
      newClicks.unshift(newClick);
    } else {
      newClicks[0] = newClick;
    }
    setClicks(newClicks);
  };

  const adjustPointsToRange = (
    points: Points,
    extraClick?: modelInputProps,
    newClicks?: modelInputProps[],
  ) => {
    const range = findClickRange(extraClick, newClicks);
    if (!range || !range.xMin || !range.yMin || !range.xMax || !range.yMax)
      return;
    let { sx, sy, x, y } = points;
    const xMin = Math.min(sx, x);
    const yMin = Math.min(sy, y);
    const xMax = Math.max(sx, x);
    const yMax = Math.max(sy, y);
    if (range.xMin < xMin) {
      if (sx < x) {
        sx = range.xMin;
      } else {
        x = range.xMin;
      }
    }
    if (range.yMin < yMin) {
      if (sy < y) {
        sy = range.yMin;
      } else {
        y = range.yMin;
      }
    }
    if (range.xMax > xMax) {
      if (sx > x) {
        sx = range.xMax;
      } else {
        x = range.xMax;
      }
    }
    if (range.yMax > yMax) {
      if (sy > y) {
        sy = range.yMax;
      } else {
        y = range.yMax;
      }
    }
    points.sx = sx;
    points.sy = sy;
    points.x = x;
    points.y = y;
  };

  const findClickRange = (
    extraClick?: modelInputProps,
    newClicks?: modelInputProps[],
  ) => {
    let xMin;
    let yMin;
    let xMax;
    let yMax;
    const allClicks = newClicks ? newClicks : clicks ? [...clicks!] : null;
    if (!allClicks) return;
    if (extraClick) {
      allClicks.push(extraClick);
    }
    for (const click of allClicks) {
      if (click.width) continue;
      if (click.clickType === 0) continue;
      if (!xMin || click.x < xMin) {
        xMin = click.x;
      }
      if (!yMin || click.y < yMin) {
        yMin = click.y;
      }
      if (!xMax || click.x > xMax) {
        xMax = click.x;
      }
      if (!yMax || click.y > yMax) {
        yMax = click.y;
      }
    }

    return { xMin, yMin, xMax, yMax };
  };

  const flexCenterClasses = "flex items-center justify-center";
  return (
    <div className={`${flexCenterClasses} w-full h-full`}>
      <div className={`${flexCenterClasses} relative w-[90%] h-[90%]`}>
        <Tool
          handleMouseMove={handleMouseMove}
          handleMouseUp={handleMouseUp}
          handleMouseDown={handleMouseDown}
          hasClicked={hasClicked}
        />
      </div>
    </div>
  );
};

export default Stage;
