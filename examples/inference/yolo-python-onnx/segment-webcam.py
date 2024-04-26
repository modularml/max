#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import time
import signal

from max.engine import InferenceSession

import torch
import onnxruntime
import numpy as np
import cv2

from argparse import ArgumentParser
from constants import CLASS_NAMES
from ultralytics.models.yolo.segment.predict import ops
from ultralytics.engine.results import Results

DESCRIPTION = "Segment images from a webcam using yolo."
DEFAULT_MODEL_DIR = "../../models/yolo"
WINDOW_NAME = "YOLOv8 Segmentation"


def resize_and_pad(image, shape):
    # Grab shape sizes.
    (h, w, _) = image.shape
    (target_h, target_w) = shape

    # Resize to fully fit within `shape`.
    min_ratio = min(target_h / h, target_w / w)
    unpadded_h = int(round(min_ratio * h))
    unpadded_w = int(round(min_ratio * w))
    image = cv2.resize(
        image, (unpadded_w, unpadded_h), interpolation=cv2.INTER_LINEAR
    )

    # Pad to be the same size as `shape`.
    delta_h = (target_h - unpadded_h) / 2
    delta_w = (target_w - unpadded_w) / 2
    top, bottom = int(round(delta_h - 0.1)), int(round(delta_h + 0.1))
    left, right = int(round(delta_w - 0.1)), int(round(delta_w + 0.1))
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def postprocess(out0, out1, input, frame):
    out0 = torch.from_numpy(out0)
    out1 = torch.from_numpy(out1)

    pred = ops.non_max_suppression(
        out0,
        conf_thres=0.25,
        iou_thres=0.70,
        agnostic=False,
        max_det=10,
        nc=len(CLASS_NAMES),
        classes=None,
    )[0]

    if not len(pred):
        result = Results(
            orig_img=frame,
            path="",
            names=CLASS_NAMES,
            boxes=pred[:, :6],
        )
    else:
        masks = ops.process_mask(
            out1[0],
            pred[:, 6:],
            pred[:, :4],
            input.shape[2:],
            upsample=True,
        )  # HWC
        pred[:, :4] = ops.scale_boxes(input.shape[2:], pred[:, :4], frame.shape)
        result = Results(
            orig_img=frame,
            path="",
            names=CLASS_NAMES,
            boxes=pred[:, :6],
            masks=masks,
        )
    return result


def main():
    # Parse args.
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the downloaded model",
    )
    parser.add_argument(
        "--onnx",
        "--onnx-runtime",
        action="store_true",
        default=False,
        help=(
            "Run video segmentation with the ONNX Runtime to compare"
            " performance"
        ),
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Compile & load models - this may take a few minutes.
    print("Loading and compiling model...")
    if args.onnx:
        engine_name = "ONNX Runtime"
        yolo = onnxruntime.InferenceSession(
            f"{args.model_dir}/yolov8n-seg.onnx"
        )
    else:
        engine_name = "MAX Engine"
        session = InferenceSession()
        yolo = session.load(f"{args.model_dir}/yolov8n-seg.onnx")
    print("Model compiled.\n")

    # Setup camera.
    print("Grabbing camera input...")
    print("Press escape or q to quit")
    cam = cv2.VideoCapture(0)

    previous_elapsed_ms = []

    # Loop camera frames running yolo.
    while cam.isOpened():
        # Load frame from camera.
        success, frame = cam.read()

        if success:
            # Resize to 640x480
            frame = resize_and_pad(frame, (480, 640))

            # Preprocess inputs.
            input = (
                frame[np.newaxis, :, :, ::-1]
                .transpose(0, 3, 1, 2)
                .astype(np.float32)
                / 255
            ).copy()

            # Run prediction.
            start = time.time()
            if args.onnx:
                outputs = yolo.run(None, {"images": input})
            else:
                outputs = list(yolo.execute(images=input).values())
            elapsed_ms = (time.time() - start) * 1000

            # Postprocess outputs.
            result = postprocess(outputs[0], outputs[1], input, frame)

            # Annotate and display frame.
            annotated_frame = result.plot()
            cv2.imshow(WINDOW_NAME, annotated_frame)

            # Calculated average fps and update window title.
            previous_elapsed_ms.append(elapsed_ms)
            previous_elapsed_ms = previous_elapsed_ms[-100:]
            average_elapsed_ms = np.average(previous_elapsed_ms)
            cv2.setWindowTitle(
                WINDOW_NAME,
                (
                    f"{engine_name}: YOLOv8 Segmentation (Average Model Time:"
                    f" {average_elapsed_ms:0.1f}ms)"
                ),
            )

            # Exit on escape or q.
            ESC = 27
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ESC]:
                break
        else:
            print("Can't review frame. Camera stream ended?")
            break

    # Clean up resources.
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
