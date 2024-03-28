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

import logging
import os
import signal
from argparse import ArgumentParser

# suppress extraneous logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras_cv
import tensorflow as tf

# In TensorFlow 2.0, setting TF_CPP_MIN_LOG_LEVEL does not work for all
# logging messages, so we have to setLevel here again to suppress them.
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


DEFAULT_MODEL_DIR = "stable-diffusion"
DESCRIPTION = "Download a Stable Diffusion model."


def export(model, output_dir: str):
    if os.path.exists(output_dir):
        print(f"Skipping {output_dir} (already exists)")
        return
    model.save(output_dir)
    print(f"Model saved to {output_dir}/.\n")


def print_metadata(metadata):
    for x in metadata:
        print(
            f"\tname: {x.name:<25} shape: {str(x.shape):<20} dtype:"
            f" {x.dtype:<15}"
        )


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Output directory for the downloaded model.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    tf.config.set_visible_devices([], "GPU")

    # Download Keras model.
    print("Downloading model ...\n")
    model = keras_cv.models.StableDiffusion(
        img_height=512, img_width=512, jit_compile=False
    )

    # Export to TF SavedModel format.
    print("\nExporting...\n")
    export(model.text_encoder, f"{args.output_dir}/txt-encoder")
    export(model.image_encoder, f"{args.output_dir}/img-encoder")
    export(model.decoder, f"{args.output_dir}/img-decoder")
    export(model.diffusion_model, f"{args.output_dir}/img-diffuser")


if __name__ == "__main__":
    main()
