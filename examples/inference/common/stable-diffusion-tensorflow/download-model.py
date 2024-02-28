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

#!/usr/bin/env python3

from max import engine

import logging
import os
from argparse import ArgumentParser

# suppress extraneous logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras_cv
import tensorflow as tf

# In TensorFlow 2.0, setting TF_CPP_MIN_LOG_LEVEL does not work for all
# logging messages, so we have to setLevel here again to suppress them.
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


DEFAULT_MODEL_DIR = "../../../models/stable-diffusion-tensorflow"
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
            f"\tname: {x.name:<25} shape: {str(x.shape):<20} dtype: {x.dtype:<15}"
        )


def load(session, output_dir: str, model_name: str):
    model = session.load(f"{output_dir}/{model_name}")
    print(f"{model_name}:")
    print("=" * 80)
    print("Inputs:")
    print_metadata(model.input_metadata)
    print("Outputs:")
    print_metadata(model.output_metadata)
    print("")


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

    # Load models and print input/output metadata. This step enables one to see
    # the input shapes and tensor names used when writing the application code.
    # NOTE: This compiles each model in full; it may take a few minutes.
    print("\nLoading...")
    session = engine.InferenceSession()
    load(session, args.output_dir, "txt-encoder")
    load(session, args.output_dir, "img-encoder")
    load(session, args.output_dir, "img-decoder")
    load(session, args.output_dir, "img-diffuser")


if __name__ == "__main__":
    main()
