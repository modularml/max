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

import os
import signal

# suppress extraneous logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from max.engine import InferenceSession, Model


import keras_cv
import numpy as np

from argparse import ArgumentParser
from constants import _ALPHAS_CUMPROD as ALPHAS
from math import log, sqrt
from PIL import Image

DEFAULT_MODEL_DIR = "../../models/stable-diffusion-tensorflow"
DESCRIPTION = "Generate an image based on the given prompt."
GUIDANCE_SCALE_FACTOR = 7.5


def tokenize(model, prompt) -> (np.ndarray, np.ndarray):
    """
    Tokenize & pad the given prompt.
    Returns: (token_ids, position_ids)
    Raises: ValueError if given prompt exceeds encoder range.
    """
    # Tokenize prompt
    N = model.text_encoder.get_config()["max_length"]
    tokens = model.tokenizer.encode(prompt)

    # Pad prompt if shorter than N
    if len(tokens) < N:
        tokens += [model.tokenizer.end_of_text] * (N - len(tokens))

    # Throw an error if the prompt is too long.
    if len(tokens) > N:
        raise ValueError(f"Prompt (len={len(tokens)} cannot exceed {N} tokens.")

    # Return token & position IDs
    return np.array(tokens), np.arange(N)


def get_timestep_embeddings(t_step: int, num_channels: int) -> np.ndarray:
    """Generate 1D embedding vector for given timestep & num-channels"""
    # Validate inputs.
    assert num_channels % 2 == 0, "num_channels must be even"
    c_by_2 = num_channels // 2
    # Generate frequencies.
    freq = np.exp(-log(10000) * np.arange(0, c_by_2) / c_by_2)
    # Scale by given timestep.
    freq *= t_step
    # Concatenate cos & sin together to form embeddings.
    return np.concatenate((np.cos(freq), np.sin(freq)))


def execute(model: Model, **kwargs) -> np.ndarray:
    """Execute the given model with the given args and return the first output tensor.
    """
    # The modular engine uses named args for both input and output tensors, but the output
    # names coming from tensorflow can be rather pedantic. Rather than dealing with them,
    # we take advantage of the fact that all models in this pipeline are single-output and
    # simply return the first output tensor.
    res = model.execute(**kwargs)
    assert len(res) == 1, "Multi-output model?"
    return list(res.values())[0]


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--prompt",
        type=str,
        metavar="<str>",
        required=True,
        help="Description of desired image.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        metavar="<str>",
        default="",
        help="Objects or styles to avoid in generated image.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        metavar="<int>",
        default=25,
        help="# of diffusion steps; trades-off speed vs quality",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="<int>",
        default=None,
        help="Seed for psuedo-random number generation.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the downloaded model.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="<outfile>",
        default="output.png",
        help="Output filename.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Set seed if requested
    if args.seed:
        np.random.seed(args.seed)

    # Download keras model
    #
    # Although we're going to run with the pre-downloaded models, we still need the top-level
    # keras model for the tokenizer and the config values. It should be cached though, so this
    # should run quick.
    print("Downloading model ...\n")
    model = keras_cv.models.StableDiffusion()
    print("\nModel downloaded.\n")

    # Compile & load models - this may take a few minutes.
    print("Loading and compiling models...")
    session = InferenceSession()
    txt_encoder = session.load(f"{args.model_dir}/txt-encoder")
    img_decoder = session.load(f"{args.model_dir}/img-decoder")
    img_diffuser = session.load(f"{args.model_dir}/img-diffuser")
    print("Models compiled.\n")

    # Tokenize inputs and run through text encoder.
    print("Processing input...")
    prompt_p, pos_p = tokenize(model, args.prompt)
    prompt_n, pos_n = tokenize(model, args.negative_prompt)
    context = execute(
        txt_encoder,
        tokens=np.stack((prompt_p, prompt_n)),
        positions=np.stack((pos_p, pos_n)),
    )
    print("Input processed.\n")

    # Initialize latent, timestep and alpha inputs.
    print("Initializing latent...")
    # The diffusion model has input_img_shape == output_img_shape. We extract the output
    # shape since there is only one output it's easier to index.
    # Note: For tensorflow, shapes are given in NHWC format.
    _, h, w, c = img_diffuser.output_metadata[0].shape
    latent = np.random.normal(size=(1, h, w, c))
    timesteps = np.arange(1, 1000, 1000 // args.num_steps)[::-1]
    alphas = [ALPHAS[t] for t in timesteps] + [1.0]

    # Loop through diffusion model
    for i, t in enumerate(timesteps):
        print(f"\rGenerating image: {i+1}/{args.num_steps}", end="")
        # Save latent from previous iteration
        latent_prev = latent

        # Get timestep embeddings
        #
        # From model inspection (either of the TF saved model or the loaded modular model;
        # see download script), we know that timestep embeddings must have 320 channels.
        temb = get_timestep_embeddings(t, 320)

        # Execute the diffusion model with bs=2. Both batches have same primary input and
        # timestep embeddings, but the context (primary prompt vs negative) differs.
        latent = execute(
            img_diffuser,
            latent=np.vstack((latent, latent)),
            context=context,
            timestep_embedding=np.stack((temb, temb)),
        )

        # Merge conditioned & unconditioned latents
        latent = latent[1] + GUIDANCE_SCALE_FACTOR * (latent[0] - latent[1])

        # Merge latent with previous iteration
        a_t, a_p = alphas[i], alphas[i + 1]
        pred = (latent_prev - sqrt(1 - a_t) * latent) / sqrt(a_t)
        latent = latent * sqrt(1 - a_p) + pred * sqrt(a_p)

    # Decode finalized latent
    print("\n\nDecoding image...")
    decoded = execute(img_decoder, input_2=latent)
    pixels = (np.clip((decoded + 1) / 2, 0, 1) * 255).astype("uint8")
    Image.fromarray(pixels.squeeze(), "RGB").save(args.output)
    print(f"Image saved to {args.output}.")
    return


if __name__ == "__main__":
    main()
