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

import signal

from max.engine import InferenceSession

import numpy as np

from argparse import ArgumentParser
from diffusers import PNDMScheduler
from transformers import CLIPTokenizer
from PIL import Image

DEFAULT_MODEL_DIR = "../../models/stable-diffusion-onnx"
DESCRIPTION = "Generate an image based on the given prompt."
GUIDANCE_SCALE_FACTOR = 7.5
LATENT_SCALE_FACTOR = 0.18215
OUTPUT_HEIGHT = 512
OUTPUT_WIDTH = 512
LATENT_WIDTH = OUTPUT_WIDTH // 8
LATENT_HEIGHT = OUTPUT_HEIGHT // 8
LATENT_CHANNELS = 4


def main():
    # Parse args.
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

    # Set seed if requested.
    if args.seed:
        np.random.seed(args.seed)

    # Compile & load models - this may take a few minutes.
    print("Loading and compiling models...")
    session = InferenceSession()
    txt_encoder = session.load(f"{args.model_dir}/text_encoder/model.onnx")
    img_decoder = session.load(f"{args.model_dir}/vae_decoder/model.onnx")
    img_diffuser = session.load(f"{args.model_dir}/unet/model.onnx")
    print("Models compiled.\n")

    # Tokenize inputs and run through text encoder.
    print("Processing input...")
    tokenizer = CLIPTokenizer.from_pretrained(f"{args.model_dir}/tokenizer")
    prompt_p = tokenizer(
        args.prompt, padding="max_length", max_length=tokenizer.model_max_length
    )
    prompt_n = tokenizer(
        args.negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
    )

    input_ids = np.stack((prompt_p.input_ids, prompt_n.input_ids))
    encoder_hidden_states = txt_encoder.execute(input_ids=input_ids)[
        "last_hidden_state"
    ]
    print("Input processed.\n")

    # Initialize latent and scheduler.
    print("Initializing latent...")
    scheduler = PNDMScheduler.from_pretrained(f"{args.model_dir}/scheduler")

    # Note: For onnx, shapes are given in NCHW format.
    latent = np.random.normal(
        size=(1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)
    )
    latent = latent * scheduler.init_noise_sigma
    latent = latent.astype(np.float32)

    # Loop through diffusion model.
    scheduler.set_timesteps(args.num_steps)
    for i, t in enumerate(scheduler.timesteps):
        print(f"\rGenerating image: {i}/{args.num_steps}", end="")

        # Duplicate input and scale based on scheduler.
        sample = np.vstack((latent, latent))
        sample = scheduler.scale_model_input(sample, timestep=t)

        # Execute the diffusion model with bs=2. Both batches have same primary input and
        # timestep, but the encoder_hidden_states (primary prompt vs negative) differs.
        noise_pred = img_diffuser.execute(
            sample=sample,
            encoder_hidden_states=encoder_hidden_states,
            timestep=np.array([t], dtype=np.int64),
        )["out_sample"]

        # Merge conditioned & unconditioned outputs.
        noise_pred_text, noise_pred_uncond = np.split(noise_pred, 2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE_FACTOR * (
            noise_pred_text - noise_pred_uncond
        )

        # Merge latent with previous iteration.
        latent = scheduler.step(noise_pred, t, latent).prev_sample

    # Decode finalized latent.
    print("\n\nDecoding image...")
    latent = latent * (1 / LATENT_SCALE_FACTOR)
    decoded = img_decoder.execute(latent_sample=latent)["sample"]
    image = np.clip(decoded / 2 + 0.5, 0, 1).squeeze()
    image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(image, "RGB").save(args.output)
    print(f"Image saved to {args.output}.")
    return


if __name__ == "__main__":
    main()
