# Llama 2 text completion demo 🔥

This is a minimal text completion demo compatible with the the official Llama 2
[text completion demo](https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/example_text_completion.py),
built with the [MAX Graph API](https://docs.modular.com/engine/graph).

## Usage

1. Set environment variables and download weights:

   ```shell
   bash setup.sh
   ```

   This script can be run from any directory and will download models to
   a `model` directory in your current working directory.

2. Run the text completion demo:

   ```shell
   bash run.sh
   ```

   or

   ```shell
   source setup.sh && \
   mojo \
    -I "$SCRIPT_DIR/tokenizer" \
    "$SCRIPT_DIR/run.🔥" \
    --model-path "${1:-$MODELS/stories15M.bin}" \
    --tokenizer-path "$MODELS/tokenizer.bin"
   ```

3. Run with the custom RoPE kernel:

Compile the custom mojo RoPE kernel and run LLama2:

   ```shell
   source setup.sh && source setup-custom-rope.sh && \
   mojo \
    -I "$SCRIPT_DIR/tokenizer" \
    "$SCRIPT_DIR/run.🔥" \
    --model-path "${1:-$MODELS/stories15M.bin}" \
    --tokenizer-path "$MODELS/tokenizer.bin" \
    --custom-ops-path "$CUSTOM_KERNELS/rope.mojopkg" \
    --enable-custom-rope-kernel
   ```
