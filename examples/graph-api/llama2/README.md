# MIN Llama 2 text completion demo 🔥

This is a minimal text completion demo compatible with the the official Llama 2
[text completion demo](https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/example_text_completion.py).

## Usage

### Basic usage

1. Set environment variables and download weights:

   ```shell
   . examples/graph-api/llama2/setup.sh
   ```

2. Run the text completion demo:

   ```shell
   mojo \
       -D LLAMA_MODEL_PATH="$MODELS/stories110M.bin" \
       -D TOKENIZER_PATH="$MODELS/tokenizer.bin" \
       -I "$SCRIPT_DIR/tokenizer" \
       "$SCRIPT_DIR/run.🔥"
   ```

   or

   ```shell
   ./quickrun.sh "$MODELS/stories110M.bin"
   ```
