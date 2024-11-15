# MAX Pipelines

These are end-to-end pipelines that demonstrate the power of
[MAX](https://docs.modular.com/max/) for accelerating common AI workloads, and
more. Each of the supported pipelines can be served via an OpenAI-compatible
endpoint.

## Pipelines

The pipelines include:

- [Llama 3.1](llama3): A text completion pipeline using the Llama 3.1 model,
implemented using the MAX Graph API. This pipeline contains everything
needed to run a self-hosted large language model with state-of-the-art serving
throughput.
- [Mistral](mistral): Another text completion pipeline using the Mistral NeMo
12B model, implemented using the MAX Graph API.
- [Replit Code](replit): Code generation via the Replit Code V1.5 3B model,
implemented using the MAX Graph API.

Instructions for how to run each pipeline can be found in their respective
subdirectories, along with all configuration parameters. A shared driver is
used to execute the pipelines.

## Usage

The easiest way to try out any of the pipelines is with our Magic command-line
tool.

1. Install Magic on macOS and Ubuntu with this command:

   ```shell
   curl -ssL https://magic.modular.com | bash
   ```

   Then run the source command that's printed in your terminal.

   To see the available commands, you can run `magic --help`.
   [Learn more about Magic here](https://docs.modular.com/magic).

2. Clone the MAX examples repository:

   If you don't already have a local clone of this repository, create one via:

   ```shell
   git clone https://github.com/modularml/max.git
   ```

   The following instructions assume that you're present within this
   directory, and you can change to it after cloning:

   ```shell
   cd max/pipelines/python/
   ```

3. Now run one of the text completion demos with any of following commands:

   ```shell
   magic run llama3 --prompt "I believe the meaning of life is"
   magic run replit --prompt "def fibonacci(n):"
   magic run mistral --prompt "Why is the sky blue?"
   ```

4. Host a text completion endpoint via MAX Serve.

   MAX Serve provides functionality to host performant OpenAI compatible
   endpoints using the FastAPI framework.

   You can configure a pipeline to be hosted by using the `--serve` argument.
   For example:

   ```shell
   magic run llama3 --serve
   ```

   A request can be submitted via a cURL command.

   ```shell
   curl -N http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
       "stream": true,
       "messages": [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Who won the world series in 2020?"}
       ]
   }'
   ```
