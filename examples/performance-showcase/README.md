# Stock framework performance comparison

This directory contains a simple benchmark program along with model download
helpers to showcase inference performance using the Max Engine compared to
stock PyTorch and Tensorflow.

Note that the goal here is to provide a quick, qualitatively accurate showcase
of the performance differences you can expect in common scenarios (rather than
to establish a rigorous benchmark comparison).

## Quickstart

```sh
# Install requirements
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
# Install the MAX Engine Python package
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine

# Run the showcase program (--model accepts roberta and clip)
python3 run.py --model roberta

# Alternatively
python3 run.py -m clip
```
