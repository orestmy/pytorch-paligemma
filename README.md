# pytorch-paligemma

This repository contains a minimal PyTorch implementation of **PaliGemma**, a multi-modal model capable of generating text from an image + text prompt.  The code mirrors the architecture used by Google's PaLiGemma model but is implemented from scratch using standard PyTorch modules.

## Repository structure

- `modeling_siglip.py` – Vision backbone implementing the SigLIP transformer.
- `modeling_gemma.py` – Language model and multi-modal fusion components.
- `processing_paligemma.py` – Image preprocessing and tokenizer wrapper.
- `inference.py` – Example script for running inference with a trained model.
- `utils.py` – Helper utilities for loading weights from HuggingFace formats.

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A set of pre-trained weights is required for inference. Download a PaLiGemma checkpoint (for example `paligemma-3b-pt-224`) and note the path on disk.
https://huggingface.co/google/paligemma-3b-pt-224/tree/main

```bash
huggingface-cli login
huggingface-cli download google/paligemma-3b-pt-224 --local-dir paligemma-weights/ --local-dir-use-symlinks False
```

## Running inference

Invoke `inference.py` with your model path, a text prompt and an image file:

```bash
python inference.py \
    --model_path /path/to/paligemma-3b-pt-224 \
    --prompt "A description of the image" \
    --image_file_path path/to/image.jpg
```

Additional flags allow configuring sampling behaviour, maximum tokens and device selection. See `launch_inference.sh` for an example.

## Web Interface

A web-based interface is available in the `paligemma-web-gui` directory. To run the web application:

1. Ensure you have the additional requirements installed:
```bash
pip install flask gunicorn
```

2. Set the model path (if different from default):
```bash
export MODEL_PATH=path/to/paligemma-weights
```

3. Launch the web interface using Gunicorn:
```bash
cd paligemma-web-gui
PYTHONPATH=/home/user/pytorch-paligemma gunicorn -b 0.0.0.0:5000 app:app
```

The web interface will be available at `http://localhost:5000`. You can upload images and ask questions about them through the browser.

Note: When running on a remote server, replace `localhost` with your server's IP address and ensure port 5000 is accessible.

## Tests

This project does not currently include automated tests.