# Download Workflow

https://www.patreon.com/RareTutor/posts/rt-minimax-h3-8-168668708

# How to use the Node in ComfyUI ???

[![Watch the Tutorial](https://img.youtube.com/vi/zUk6mqruWdA/maxresdefault.jpg)](https://www.youtube.com/watch?v=zUk6mqruWdA)


# RT Minimax H3 VDN Pro (VideoDeltaNet on MiniMax H3)

This custom node brings **VideoDeltaNet (VDN-H3)** hybrid attention into ComfyUI. It replaces standard quadratic softmax attention with chunked sliding-window softmax and bidirectional linear delta-rule memory, fully compatible with all NVIDIA GPUs on Windows and Linux.

---

## 🚀 Key Features

* **ComfyUI Drop-in Node**: Connects directly between your Model Loader and `KSampler`.
* **Universal NVIDIA Support**: Runs smoothly on any NVIDIA GPU supported by PyTorch and CUDA (RTX 30-series, 40-series, 50-series, Ada, Hopper, Blackwell, and enterprise cards).
* **Native PyTorch Decomposed SDPA**: Runs fast, chunk-aligned windowed attention without requiring complex external compilers.
* **VRAM Streaming Compatible**: Allows ComfyUI's memory manager to stream model blocks to your GPU without running out of VRAM.
* **8-Step DMD Distillation**: Full support for the distilled 8-step Turbo adapter for ultra-fast generation.

---

## 📦 Installation

Clone this repository directly into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/monnky/ComfyUI-VDN-Minimax-H3
```

## Credits & License
- Original Base Weights: OpenVDN/vdn-minimax-h3 (https://huggingface.co/OpenVDN/vdn-minimax-h3)
- Project Blog: Video DeltaNet (https://openvdn.github.io/)
- License: MiniMax H3 Community License Agreement (https://huggingface.co/MiniMaxAI/MiniMax-H3)
