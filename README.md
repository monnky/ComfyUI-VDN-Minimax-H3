(dont install : incomplete repository, wait few more hours)

-----------------------------------------
----------------------------------------
-------------------------------------
---------------------------------
-----------------------------


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
git clone <repository_url>
