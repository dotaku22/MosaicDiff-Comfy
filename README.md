# MosaicDiff for ComfyUI

Restore mosaic in ComfyUI with BasicVSR++, then MiniMax H3. The workflow uses Comfy's own video loader, model loaders, and sampler. Three custom node packs cover the parts Comfy does not ship.

The desktop program is a separate download: [MosaicDiff](https://github.com/dotaku22/MosaicDiff).

Censored AI-generated clips made with MosaicDiff are on [Civitai](https://civitai.red/models/2990026/mosaic-restoration).

It needs an NVIDIA GPU. It is slow. One second of video takes about one minute on an RTX 3070 Ti. RTX Video Super Resolution is used when the sampled frames are fitted back onto the crop.

## Install

Use a current ComfyUI that already has the MiniMax H3 nodes.

Copy these folders into `ComfyUI/custom_nodes`:

- `custom_nodes/ComfyUI-MosaicDiff`
- `custom_nodes/ComfyUI-H3-ContextWindows`
- `custom_nodes/comfyui_nvidia_rtx_nodes`

Copy `workflow/MosaicDiff.json` into `ComfyUI/user/default/workflows`.

From the ComfyUI Python environment:

```
pip install --no-deps mmengine==0.10.7
pip install -r custom_nodes/ComfyUI-MosaicDiff/requirements.txt
pip install -r custom_nodes/comfyui_nvidia_rtx_nodes/requirements.txt
```

Install `mmengine` with `--no-deps` so it does not replace ComfyUI's PyTorch. If `onnxruntime`, `opencv-python`, or `av` are already installed, leave them.

Restart ComfyUI. Open the workflow **MosaicDiff** and choose the clip on **Load Video**.

## Models included here

These two files are not published anywhere else. They are already in the node pack. Clone this repo with [Git LFS](https://git-lfs.com). The green **Code → Download ZIP** button on GitHub does not download them.

| File | Where it already is |
| --- | --- |
| `rfdetr.onnx` | `custom_nodes/ComfyUI-MosaicDiff/models/` |
| `DecensoreH3_000002500.safetensors` | `custom_nodes/ComfyUI-MosaicDiff/loras/` |

The LoRA shows up in **Load LoRA** as `DecensoreH3_000002500`. The workflow uses strength 1.

## Models you download

| File | Put it in |
| --- | --- |
| [lada_mosaic_restoration_model_generic_v1.2.pth](https://huggingface.co/ladaapp/lada/resolve/3bfd69ffc21518bde80ba6b61696d51efd0a398b/lada_mosaic_restoration_model_generic_v1.2.pth) | `ComfyUI/custom_nodes/ComfyUI-MosaicDiff/models/basicvsr.pth` |
| [10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors](https://huggingface.co/TenStrip/10Eros-Max) | `ComfyUI/models/diffusion_models/` |
| [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) | `ComfyUI/models/text_encoders/` |
| [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `ComfyUI/models/vae/` |

The BasicVSR++ file must be named `basicvsr.pth`. Eros Max is about 20 GB.

## What the graph does

**Load Video**, **Load Diffusion Model**, **Load LoRA**, **Load CLIP**, and **Load VAE** are the normal Comfy nodes.

Sampling is **KSampler Select** (euler), **Basic Scheduler** (simple, 8 steps), **Random Noise** (seed 0), **Basic Guider**, and **Sampler Custom Advanced**.

**MosaicDiff Prepare** finds the mosaic and restores it with BasicVSR++. Resolution defaults to 512. Context defaults to 5 seconds.

**MiniMax H3 Context Windows** splits a long sample. Its length and overlap come from Prepare.

**MosaicDiff Pin Reference** keeps the border of the reference crop.

**MosaicDiff Place** fits the frames with RTX Video Super Resolution, pastes them back, and writes `<name>_mosaicdiff.mp4` in ComfyUI's output folder. **Save Video** previews that file.

## Licenses

See `licenses/NOTICES.txt`. The detector is Apache-2.0, copyright 2026 Kruk2. The BasicVSR++ code is AGPL-3.0 from Lada. The NVIDIA RTX node is Apache-2.0.
