import folder_paths
from pathlib import Path

_models = str(Path(__file__).resolve().parent / "models")
folders, extensions = folder_paths.folder_names_and_paths.setdefault("mosaicdiff", ([], {".onnx", ".pth"}))
if _models not in folders:
    folders.append(_models)
extensions.update({".onnx", ".pth"})

_lora_dir = str(Path(__file__).resolve().parent / "loras")
lora_folders, _lora_ext = folder_paths.folder_names_and_paths["loras"]
if _lora_dir not in lora_folders:
    lora_folders.append(_lora_dir)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
