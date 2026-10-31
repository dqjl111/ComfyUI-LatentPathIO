import os
import json
import traceback

import folder_paths
import torch
import safetensors.torch

# ===========================================================================
# Helpers
# ===========================================================================

BASE_PATH = getattr(folder_paths, "base_path", os.getcwd())
NO_FILES = "(none found)"


def _out_dir():
    return os.path.abspath(folder_paths.get_output_directory())


def _in_dir():
    try:
        return os.path.abspath(folder_paths.get_input_directory())
    except Exception:
        return ""


def _latent_roots():
    roots, seen = [], set()
    for tag, d in (("output", _out_dir()), ("input", _in_dir()),
                   ("base", os.path.join(BASE_PATH, "output"))):
        if not d:
            continue
        d = os.path.abspath(d)
        if d in seen:
            continue
        seen.add(d)
        roots.append((tag, d))
    return roots


def _scan_latent_items(limit=3000):
    """Structured scan -> [{"dir": <relative dir>, "name": <file name>}].

    `dir` is relative to the output dir ("" = output root); files from other
    roots get an "input/" / "base/" prefix.
    """
    items, seen = [], set()
    seen_dirs = set()

    for tag, root in _latent_roots():
        if not os.path.isdir(root) or root in seen_dirs:
            continue
        seen_dirs.add(root)
        prefix = "" if tag == "output" else f"{tag}/"

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for f in sorted(filenames):
                if not f.lower().endswith(".latent"):
                    continue
                reldir = os.path.relpath(dirpath, root).replace("\\", "/")
                reldir = "" if reldir == "." else reldir
                d = (prefix + reldir).rstrip("/")
                key = (d, f)
                if key in seen:
                    continue
                seen.add(key)
                items.append({"dir": d, "name": f})
                if len(items) >= limit:
                    return items
    return items


def _scan_latents(limit=3000):
    """Combo values: unique FILE NAMES only (no folder prefix)."""
    names, seen = [], set()
    for it in _scan_latent_items(limit):
        if it["name"] not in seen:
            seen.add(it["name"])
            names.append(it["name"])
    return sorted(names) if names else [NO_FILES]


def _resolve_dir(folder):
    """Resolve the `folder` widget to an absolute directory."""
    folder = (folder or "").strip()
    if not folder:
        return _out_dir()
    if os.path.isabs(folder):
        return os.path.abspath(folder)
    cands = [os.path.join(BASE_PATH, folder), os.path.join(_out_dir(), folder)]
    if _in_dir():
        cands.append(os.path.join(_in_dir(), folder))
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[0]


def _resolve_entry(entry):
    """Resolve a legacy path-form value like "audioLatents/foo.latent"."""
    if not entry:
        return None
    entry = entry.replace("\\", "/").lstrip("/")
    out = _out_dir()
    inp = _in_dir()
    base_out = os.path.abspath(os.path.join(BASE_PATH, "output"))

    if entry.startswith("output/"):
        entry = entry[len("output/"):]
    if entry.startswith("input/") and inp:
        return os.path.join(inp, entry[len("input/"):])
    if entry.startswith("base/"):
        return os.path.join(base_out, entry[len("base/"):])

    candidate = os.path.join(out, entry)
    if os.path.isfile(candidate):
        return candidate
    if inp:
        alt = os.path.join(inp, entry)
        if os.path.isfile(alt):
            return alt
    return candidate


def _resolve_name(folder, name):
    """Resolve a plain file name.

    Strict: when `folder` is set, the file MUST be inside it -- there is
    deliberately NO cross-folder fallback, so an audio latent can never
    silently load into the video branch.
    """
    name = (name or "").strip().replace("\\", "/")
    if not name:
        return None

    # legacy value that still carries a relative path
    if "/" in name:
        return _resolve_entry(name)

    if (folder or "").strip():
        candidate = os.path.join(_resolve_dir(folder), name)
        return candidate

    # folder empty -> search every latent root
    for _tag, root in _latent_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if name in filenames:
                return os.path.join(dirpath, name)
    return None


def _resolve_path(path):
    """Resolve a user-typed path for the Path node."""
    path = (path or "").strip()
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cands = [os.path.join(BASE_PATH, path), os.path.join(_out_dir(), path)]
    if _in_dir():
        cands.append(os.path.join(_in_dir(), path))
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def _load_latent_file(path):
    latent = safetensors.torch.load_file(path, device="cpu")
    multiplier = 1.0
    if "latent_format_version_0" not in latent:
        multiplier = 1.0 / 0.18215
    return {"samples": latent["latent_tensor"].float() * multiplier}


# ===========================================================================
# Save Latent (Custom Path)
# ===========================================================================

class SaveLatentCustom:
    @classmethod
    def INPUT_TYPES(s):
        default_dir = os.path.join(folder_paths.get_output_directory(), "latents")
        return {
            "required": {
                "samples": ("LATENT",),
                "directory": ("STRING", {"default": default_dir, "multiline": False}),
                "filename": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "add_counter": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "latent/io"

    def save(self, samples, directory, filename, add_counter, overwrite, prompt=None, extra_pnginfo=None):
        directory = _resolve_dir(directory)
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"无法创建目录: {directory}\n{e}")

        filename = "".join(c for c in filename if c not in r'\/:*?"<>|').strip() or "ComfyUI"

        if add_counter and not overwrite:
            i = 0
            while True:
                candidate = os.path.join(directory, f"{filename}_{i:05}.latent")
                if not os.path.exists(candidate):
                    break
                i += 1
            file = candidate
        else:
            file = os.path.join(directory, f"{filename}.latent")
            if os.path.exists(file) and not overwrite:
                raise FileExistsError(f"文件已存在且不覆盖: {file}\n(把 overwrite 设为 true 可覆盖)")

        output = {
            "latent_tensor": samples["samples"].contiguous(),
            "latent_format_version_0": torch.tensor([]),
        }
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for k in extra_pnginfo:
                metadata[k] = json.dumps(extra_pnginfo[k])

        safetensors.torch.save_file(output, file, metadata=metadata)
        return {"ui": {"path": [file]}, "result": (samples, file)}


# ===========================================================================
# Load Latent (Path)
# ===========================================================================

class LoadLatentCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"path": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "latent/io"

    def load(self, path):
        if not (path or "").strip():
            raise ValueError("path 为空，请提供 .latent 文件的路径")
        resolved = _resolve_path(path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"找不到 latent 文件: {resolved}\n(原始输入: {path})")
        return (_load_latent_file(resolved),)


# ===========================================================================
# Load Latent (Folder)
#   folder      : which directory this node loads from (scopes the list)
#   latent_name : file name only -- never a folder prefix
# ===========================================================================

class LoadLatentCustomFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": "", "multiline": False}),
                "latent_name": (_scan_latents(),),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "latent/io"

    def load(self, folder, latent_name):
        if not latent_name or latent_name == NO_FILES:
            raise ValueError(
                "下拉里没有可加载的 latent。请先用 Save Latent (Custom Path) 保存一个 .latent，"
                "或改用 Load Latent (Path) 直接填完整路径。"
            )
        path = _resolve_name(folder, latent_name)
        if path and os.path.isfile(path):
            return (_load_latent_file(path),)

        where = _resolve_dir(folder) if (folder or "").strip() else "所有 latent 目录"
        raise FileNotFoundError(
            f"在 {where} 下找不到 {latent_name}。\n"
            f"请确认 folder 与文件名匹配（不会跨目录兜底，避免音频/视频 latent 串用）。"
        )


# ===========================================================================
# Server route
# ===========================================================================

try:
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    @routes.post("/latentpathio/list")
    async def _api_list_latents(request):
        items = _scan_latent_items()
        names = sorted({it["name"] for it in items}) or [NO_FILES]
        return web.json_response({"items": items, "files": names, "output_dir": _out_dir()})

    print("[LatentPathIO] 路由 /latentpathio/list 已注册")
except Exception:
    print("[LatentPathIO] 路由注册失败:")
    traceback.print_exc()


NODE_CLASS_MAPPINGS = {
    "Save Latent (Custom Path)": SaveLatentCustom,
    "Load Latent (Path)": LoadLatentCustom,
    "Load Latent (Folder)": LoadLatentCustomFromFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Latent (Custom Path)": "Save Latent (Custom Path)",
    "Load Latent (Path)": "Load Latent (Path)",
    "Load Latent (Folder)": "Load Latent (Folder)",
}
