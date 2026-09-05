"""Repository-relative paths and the Hub fallback for weights.

A ModelSpec's `base.source`, a config's `checkpoint` / `base_source` / `vae_source` and
the render default are all written relative to the repository root
(`ckpts/<name>`), so an artifact carries no machine-specific path. When such a
directory is not present locally it is fetched from the Hub (`HF_WEIGHTS_REPO`, the
same layout) into the huggingface cache and used from there. Absolute paths pass through.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_WEIGHTS_REPO = "OpenVDN/vdn-minimax-h3"   # h3-base/ stage-b-step-2000/ stage-dmd-step-250/
WEIGHTS_PREFIX = "ckpts/"
LEGACY_WEIGHTS_PREFIX = "ckpts/converted/"   # how the training repo spells the same directories

H3_BASE = WEIGHTS_PREFIX + "h3-base"     # transformer/ vae/ audio_vae/ of the release
H3_UPSTREAM_REPO = "MiniMaxAI/MiniMax-H3"   # text_encoder/ tokenizer/ -- prompt encoding only


def resolve_repo_path(path):
    """Absolute path for `path`: unchanged if absolute or None, else joined to the
    repository root (not the cwd)."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def resolve_weights(path):
    """`resolve_repo_path`, plus the Hub fallback: a `ckpts/<name>` that does not exist
    locally is downloaded from HF_WEIGHTS_REPO (that subfolder only) and the cached copy
    is returned. `ckpts/converted/<name>` (a checkpoint exported by the training repo
    names its base that way) means the same directory."""
    if path is not None and path.startswith(LEGACY_WEIGHTS_PREFIX):
        path = WEIGHTS_PREFIX + path[len(LEGACY_WEIGHTS_PREFIX):]

    local = resolve_repo_path(path)
    if local is None or os.path.exists(local):
        return local

    if path.startswith(WEIGHTS_PREFIX):
        name = path[len(WEIGHTS_PREFIX):].strip("/")
        from huggingface_hub import snapshot_download

        root = snapshot_download(HF_WEIGHTS_REPO, allow_patterns=[f"{name}/**"])
        return os.path.join(root, name)

    return local


def upstream_snapshot(*subfolders):
    """The MiniMax-H3 Hub snapshot restricted to `subfolders` (downloaded on first use)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(H3_UPSTREAM_REPO,
                             allow_patterns=[f"{s}/**" for s in subfolders])
