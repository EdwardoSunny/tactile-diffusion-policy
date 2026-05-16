"""
Convert a tactile-data-collection canonical training zarr into the zarr layout
that diffusion_policy's ``ReplayBuffer`` reads directly.

Upstream pipeline (lives in /data/edward/tactile-data-collection — see that
repo's README.md): a teleop session produces a raw zarr, a render script
augments it with overlay variants, and ``build_training_datasets.py`` splits
that into one zarr per overlay mode under ``/data/edward/training_datasets/``
(raw / arrow / grid / point / bar). Each training zarr has:

    /data/state    (N, 10) float32   xyz_mm + 6D rotation + grasp
    /data/action   (N, 10) float32   action[t] = state[t+1] within episode
    /data/img_0    (N, 224, 224, 3) float32 in [0,1]   agent cam (BGR)
    /data/img_1    (N, 224, 224, 3) float32 in [0,1]   wrist cam (BGR)
    /meta/episode_ends                                  cumulative ends
    /meta/{image_mode, action_kind, state_kind, tactile_baseline, camera_*, ...}

This script renames the per-camera image keys, converts them to uint8, and
passes everything else through. Output schema (matches what
``ReplayBuffer.copy_from_path(..., keys=['image','wrist_image','state','action'])``
expects):

    /data/image         (N, 224, 224, 3) uint8     agentview (img_0 from source)
    /data/wrist_image   (N, 224, 224, 3) uint8     wrist (img_1 from source)
    /data/state         (N, 10) float32            copied verbatim
    /data/action        (N, 10) float32            copied verbatim
    /meta/episode_ends  (E,) int64                 copied verbatim
    /meta/* (all)       copied verbatim from the source

Image arrays are converted from float32 [0,1] -> uint8 [0,255] because
(a) diffusion_policy's example datasets store uint8, (b) it's 4x smaller on
disk, and (c) the dataset class divides by 255 at load time anyway.

Use with the companion ``diffusion_policy.dataset.xarm_image_dataset.XArmImageDataset``
(in this repo at ``diffusion_policy/dataset/xarm_image_dataset.py``).

Usage (run from anywhere; nothing is imported from the tactile-data-collection repo):
    python -m diffusion_policy.scripts.tactile_xarm_conversion \\
        /data/edward/training_datasets/arrow.zarr \\
        /data/edward/diffusion_policy_data/xarm_arrow.zarr

Or directly:
    python /data/edward/diffusion_policy/diffusion_policy/scripts/tactile_xarm_conversion.py \\
        /data/edward/training_datasets/arrow.zarr \\
        /data/edward/diffusion_policy_data/xarm_arrow.zarr
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import zarr


def _bare_camera_indices(group):
    import re
    pat = re.compile(r"^img_(\d+)$")
    return sorted(int(pat.match(k).group(1)) for k in group.keys() if pat.match(k))


def _stream_copy_uint8(src_arr, dst_arr):
    """Convert float32 [0,1] -> uint8 [0,255] in chunks; write directly to dst."""
    chunk = src_arr.chunks[0]
    for s in range(0, src_arr.shape[0], chunk):
        e = min(s + chunk, src_arr.shape[0])
        block = np.asarray(src_arr[s:e])
        dst_arr[s:e] = np.clip(block * 255.0, 0, 255).astype(np.uint8)


def _stream_copy_passthrough(src_arr, dst_arr):
    chunk = src_arr.chunks[0]
    for s in range(0, src_arr.shape[0], chunk):
        e = min(s + chunk, src_arr.shape[0])
        dst_arr[s:e] = src_arr[s:e]


def convert(src_path, dst_path):
    if not os.path.isdir(src_path):
        print(f"  [error] source zarr not found: {src_path}")
        sys.exit(1)
    if os.path.isdir(dst_path):
        print(f"  Wiping existing {dst_path} ...")
        shutil.rmtree(dst_path)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)

    src = zarr.open(src_path, mode="r")
    src_data = src["data"]
    src_meta = src["meta"]

    cam_idxs = _bare_camera_indices(src_data)
    if len(cam_idxs) < 1:
        print(f"  [error] source has no img_{{i}} arrays")
        sys.exit(1)
    if len(cam_idxs) > 2:
        print(f"  [warn] source has {len(cam_idxs)} cameras; only the first two "
              f"(img_0, img_1) will be mapped to image / wrist_image.")

    dst = zarr.open(dst_path, mode="a")
    dst_data = dst.require_group("data")
    dst_meta = dst.require_group("meta")

    t0 = time.time()

    # state + action: pass through (already 10-dim, already float32).
    for key in ("state", "action"):
        src_arr = src_data[key]
        dst_arr = dst_data.create_dataset(
            key,
            shape=src_arr.shape,
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
        )
        _stream_copy_passthrough(src_arr, dst_arr)
        print(f"  [copy] {key:14s}  shape={src_arr.shape}  dtype={src_arr.dtype}")

    # Per-frame tactile / n_contacts: pass through too. Some users will want
    # them as auxiliary obs (e.g. concat to state). Free to ignore otherwise.
    for key in ("n_contacts", "tactile", "tactile_connected",
                "tactile_ts_ms", "tactile_lag_ms"):
        if key not in src_data:
            continue
        src_arr = src_data[key]
        dst_arr = dst_data.create_dataset(
            key,
            shape=src_arr.shape,
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
        )
        _stream_copy_passthrough(src_arr, dst_arr)

    # Images: rename img_0 -> image, img_1 -> wrist_image; convert to uint8.
    img_compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    rename_map = []
    if len(cam_idxs) >= 1:
        rename_map.append((cam_idxs[0], "image"))
    if len(cam_idxs) >= 2:
        rename_map.append((cam_idxs[1], "wrist_image"))
    for cam_i, new_key in rename_map:
        src_arr = src_data[f"img_{cam_i}"]
        dst_arr = dst_data.create_dataset(
            new_key,
            shape=src_arr.shape,
            chunks=src_arr.chunks,
            dtype=np.uint8,
            compressor=img_compressor,
        )
        _stream_copy_uint8(src_arr, dst_arr)
        print(f"  [conv] img_{cam_i} -> {new_key}  ({src_arr.shape}, float32 [0,1] -> uint8)")

    # /meta: full pass-through.
    for key in src_meta.keys():
        src_arr = src_meta[key]
        dst_arr = dst_meta.create_dataset(key, shape=src_arr.shape, dtype=src_arr.dtype)
        dst_arr[...] = src_arr[...]

    print()
    print(f"  Done in {time.time() - t0:.1f}s.  Wrote {dst_path}.")
    print()
    print(f"  Reference the companion dataset class")
    print(f"    diffusion_policy.dataset.xarm_image_dataset.XArmImageDataset")
    print(f"  from a task YAML; template lives in that file's docstring.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="Source training zarr (e.g. "
                                "/data/edward/training_datasets/arrow.zarr)")
    ap.add_argument("dst", help="Destination zarr for diffusion_policy")
    args = ap.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
