"""
Diffusion-policy Dataset class for the xArm tactile dataset.

Companion to ``diffusion_policy.scripts.tactile_xarm_conversion``, which
converts a canonical training zarr (produced by the tactile-data-collection
repo at /data/edward/tactile-data-collection — see its README.md) into the
zarr layout this class reads.

Create a task YAML at ``diffusion_policy/config/task/xarm_image.yaml`` shaped
like the PushT example:

    name: xarm_image
    image_shape: [3, 224, 224]
    shape_meta:
      obs:
        image:        {shape: [3, 224, 224], type: rgb}
        wrist_image:  {shape: [3, 224, 224], type: rgb}
        agent_pos:    {shape: [10]}             # full 10-dim state
      action:
        shape: [10]
    dataset:
      _target_: diffusion_policy.dataset.xarm_image_dataset.XArmImageDataset
      zarr_path: /data/edward/diffusion_policy_data/xarm_arrow.zarr
      horizon: ${horizon}
      pad_before: ${eval:'${n_obs_steps}-1'}
      pad_after: ${eval:'${n_action_steps}-1'}
      seed: 42
      val_ratio: 0.05

…and train with:
    python train.py --config-name=train_diffusion_unet_image_workspace task=xarm_image

To ablate overlay modes, run ``diffusion_policy.scripts.tactile_xarm_conversion``
once per mode and swap the ``zarr_path`` in the task YAML.

Schema expected at ``zarr_path`` (matches what tactile_xarm_conversion produces):
    /data/image        (N, 224, 224, 3) uint8     agentview, BGR
    /data/wrist_image  (N, 224, 224, 3) uint8     wrist, BGR
    /data/state        (N, 10) float32            xyz_mm + 6D rotation + grasp
    /data/action       (N, 10) float32            10-dim absolute next-state action
    /meta/episode_ends (E,)    int64
"""
from typing import Dict

import copy
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class XArmImageDataset(BaseImageDataset):
    """Two-camera xArm dataset for diffusion_policy.

    Mirrors the simple PushTImageDataset structure but with two image keys.
    State and action are both 10-dim (xyz + 6D rotation + grasp); they pass
    through to the normalizer without slicing — the policy sees the full
    state.
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int = None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["image", "wrist_image", "state", "action"]
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        # State + action go through the standard min/max normalizer; both
        # image keys reuse the canned [0, 255] -> [-1, 1] range normalizer.
        data = {
            "agent_pos": self.replay_buffer["state"],
            "action":    self.replay_buffer["action"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        normalizer["wrist_image"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample) -> Dict[str, np.ndarray]:
        # uint8 HWC -> float32 CHW [0,1]. get_image_range_normalizer above
        # then maps [0,1] -> [-1,1] inside the policy.
        image = np.moveaxis(sample["image"], -1, 1).astype(np.float32) / 255.0
        wrist = np.moveaxis(sample["wrist_image"], -1, 1).astype(np.float32) / 255.0
        return {
            "obs": {
                "image": image,                                          # (T, 3, 224, 224)
                "wrist_image": wrist,                                    # (T, 3, 224, 224)
                "agent_pos": sample["state"].astype(np.float32),         # (T, 10)
            },
            "action": sample["action"].astype(np.float32),               # (T, 10)
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
