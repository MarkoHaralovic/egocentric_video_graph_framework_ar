import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        input_folder,
        clip_names,
        samples=None,
        model_name="dinov3h16+",
        pooling=None,
        frames=None,
        load_visual=True,
        load_text=False,
        activity_to_idx=None,
    ):
        self.input_folder = input_folder
        self.model_name = model_name
        self.pooling = pooling
        self.frames = frames
        self.load_visual = load_visual
        self.load_text = load_text

        if not load_visual and not load_text:
            raise ValueError("At least one of load_visual or load_text must be True")

        if samples is not None:
            self._init_from_samples(samples, activity_to_idx)
        else:
            self._init_from_clips(clip_names, activity_to_idx)

        print(
            f"Loaded {len(self.h5_paths)} clips with {len(self.sample_index)} activity blocks"
        )
        print(f"Activity vocabulary size: {len(self.activity_to_idx)}")

    def _init_from_samples(self, samples, activity_to_idx):
        self.samples = samples
        self.h5_paths = sorted(list({s[1] for s in samples}))
        self.h5_to_file_idx = {p: i for i, p in enumerate(self.h5_paths)}
        self.clip_names = [None] * len(self.h5_paths)

        for clip_name, h5_path, _, _ in samples:
            self.clip_names[self.h5_to_file_idx[h5_path]] = clip_name

        self.activity_to_idx = activity_to_idx or {
            a: i for i, a in enumerate(sorted({s[3] for s in samples}))
        }
        self.idx_to_activity = {v: k for k, v in self.activity_to_idx.items()}
        self.sample_index = [(self.h5_to_file_idx[s[1]], s[2], s[3]) for s in samples]

    def _init_from_clips(self, clip_names, activity_to_idx):
        self.h5_paths = []
        self.clip_names = []

        for clip_name in clip_names:
            if self.pooling is not None:
                h5_path = os.path.join(
                    self.input_folder,
                    clip_name,
                    f"activity_features_model_{self.model_name}_pooling_{self.pooling}.h5",
                )
            elif self.frames is not None:
                h5_path = os.path.join(
                    self.input_folder,
                    clip_name,
                    f"activity_features_model_{self.model_name}_numframes_{self.frames}.h5",
                )

            if os.path.exists(h5_path):
                self.h5_paths.append(h5_path)
                self.clip_names.append(clip_name)

        if not self.h5_paths:
            raise ValueError(f"No h5 files found matching pattern")

        self.activity_to_idx = activity_to_idx or self._build_activity_vocab()
        self.idx_to_activity = {v: k for k, v in self.activity_to_idx.items()}

        self.sample_index = []
        for file_idx, h5_path in enumerate(self.h5_paths):
            with h5py.File(h5_path, "r") as f:
                for block_idx in range(f["activity_labels"].shape[0]):
                    label = f["activity_labels"][block_idx]
                    label = label.decode("utf-8") if isinstance(label, bytes) else label
                    if label in self.activity_to_idx:
                        self.sample_index.append((file_idx, block_idx, label))

    def _build_activity_vocab(self):
        activities = set()
        for h5_path in self.h5_paths:
            with h5py.File(h5_path, "r") as f:
                for l in f["activity_labels"][:]:
                    activities.add(l.decode("utf-8") if isinstance(l, bytes) else l)
        return {act: idx for idx, act in enumerate(sorted(activities))}

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        file_idx, block_idx, label_str = self.sample_index[idx]
        output = {"clip_name": self.clip_names[file_idx], "block_idx": block_idx}

        with h5py.File(self.h5_paths[file_idx], "r") as f:
            if self.load_visual:
                output["visual_features"] = torch.from_numpy(
                    f["visual_features"][block_idx].astype(np.float32)
                )
            if self.load_text:
                output["text_features"] = torch.from_numpy(
                    f["text_features"][block_idx].astype(np.float32)
                )

            output["activity_label"] = torch.tensor(
                self.activity_to_idx[label_str], dtype=torch.long
            )
            output["activity_name"] = label_str

        return output


def feature_collate_fn(batch):
    output = {}

    for key in ["visual_features", "text_features"]:
        if key in batch[0]:
            feats = [item[key] for item in batch]
            output[key] = (
                torch.stack(feats)
                if all(f.shape == feats[0].shape for f in feats)
                else feats
            )

    output["activity_label"] = torch.stack([item["activity_label"] for item in batch])
    output["activity_name"] = [item["activity_name"] for item in batch]
    output["clip_name"] = [item["clip_name"] for item in batch]
    output["block_idx"] = [item["block_idx"] for item in batch]

    return output
