import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np


class SequenceDataset(Dataset):
   def __init__(
      self, 
      input_folder, 
      clip_names,
      model_name="dinov3h16+",
      pooling="concat",
      load_visual=True, 
      load_text=False,
      activity_to_idx=None
   ):
      self.input_folder = input_folder
      self.model_name = model_name
      self.pooling = pooling
      self.load_visual = load_visual
      self.load_text = load_text
      
      if not load_visual and not load_text:
         raise ValueError("At least one of load_visual or load_text must be True")
      
      self.h5_paths = []
      self.clip_names = []
      
      for clip_name in clip_names:
         clip_path = os.path.join(input_folder, clip_name)
         if not os.path.isdir(clip_path):
            continue
            
         h5_path = os.path.join(clip_path, f"activity_features_model_{model_name}_pooling_{pooling}.h5")
         if os.path.exists(h5_path):
            self.h5_paths.append(h5_path)
            self.clip_names.append(clip_name)
      
      if len(self.h5_paths) == 0:
         raise ValueError(f"No h5 files found in {input_folder} with pattern activity_features_model_{model_name}_pooling_{pooling}.h5")
      
      if activity_to_idx is None:
         self.activity_to_idx = self._build_activity_vocab()
      else:
         self.activity_to_idx = activity_to_idx
      
      self.idx_to_activity = {v: k for k, v in self.activity_to_idx.items()}
      
      self.sample_index = []
      for file_idx, h5_path in enumerate(self.h5_paths):
         with h5py.File(h5_path, 'r') as f:
            num_blocks = f['activity_labels'].shape[0]
            for block_idx in range(num_blocks):
               self.sample_index.append((file_idx, block_idx))
      
      print(f"Loaded {len(self.h5_paths)} clips with {len(self.sample_index)} activity blocks")
      print(f"Activity vocabulary size: {len(self.activity_to_idx)}")
      
   def _build_activity_vocab(self):
      activities = set()
      for h5_path in self.h5_paths:
         with h5py.File(h5_path, 'r') as f:
            labels = f['activity_labels'][:]
            if labels.dtype.kind == 'S':  
               labels = [l.decode('utf-8') if isinstance(l, bytes) else l for l in labels]
            activities.update(labels)
      
      activities = sorted(activities)
      return {act: idx for idx, act in enumerate(activities)}
   
   def __len__(self):
      return len(self.sample_index)
   
   def __getitem__(self, idx):
      file_idx, block_idx = self.sample_index[idx]
      h5_path = self.h5_paths[file_idx]
      
      output = {}
      
      with h5py.File(h5_path, 'r') as f:
         # Load visual features
         if self.load_visual:
            visual_features = torch.from_numpy(f['visual_features'][block_idx].astype(np.float32))
            output['visual_features'] = visual_features
         
         # Load text features
         if self.load_text:
            text_features = torch.from_numpy(f['text_features'][block_idx].astype(np.float32))
            output['text_features'] = text_features
         
         # Load activity label
         activity_label_raw = f['activity_labels'][block_idx]
         if isinstance(activity_label_raw, bytes):
            activity_label_raw = activity_label_raw.decode('utf-8')
         
         if activity_label_raw not in self.activity_to_idx:
            raise ValueError(f"Activity '{activity_label_raw}' not found in activity_to_idx mapping. "
                           f"Available activities: {list(self.activity_to_idx.keys())[:10]}...")
         
         activity_label = self.activity_to_idx[activity_label_raw]
         output['activity_label'] = torch.tensor(activity_label, dtype=torch.long)
         output['activity_name'] = activity_label_raw
      
      output['clip_name'] = self.clip_names[file_idx]
      output['block_idx'] = block_idx
      
      return output


def feature_collate_fn(batch):
   output = {}
   
   if 'visual_features' in batch[0]:
      visual_feats = [item['visual_features'] for item in batch]
      if all(v.shape == visual_feats[0].shape for v in visual_feats):
         output['visual_features'] = torch.stack(visual_feats)
      else:
         output['visual_features'] = visual_feats 
   
   if 'text_features' in batch[0]:
      text_feats = [item['text_features'] for item in batch]
      if all(t.shape == text_feats[0].shape for t in text_feats):
         output['text_features'] = torch.stack(text_feats)
      else:
         output['text_features'] = text_feats  
   
   output['activity_label'] = torch.stack([item['activity_label'] for item in batch])
   output['activity_name'] = [item['activity_name'] for item in batch]
   output['clip_name'] = [item['clip_name'] for item in batch]
   output['block_idx'] = [item['block_idx'] for item in batch]
   
   return output
