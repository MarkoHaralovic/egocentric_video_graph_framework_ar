import torch
from torch.utils.data import Dataset
from PIL import Image


class SequenceDataset(Dataset):
   def __init__(self, sequences, action_labels, activity_labels, transform=None):
      assert len(sequences) == len(action_labels) == len(activity_labels), \
         "sequences, action_labels and activity_labels must have same length"

      self.sequences = sequences
      self.action_labels = torch.as_tensor(action_labels, dtype=torch.long)
      self.activity_labels = torch.as_tensor(activity_labels, dtype=torch.long)
      self.transform = transform
      
   def __len__(self):
      return len(self.sequences)

   def __getitem__(self, idx):
      img_paths = self.sequences[idx]  

      imgs = []
      for p in img_paths:
         img = Image.open(p).convert("RGB")
         if self.transform is not None:
               img = self.transform(img)
         imgs.append(img)

      images = torch.stack(imgs, dim=0)

      action_label = self.action_labels[idx]  
      activity_label = self.activity_labels[idx] 

      return {
         "images": images,                     # (N, C, H, W)
         "action_label": action_label,         # (N)
         "activity_label": activity_label,     # (1)
      }
