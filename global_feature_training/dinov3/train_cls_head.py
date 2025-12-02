import torch
import torchvision
from torchvision.transforms import v2
from transformers import pipeline, AutoTokenizer, CLIPTextModel
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=8,        # each batch = 8 sequences → (8, 10, C, H, W)
    shuffle=True,
    num_workers=4,
    pin_memory=True
)