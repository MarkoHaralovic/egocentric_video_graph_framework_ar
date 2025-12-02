import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from ..base_model.sequence_model import ImageSequenceClassificator
from ..dinov3.dinov3_feature_extraction import (
    get_clip_text_encoder,
    get_dinov3_extractor,
)
from ..data_loading.sequence_dataset import SequenceDataset, feature_collate_fn

USE_PRECOMPUTED_FEATURES = True
INPUT_FOLDER = ""
dinov3_model_name = "dinov3h16+"
dinov3_model_path = ""
clip_model_path = ""
dino_cache_dir = ""
clip_cache_dir = ""

dataset = SequenceDataset(
    input_folder = INPUT_FOLDER,
    model_name = dinov3_model_name, 
    load_viusal=True,
    load_text=False,
    activity_to_idx = None
)
cls_mapping = dataset.activity_to_idx

train_loader = DataLoader(
    dataset,
    batch_size=2,        # each batch = (BS, SEQUENCE_SIZE, C, H, W) - temporal dimension similar to video
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    collate_fn=feature_collate_fn
)

if not USE_PRECOMPUTED_FEATURES:
    dino_backbone = get_dinov3_extractor(dinov3_model_name)
    clip_text_encoder = get_clip_text_encoder(clip_model_path, clip_cache_dir)

model  = ImageSequenceClassificator(
    vision_backbone=get_dinov3_extractor(dinov3_model_path, cache_dir=dino_cache_dir),
    n_classes = len(cls_mapping),
    fc_layers_num=5,
    use_precomputed_features=USE_PRECOMPUTED_FEATURES,
    transforms=None,
    device="cuda",
    text_backbone=get_clip_text_encoder(clip_model_path, clip_cache_dir)
    
)
