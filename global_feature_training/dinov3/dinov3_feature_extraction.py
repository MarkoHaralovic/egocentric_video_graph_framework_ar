import torch
from torchvision.transforms import v2
from transformers import pipeline, AutoTokenizer, CLIPTextModel
import os
import tqdm
import cv2
import pandas as pd
import h5py
import numpy as np
from PIL import Image

def make_transform(resize_size: int = 256):
   to_tensor = v2.ToImage()
   resize = v2.Resize((resize_size, resize_size), antialias=True)
   to_float = v2.ToDtype(torch.float32, scale=True)
   normalize = v2.Normalize(
      mean=(0.485, 0.456, 0.406),
      std=(0.229, 0.224, 0.225),
   )
   return v2.Compose([to_tensor, resize, to_float, normalize])
 
def get_visual_features(model, images, pooling):
   # pooling is either average or concat
   pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
   
   feats = []
   for img in pil_images:
      result = model(img)
      if isinstance(result, list) and len(result) > 0:
         feat = torch.tensor(result[0])
      else:
         feat = torch.tensor(result)
      feats.append(feat)
   
   feats = torch.stack(feats, dim=0)
   
   if pooling == "average":
      feats = feats.mean(dim=0)
   elif pooling == "concat":
      feats = feats.reshape(-1)
   
   return feats   

def get_text_features(text_encoder, tokenizer, actions, pooling, device="cuda"):
   # pooling is either average or none
   text_inputs = tokenizer(actions, padding=True, return_tensors="pt").to(device)
   
   with torch.no_grad():
      outputs = text_encoder(**text_inputs)

   hidden = outputs.last_hidden_state  
   cls_embed = hidden[:, 0, :]            

   if pooling == "average":
      return cls_embed.mean(dim=0)
   else:
      return cls_embed

def get_clip_text_encoder(model_name, cache_dir=None):
   tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      cache_dir=cache_dir,
      local_files_only=True if cache_dir else False
   )
   text_encoder = CLIPTextModel.from_pretrained(
      model_name,
      cache_dir=cache_dir,
      local_files_only=True if cache_dir else False
   )
   configuration = text_encoder.config
   return tokenizer, text_encoder, configuration

def get_dinov3_extractor(model_name_or_path, cache_dir=None):
   if os.path.exists(model_name_or_path):
      feature_extractor = pipeline(
         model=model_name_or_path,
         task="image-feature-extraction"
      )
   else:
      feature_extractor = pipeline(
         model=model_name_or_path,
         task="image-feature-extraction",
         model_kwargs={
            "cache_dir": cache_dir,
            "local_files_only": True if cache_dir else False
         }
      )
   return feature_extractor

def process_folder(vision_backbone, text_backbone, tokenizer, input_folder_path, model_name, pooling="average", output_path="./", device="cuda"):
   clips = [clip for clip in os.listdir(input_folder_path)]
   text_backbone = text_backbone.to(device)
   text_backbone.eval()
   
   for clip in tqdm.tqdm(clips, desc=f"Extracting features from {input_folder_path}"):
      frames_folder = os.path.join(input_folder_path, clip, "frames")
      if not os.path.exists(frames_folder):
         print(f"Skipping {clip}: frames folder not found")
         continue
         
      image_filenames = sorted([img for img in os.listdir(frames_folder) if img.endswith(".jpg")])
      rgb_images = [cv2.cvtColor(cv2.imread(os.path.join(frames_folder, img)), cv2.COLOR_BGR2RGB) for img in image_filenames]
      
      with open(os.path.join(input_folder_path, clip, "actions.txt"), "r") as f:
         actions_list = [line.strip() for line in f.readlines()]
      
      with open(os.path.join(input_folder_path, clip, "activities.txt"), "r") as f:
         activities_list = [line.strip() for line in f.readlines()]
      
      annotations = pd.read_csv(os.path.join(input_folder_path, clip, "annotations.csv"))
      
      activity_blocks_ids = sorted(set(annotations["activity_block_id"]))
      
      visual_feats = []
      text_feats = []
      activity_labels = []
      gaze_labels = []
      
      clip_output_path = os.path.join(output_path, clip)
      os.makedirs(clip_output_path, exist_ok=True)
      h5_path = os.path.join(clip_output_path, f"features_{model_name}.h5")
      
      max_len = None
      for activity_block in tqdm.tqdm(activity_blocks_ids, f"Processing {clip}"):
         data_block = annotations[annotations["activity_block_id"] == activity_block]
         frame_idxs = data_block["frame_index"].tolist()
         
         frames = [rgb_images[idx] for idx in frame_idxs]
         if max_len is None:
            max_len = len(frames)
         
         actions = data_block["action"].tolist()
         activity = data_block["activity"].iloc[0]
         gaze_x = data_block["gaze_x"].to_numpy()
         gaze_y = data_block["gaze_y"].to_numpy()
         gazes = np.stack([gaze_x, gaze_y], axis=1)
         
         image_features = get_visual_features(vision_backbone, frames, pooling)
         text_features = get_text_features(text_backbone, tokenizer, actions, pooling, device)
         
         if isinstance(image_features, torch.Tensor):
            image_features = image_features.detach().cpu().numpy()
         if isinstance(text_features, torch.Tensor):
            text_features = text_features.detach().cpu().numpy()
         
         if len(frames) < max_len:
            pad_count = max_len - len(frames)
            
            if pooling == "average":
               pass
            else:
               last_visual = image_features[-1:] if image_features.ndim > 1 else image_features
               last_text = text_features[-1:] if text_features.ndim > 1 else text_features
               last_gaze = gazes[-1:]
               
               image_features = np.concatenate([image_features] + [last_visual] * pad_count, axis=0)
               text_features = np.concatenate([text_features] + [last_text] * pad_count, axis=0)
               gazes = np.concatenate([gazes] + [last_gaze] * pad_count, axis=0)
            
         visual_feats.append(image_features)
         text_feats.append(text_features)
         activity_labels.append(activity)
         gaze_labels.append(gazes)
            
      visual_feats = np.stack(visual_feats, axis=0)
      text_feats = np.stack(text_feats, axis=0)
      activity_labels = np.array([str(a) for a in activity_labels], dtype=h5py.string_dtype(encoding='utf-8'))
      gaze_feats = np.stack(gaze_labels, axis=0)

      with h5py.File(h5_path, "w") as f:
         f.create_dataset("visual_features", data=visual_feats, dtype="float32")
         f.create_dataset("text_features", data=text_feats, dtype="float32")
         f.create_dataset("activity_labels", data=activity_labels)
         f.create_dataset("gaze_labels", data=gaze_feats)
         
     
POOLING="average"   
MODEL_CACHE_DIR = "/home/s3758869/models"
clip_model_path = "/home/s3758869/models/clip-vit-base-patch32"
dinov3_model_path = "/home/s3758869/models/dinov3-vith16plus-pretrain-lvd1689m"
dinov3_model_name_ab = "dinov3h16+"
INPUT_DATA_FOLDER = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-mistral-7b-hf"
OUTPUT_DATA_FOLDER = f"/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-mistral-7b-hf_features_{dinov3_model_name_ab}_pooling_{POOLING}"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

vision_backbone = get_dinov3_extractor(dinov3_model_path, cache_dir=None)
tokenizer, text_backbone, _ = get_clip_text_encoder(clip_model_path, cache_dir=None)

os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)

process_folder(
   vision_backbone, 
   text_backbone, 
   tokenizer, 
   INPUT_DATA_FOLDER, 
   dinov3_model_name_ab, 
   POOLING,
   OUTPUT_DATA_FOLDER,
   device
)
