import torch
from torchvision.transforms import v2
from transformers import AutoTokenizer, CLIPTextModel
from TimeSformer.timesformer.models.vit import TimeSformer
import os
import tqdm
import cv2
import pandas as pd
import h5py
import numpy as np
from PIL import Image
import torchvision.transforms as T
import random
 
def sample_frames_and_labels(frames, actions, gazes, num_frames):
   total_frames = len(frames)
   
   if total_frames >= num_frames:
      sampled_indices = sorted(random.sample(range(total_frames), num_frames))
   else:
      sampled_indices = list(range(total_frames))
      sampled_indices.extend([total_frames - 1] * (num_frames - total_frames))
   
   sampled_frames = [frames[idx] for idx in sampled_indices]
   sampled_actions = [actions[idx] for idx in sampled_indices]
   sampled_gazes = gazes[sampled_indices]
   
   return sampled_frames, sampled_actions, sampled_gazes

def get_visual_features(model, images, resize_size=224, device="cuda"):
   to_tensor = v2.ToImage()
   resize = v2.Resize((resize_size, resize_size), antialias=True)
   to_float = v2.ToDtype(torch.float32, scale=True)
   normalize = v2.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
   )
   transform = v2.Compose([to_tensor, resize, to_float, normalize])
   pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
   frames = [transform(img) for img in pil_images] 
   frames = torch.stack(frames, dim=0)
   frames = frames.permute(1, 0, 2, 3) #cfhw
   frames = frames.unsqueeze(0).to(device) #bcfhw
   
   with torch.no_grad():
      feats = model(frames)
   return feats   

def get_text_features(text_encoder, tokenizer, actions, device="cuda"):
   text_inputs = tokenizer(actions, padding=True, return_tensors="pt").to(device)
   
   with torch.no_grad():
      outputs = text_encoder(**text_inputs)

   hidden = outputs.last_hidden_state  
   cls_embed = hidden[:, 0, :]            

   return cls_embed.mean(dim=0)


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

def get_timesformer_visual_backbone(img_size=224, num_classes=600, num_frames=8, attention_type='divided_space_time', pretrained_model = 'path'):
   model = TimeSformer(
      img_size=img_size, 
      num_classes=num_classes, 
      num_frames=num_frames, 
      attention_type=attention_type,  
      pretrained_model=pretrained_model
   )
   return model

def process_folder(vision_backbone, text_backbone, tokenizer, input_folder_path, model_name, num_frames=8, device="cuda"):
   clips = [clip for clip in os.listdir(input_folder_path)]
   text_backbone = text_backbone.to(device)
   text_backbone.eval()
   vision_backbone = vision_backbone.to(device)
   vision_backbone.eval()
   
   for clip in tqdm.tqdm(clips, desc=f"Extracting features from {input_folder_path}"):
      frames_folder = os.path.join(input_folder_path, clip, "frames")
      if not os.path.exists(frames_folder):
         print(f"Skipping {clip}: frames folder not found")
         continue
      clip_output_path = os.path.join(input_folder_path, clip)
      h5_path = os.path.join(clip_output_path, f"activity_features_model_{model_name}_numframes_{num_frames}.h5")
      
      if os.path.exist(h5_path):
         continue
      image_filenames = sorted([img for img in os.listdir(frames_folder) if img.endswith(".jpg")])
      rgb_images = [cv2.cvtColor(cv2.imread(os.path.join(frames_folder, img)), cv2.COLOR_BGR2RGB) for img in image_filenames]
      
      annotations = pd.read_csv(os.path.join(input_folder_path, clip, "annotations.csv"))
      
      activity_blocks_ids = sorted(set(annotations["activity_block_id"]))
      
      visual_feats = []
      text_feats = []
      activity_labels = []
      gaze_labels = []
      
      for activity_block in activity_blocks_ids:
         data_block = annotations[annotations["activity_block_id"] == activity_block]
         frame_idxs = data_block["frame_index"].tolist()
         
         frames = [rgb_images[idx] for idx in frame_idxs]
         actions = data_block["action"].tolist()
         activity = data_block["activity"].iloc[0]
         gaze_x = data_block["gaze_x"].to_numpy()
         gaze_y = data_block["gaze_y"].to_numpy()
         gazes = np.stack([gaze_x, gaze_y], axis=1)
         
         sampled_frames, sampled_actions, sampled_gazes = sample_frames_and_labels(frames, actions, gazes, num_frames)
      
         image_features = get_visual_features(vision_backbone, sampled_frames, device=device)
         text_features = get_text_features(text_backbone, tokenizer, sampled_actions, device)
         
         if isinstance(image_features, torch.Tensor):
            image_features = image_features.detach().cpu().numpy()
         if isinstance(text_features, torch.Tensor):
            text_features = text_features.detach().cpu().numpy()
            
         visual_feats.append(image_features)
         text_feats.append(text_features)
         activity_labels.append(activity)
         gaze_labels.append(sampled_gazes)
            
      visual_feats = np.stack(visual_feats, axis=0)
      text_feats = np.stack(text_feats, axis=0)
      activity_labels = np.array([str(a) for a in activity_labels], dtype=h5py.string_dtype(encoding='utf-8'))
      gaze_feats = np.stack(gaze_labels, axis=0)

      with h5py.File(h5_path, "w") as f:
         f.create_dataset("visual_features", data=visual_feats, dtype="float32")
         f.create_dataset("text_features", data=text_feats, dtype="float32")
         f.create_dataset("activity_labels", data=activity_labels)
         f.create_dataset("gaze_labels", data=gaze_feats)
         
     
MODEL_CACHE_DIR = "/home/s3758869/models"
clip_model_name = "/home/s3758869/models/clip-vit-base-patch32"
timesformer_model_path = "/home/s3758869/egocentric_video_graph_framework_ar/global_feature_training/timesformer/TimeSformer/checkpoints/TimeSformer_divST_8x32_224_K600.pyth"
timesformer_model_name_ab = "timesformer_k600_8fr_224"
INPUT_DATA_FOLDER = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-mistral-7b-hf"
NUM_FRAMES = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

vision_backbone = get_timesformer_visual_backbone(
   img_size=224,
   num_classes=600,
   num_frames=NUM_FRAMES,
   attention_type='divided_space_time',
   pretrained_model=timesformer_model_path
)
tokenizer, text_backbone, _ = get_clip_text_encoder(clip_model_name, cache_dir=MODEL_CACHE_DIR)

process_folder(
   vision_backbone, 
   text_backbone, 
   tokenizer, 
   INPUT_DATA_FOLDER, 
   timesformer_model_name_ab,
   num_frames=NUM_FRAMES,
   device=device
)
