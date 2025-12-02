import torch
import torchvision
from torchvision.transforms import v2
from transformers import pipeline, AutoTokenizer, CLIPTextModel

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
   # pooling is eithet average or concat 
   feats = [model(image) for image in images] 
   feats = torch.stack(feats, dim = 0)
   if pooling == "average":
      feats = feats.mean(dim=0)
   elif pooling == "concat":
      feats = feats.reshape(-1)
   
   return feats   

def get_text_features(text_encoder, tokenizer, actions, pooling):
   # pooling is either average or none
   text_inputs = tokenizer(actions, padding=True, return_tensors="pt")
   outputs = text_encoder(**text_inputs)

   hidden = outputs.last_hidden_state  
   cls_embed = hidden[:, 0, :]            

   if pooling == "average":
      return cls_embed.mean(dim=0)

   else:
      return cls_embed

def get_clip(model_name):
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   text_encoder = CLIPTextModel.from_pretrained(model_name)
   configuration = text_encoder.config
   return tokenizer, text_encoder, configuration

def get_dinov3_extractor(model_name):
   feature_extractor = pipeline(
      model=model_name,
      task="image-feature-extraction", 
   )
   return feature_extractor

clip_model_name = "openai/clip-vit-base-patch32" 
dinov3_model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

