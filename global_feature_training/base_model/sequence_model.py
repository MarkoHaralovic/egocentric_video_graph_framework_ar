import torch
from torch import nn as nn
from torch.nn import init

class ImageSequenceClassificator(nn.Module):
   def __init__(
      self,
      vision_backbone,
      n_classes,
      feature_map_depth,
      linear_layer_input_dim,
      fc_layers_num,
      use_precomputed_features,
      transforms,
      device="cuda",
      text_backbone=None
    ):
      super(ImageSequenceClassificator, self).__init__()

      self.vision_backbone = vision_backbone
      self.text_backbone = text_backbone
      self.n_classes = n_classes
      self.feature_map_depth = feature_map_depth
      self.linear_layer_input_dim = linear_layer_input_dim 
      self.fc_layers_num = fc_layers_num
      self.device = device
      self.use_precomputed_features = use_precomputed_features
      self.transforms = transforms

      if self.fc_layers_num == 1 :
         fc = nn.Linear(self.linear_layer_input_dim, self.n_classes)
      else:
         layers = []
         for _ in range(self.fc_layers_num - 1):
            layers.append(nn.Linear(self.linear_layer_input_dim, self.linear_layer_input_dim))
            layers.append(nn.ReLU())
         layers.append(nn.Linear(self.linear_layer_input_dim, self.n_classes))
         fc = nn.Sequential(*layers)

      if isinstance(fc, nn.Sequential):
         for layer in fc:
            if isinstance(layer, nn.Linear):
               init.xavier_uniform_(layer.weight)
               init.zeros_(layer.bias)
      else:
         init.xavier_uniform_(fc.weight)
         init.zeros_(fc.bias)

      self.__setattr__(f"head", fc)
      self.head = fc

   def forward(self, x_img, x_text=None):
      if not self.use_precomputed_features :
         x_img = self.transforms(x_img)
         x_img = self.vision_backbone(x_img)
         
         if x_text is not None and self.text_backbone is not None:
            x_text = self.text_backbone(x_text)
         
      
      if x_text is not None:
         x = torch.concat([x_img, x_text], dim=0)
      else:
         x = x_img

      output = self.head(x)
      return output
   