import torch
import torch.nn as nn
from torch.nn import init
from typing import Dict,List

class AttentionPooling(nn.Module):
   def __init__(self, in_features, out_features,hidden_features=128):
      super().__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.hidden_features = hidden_features
      
      self.score = nn.Sequential(
         nn.Linear(self.in_features, self.hidden_features),
         nn.Tanh(),
         nn.Linear(self.hidden_features, 1),
      )
      
      self.proj = nn.Linear(self.in_features, self.out_features)
      
   def forward(self, x: torch.Tensor) :
      if x is None or x.shape[0] == 0:
         return torch.zeros(self.out_features, device=x.device if x is not None else None)
      scores = self.score(x)
      w = torch.softmax(scores, dim=0)
      pooled = (w*x).sum(dim=0)
      return self.proj(pooled)
      
      
class MultiQueryPooling(nn.Module):
   def __init__(self, in_features, out_features,hidden_features=128, k=4):
      super().__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.hidden_features = hidden_features
      self.k = k
      
      self.queries = nn.Parameter(torch.randn(k, self.in_features))
      self.proj = nn.Linear(self.in_features, self.out_features)

   def forward(self, x: torch.Tensor) :
      if x.shape[0] == 0:
            return torch.zeros(self.k * self.out_features, device=x.device, dtype=x.dtype)
      scores = (self.queries @ x.T) / (self.in_features ** 0.5)
      attn =  torch.softmax(scores, dim=-1)
      pooled = attn@x
      projs = self.proj(pooled)
      return projs.reshape(-1)
   
class Embedder(nn.Module):
   def __init__(self, num_instances: int, emb_dim: int = 32):
      super().__init__()
      self.emb = nn.Embedding(num_instances, emb_dim)

   def forward(self, idx: torch.Tensor) -> torch.Tensor:
      idx = idx.long()
      if idx.ndim == 1 and idx.numel() == 1:
         idx = idx.squeeze(0)        
      elif idx.ndim == 2 and idx.size(-1) == 1:
         idx = idx.squeeze(-1)   
      return self.emb(idx)  
     
class ActionGraphEmbedding(nn.Module):
   def __init__(
      self,
      num_verbs,num_objects,num_rels,num_attrs,clip_dim,obj_feat_dim,
      emb_dim=32,obj_out=128,aux_out=32,attr_out=32,rel_out=32,trip_out=32,
      k_obj=4,k_aux=2,k_trip=2,use_triplets=True,device="cuda"
   ):
      super().__init__()
      self.device = device
      self.use_triplets = use_triplets

      self.verb_emb = Embedder(num_verbs, emb_dim)
      self.obj_emb = Embedder(num_objects, emb_dim)
      self.rel_emb = Embedder(num_rels, emb_dim)

      self.obj_pool = MultiQueryPooling(obj_feat_dim + emb_dim, obj_out, k=k_obj)
      self.aux_pool = MultiQueryPooling(emb_dim, aux_out, k=k_aux)

      self.attr_proj = nn.Linear(num_attrs * 2, attr_out)
      self.rel_proj = nn.Linear(num_rels * 2, rel_out)

      if use_triplets:
         self.trip_mlp = nn.Sequential(
               nn.Linear(emb_dim * 3, emb_dim),
               nn.ReLU(),
               nn.Linear(emb_dim, emb_dim),
         )
         self.trip_pool = MultiQueryPooling(emb_dim, trip_out, k=k_trip)

      self.clip_proj = nn.Linear(clip_dim, clip_dim)

      self.out_dim = (clip_dim + emb_dim + k_aux * aux_out + k_obj * obj_out + attr_out + rel_out + (k_trip * trip_out if use_triplets else 0))

   def forward(self, g: Dict[str, torch.Tensor]) -> torch.Tensor:
      clip = torch.from_numpy(g["clip_feat"])
      
      v = self.verb_emb(g["verb_idx"])
      aux_idx = g.get("aux_verb_idx", None)
      if aux_idx is None or aux_idx.numel() == 0:
         aux_tokens = torch.zeros((0, v.shape[0]), device=self.device)
      else:
         aux_tokens = self.verb_emb(aux_idx)
         if aux_tokens.ndim == 1: aux_tokens = aux_tokens.unsqueeze(0)
      aux_vec = self.aux_pool(aux_tokens)

      obj_feats = g["obj_feats"]
      if obj_feats.shape[0] == 0:
         obj_tokens = torch.zeros(
               (0, obj_feats.shape[1] + self.obj_emb.emb.embedding_dim),
               device=self.device,
               dtype=obj_feats.dtype,
         )
      else:
         obj_ids = self.obj_emb(g["obj_indices"])
         if obj_ids.ndim == 1:
               obj_ids = obj_ids.unsqueeze(0)
         obj_tokens = torch.cat([obj_feats, obj_ids.to(obj_feats.dtype)], dim=-1)
      obj_vec = self.obj_pool(obj_tokens)

      attr_vecs = g["attr_vecs"]
      attr_sum = attr_vecs.sum(dim=0) if attr_vecs.shape[0] > 0 else torch.zeros(attr_vecs.shape[1], device=self.device)
      attr_emb = self.attr_proj(torch.cat([attr_sum, torch.log1p(attr_sum)], dim=0))

      rels_vecs = g["rels_vecs"]
      rel_sum = rels_vecs.sum(dim=0) if rels_vecs.shape[0] > 0 else torch.zeros(rels_vecs.shape[1], device=self.device)
      rel_emb = self.rel_proj(torch.cat([rel_sum, torch.log1p(rel_sum)], dim=0))

      if self.use_triplets:
         trip = g["triplets"]
         if trip.shape[0] == 0:
               trip_tokens = torch.zeros((0, v.shape[0]), device=self.device)
         else:
               t = torch.cat(
                  [
                     self.verb_emb(trip[:, 0]),
                     self.obj_emb(trip[:, 1]),
                     self.rel_emb(trip[:, 2]),
                  ],
                  dim=-1,
               )
               
               trip_tokens = self.trip_mlp(t)
         trip_vec = self.trip_pool(trip_tokens)

      parts = [
         self.clip_proj(clip),
         v.to(clip.dtype),
         aux_vec.to(clip.dtype),
         obj_vec.to(clip.dtype),
         attr_emb.to(clip.dtype),
         rel_emb.to(clip.dtype),
      ]
      if self.use_triplets:
         parts.append(trip_vec.to(clip.dtype))

      return torch.cat(parts, dim=-1)

class GraphMLP(nn.Module):
   def __init__(self,num_graphs,num_verbs,num_objects,num_rels,num_attrs,clip_dim,obj_feat_dim, n_classes, fc_layers_num, device="cuda"):
      super().__init__()
      self.device=device
      self.num_graphs = num_graphs
      self.action_graph_embedder = ActionGraphEmbedding(num_verbs,num_objects,num_rels,num_attrs,clip_dim,obj_feat_dim,device=device)
      self.input_dim = self.action_graph_embedder.out_dim *  num_graphs
      self.fc_layers_num = fc_layers_num
      self.n_classes =  n_classes
      
      if self.fc_layers_num == 1 :
         fc = nn.Linear(self.input_dim, self.n_classes)
         init.xavier_uniform_(fc.weight)
         init.zeros_(fc.bias)
      else:
         layers = []
         for _ in range(self.fc_layers_num - 1):
            layers.append(nn.Linear(self.input_dim, self.input_dim))
            layers.append(nn.ReLU())
         layers.append(nn.Linear(self.input_dim, self.n_classes))
         fc = nn.Sequential(*layers)

      for layer in fc:
         if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
             
      self.__setattr__(f"head", fc)
      self.head = fc  
      
   def forward(self,sequence_graphs : List):
      graph_embs = [self.action_graph_embedder(graph.to_easg_tensors()) for graph in sequence_graphs]
      if len(sequence_graphs) <self.num_graphs:
         for _ in range(self.num_graphs - len(sequence_graphs)):
            graph_embs.append(graph_embs[-1])
      graph_embs = torch.stack(graph_embs,dim=0).reshape(-1)
      return self.head(graph_embs)
