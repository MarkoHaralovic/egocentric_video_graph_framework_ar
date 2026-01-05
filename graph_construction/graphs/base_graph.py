from dataclasses import dataclass
from typing import Dict, Optional
import torch

@dataclass
class Node:
   node_id: int
   label: str          # "camera_wearer", "drop", "screwdriver"
   node_type: str      # "camera_wearer", "verb", "object", "aux_verb"
   idx: Optional[int]  
   feat: Optional[torch.Tensor]  # clip_feat for verb, obj_feat for object
   attr : Optional[str] # attribute for the given object
   attr_feat : Optional[torch.Tensor] # this can be tex embeddings from clip, but i am using multi_hot representation
   related_to_verb: Optional[int] = None  # verb_idx this object is related to

@dataclass
class Edge:
   src: int
   dst: int
   rel_idx: Optional[int]    
   rel_label: str             #"agent", "direct_obj", "on", "with"
 
class BaseGraph:
   def __init__(self, verbs: Dict, objs: Dict, rels: Dict, attrs = Dict):
      super(BaseGraph).__init__()
      self.verbs = verbs
      self.objs = objs
      self.rels = rels # must contain direct_object and aux_direct_object
      self.attrs = attrs
      self.node_types = ["camera_wearer", "verb", "object", "aux_verb"]

      self.id_to_verb = {v:k for k,v in self.verbs.items()}
      self.id_to_objs = {v:k for k,v in self.objs.items()}
      self.id_to_rels = {v:k for k,v in self.rels.items()}
      self.id_to_attrs= {v:k for k,v in self.attrs.items()}
      
      # camera_wearer -> verb: agent
      self.special_rels = ["agent", "same_as"]  
      self.all_edge_labels = self.special_rels + list(rels.keys())

   def new_camera_wearer_node(self, node_id: int) -> Node:
      return Node(
         node_id=node_id,
         label="camera_wearer",
         node_type="camera_wearer",
         idx=None,
         feat=None,
         attr=None,
         attr_feat=None
      )


   def new_main_verb_node(self, node_id: int, verb : str, clip_feat: torch.Tensor) -> Node:
      return Node(
         node_id=node_id,
         label=verb,
         node_type="verb",
         idx=self.verbs[verb],
         feat=clip_feat,   # cframe wise feature
         attr=None,
         attr_feat=None         
      )
      
   def new_aux_verb_node(self, node_id: int, verb : str, clip_feat: torch.Tensor) -> Node:
      return Node(
         node_id=node_id,
         label=verb,
         node_type="aux_verb",
         idx=self.verbs[verb],
         feat=None,
         attr=None,
         attr_feat=None 
      )   
      
   def new_object_node(self, node_id: int, obj_idx: int, obj_feat: torch.Tensor, attr : str, attr_feat : torch.Tensor, verb_idx: int = None) -> Node:
      return Node(
         node_id=node_id,
         label=self.id_to_objs[obj_idx],
         node_type="object",
         idx=obj_idx,
         feat=obj_feat,    # object feature
         attr=attr,
         attr_feat=attr_feat,
         related_to_verb=verb_idx
      )
      
   def agent_edge(self, camera_wearer_id: int, verb_id: int) -> Edge:
      return Edge(
         src=camera_wearer_id,
         dst=verb_id,
         rel_idx=None,
         rel_label="agent",
      )

   def rel_edge(self, verb_id: int, obj_id: int, rel_idx: int) -> Edge:
      return Edge(
         src=verb_id,
         dst=obj_id,
         rel_idx=rel_idx,
         rel_label=self.id_to_rels[rel_idx],
      )

   def temporal_edge(self, src_obj_id: int, dst_obj_id: int) -> Edge:
      return Edge(
         src=src_obj_id, 
         dst=dst_obj_id, 
         rel_idx=None, 
         rel_label="same_as"
      )