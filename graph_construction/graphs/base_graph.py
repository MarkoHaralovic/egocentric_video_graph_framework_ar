from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class Node:
   node_id: int
   label: str          # "camera_wearer", "drop", "screwdriver"
   node_type: str      # "camera_wearer", "verb", "object"
   idx: Optional[int]  
   feat: Optional[torch.Tensor]  # clip_feat for verb, obj_feat for object


@dataclass
class Edge:
   src: int
   dst: int
   rel_idx: Optional[int]     # index in rels list, or None (e.g. camera_wearer->verb)
   rel_label: str             #"agent", "direct_obj", "on", "with"
 
class BaseGraph:
   def __init__(self, verbs: List[str], objs: List[str], rels: List[str]):
      super(BaseGraph).__init__()
      self.verbs = verbs
      self.objs = objs
      self.rels = rels

      self.node_types = ["camera_wearer", "verb", "object"]

      # edge "roles" on top of rels
      # camera_wearer -> verb: 'agent'
      # verb -> obj: one of rels (direct_obj, on, with, ...)
      self.special_rels = ["agent"]  # camera_wearer -> verb
      self.all_edge_labels = self.special_rels + rels

   def new_camera_wearer_node(self, node_id: int) -> Node:
      return Node(
         node_id=node_id,
         label="camera_wearer",
         node_type="camera_wearer",
         idx=None,
         feat=None,
      )

   def new_verb_node(self, node_id: int, verb_idx: int, clip_feat: torch.Tensor) -> Node:
      return Node(
         node_id=node_id,
         label=self.verbs[verb_idx],
         node_type="verb",
         idx=verb_idx,
         feat=clip_feat,   # cframe wise feature
      )

   def new_object_node(self, node_id: int, obj_idx: int, obj_feat: torch.Tensor) -> Node:
      return Node(
         node_id=node_id,
         label=self.objs[obj_idx],
         node_type="object",
         idx=obj_idx,
         feat=obj_feat,    # object feature
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
         rel_label=self.rels[rel_idx],
      )

