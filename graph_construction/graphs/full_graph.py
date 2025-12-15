from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from .base_graph import BaseGraph,Node,Edge

class FullActionGraph(BaseGraph):
   def __init__(self, verbs, objs, rels):
      super(BaseGraph).__init__(verbs, objs, rels)
      self.nodes: Dict[int, Node] = {}
      self.edges: List[Edge] = []

   def create_graph(self,
                    verb_idx,
                    clip_feat,
                    obj_indices,
                    obj_feats,
                    rels_vector):
      
      node_id_counter = 0

      # 1) CW node
      cw_id = node_id_counter
      node_id_counter += 1
      cw_node = self.new_cw_node(cw_id)
      self.nodes[cw_id] = cw_node

      # 2) Verb node
      verb_id = node_id_counter
      node_id_counter += 1
      verb_node = self.new_verb_node(verb_id, verb_idx, clip_feat)
      self.nodes[verb_id] = verb_node

      # CW -> verb edge
      self.edges.append(self.agent_edge(cw_id, verb_id))

      # 3) Object nodes + edges verb -> object
      obj_indices = obj_indices        # (num_obj,)
      obj_feats = obj_feats            # (num_obj, 1024)
      rels_vecs = rels_vecs          # (num_obj, num_rels)

      obj_id_map: Dict[int, int] = {}  # obj_idx -> node_id

      for object_num, obj_idx in enumerate(obj_indices.tolist()):
         obj_feat = obj_feats[object_num]                # (1024,)
         node_id = node_id_counter
         node_id_counter += 1

         obj_node = self.new_object_node(node_id, obj_idx, obj_feat)
         self.nodes[node_id] = obj_node
         obj_id_map[obj_idx] = node_id

         # Add edges for all relationships where rel_vec == 1
         rels_vec = rels_vecs[object_num]               # (num_rels,)
         for rel_idx in torch.where(rels_vec > 0)[0].tolist():
               self.edges.append(self.rel_edge(verb_id, node_id, rel_idx))

      return self

   def to_easg_tensors(self) -> Dict[str, torch.Tensor]:
      """
      Optional: convert back to the old tensor format (for compatibility
      with your EASG model if needed).
      """
      # find verb node
      verb_nodes = [n for n in self.nodes.values() if n.node_type == "verb"]
      assert len(verb_nodes) == 1
      verb_node = verb_nodes[0]

      # object nodes
      obj_nodes = [n for n in self.nodes.values() if n.node_type == "object"]

      verb_idx_tensor = torch.tensor([verb_node.idx], dtype=torch.long)
      clip_feat_tensor = verb_node.feat  # (2304,)

      obj_indices = torch.tensor([n.idx for n in obj_nodes], dtype=torch.long)
      obj_feats = torch.stack([n.feat for n in obj_nodes], dim=0)

      num_rels = len(self.self.rels)
      rels_vecs = torch.zeros((len(obj_nodes), num_rels), dtype=torch.float32)

      for e in self.edges:
         if e.rel_idx is None:
               continue
         # e.src is verb_id, e.dst is object node
         obj_node = self.nodes[e.dst]
         obj_row = obj_nodes.index(obj_node)
         rels_vecs[obj_row, e.rel_idx] = 1.0

      # triplets (verb, obj_idx, rel_idx)
      triplets = []
      for obj_node in obj_nodes:
         obj_row = obj_nodes.index(obj_node)
         for rel_idx in torch.where(rels_vecs[obj_row] > 0)[0]:
               triplets.append((verb_node.idx, obj_node.idx, rel_idx.item()))
      triplets = torch.tensor(triplets, dtype=torch.long) if len(triplets) > 0 else torch.zeros((0, 3), dtype=torch.long)

      return {
         "verb_idx": verb_idx_tensor,
         "clip_feat": clip_feat_tensor,
         "obj_indices": obj_indices,
         "obj_feats": obj_feats,
         "rels_vecs": rels_vecs,
         "triplets": triplets,
      }
