from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from .base_graph import BaseGraph,Node,Edge
import torch
import spacy
nlp = spacy.load("en_core_web_sm")

def to_singular(word: str) -> str:
   doc = nlp(word)
   return_class =[]
   for tok in doc:
      if tok.pos_ in ("NOUN", "PROPN"):
         return_class.append(tok.lemma_.lower())
   if return_class:
      return " ".join(c for c in return_class)
   return word.lower()
   
class FullActionGraph(BaseGraph):
   def __init__(self, verbs : Dict, objs : Dict, rels : Dict, attrs : Dict):
      super(FullActionGraph,self).__init__(verbs, objs, rels, attrs)
      self.nodes: Dict[int, Node] = {}
      self.edges: List[Edge] = []

   def create_graph(
      self,
      verb,
      direct_object,
      objects_atr_map,
      clip_feat,
      obj_feats,
      rels_dict, 
      aux_verbs = None,
      aux_direct_objects_map = None,
   ):
      
      aux_verb_idxs = [self.verbs[aux_verb] for aux_verb in aux_verbs] if aux_verbs is not None else None
      
      obj_data = []  
      for obj_name, obj_info in objects_atr_map.items():
         try:
            obj_idx = self.objs[to_singular(obj_info["base_object"])]
            obj_data.append((obj_idx, obj_name, obj_info["attributes"]))
         except:
            print(to_singular(obj_info["base_object"]))
            print(self.objs)  
      obj_indices = [item[0] for item in obj_data]
      rels_vecs = torch.zeros(len(self.objs), len(self.rels))
      for _item in rels_dict:
         obj_name, rel = list(_item.items())[0]
         if rel not in self.rels or to_singular(objects_atr_map.get(obj_name, {}).get("base_object", "")) not in self.objs:
            continue
         rels_vecs[self.objs[to_singular(objects_atr_map[obj_name]["base_object"])], self.rels[rel]] = 1.0
      node_id_counter = 0

      try:
         direct_object_id = self.objs[to_singular(direct_object)]
      except:
         print(self.objs)
         print(to_singular(direct_object))
         print(direct_object)
      rels_vecs[direct_object_id, self.rels["direct_object"]] = 1.0
      
      aux_direct_objects = [v[0] for v in aux_direct_objects_map.values() if v and len(v) > 0] if aux_direct_objects_map else None
      
      try:
         aux_direct_object_ids = [self.objs[aux_direct_object] for aux_direct_object in aux_direct_objects]  if aux_direct_objects is not None else None
      except:
         print(aux_direct_objects)
         print(self.objs)
      if aux_direct_object_ids is not None:
         for aux_id in aux_direct_object_ids:
            rels_vecs[aux_id, self.rels["aux_direct_object"]] = 1.0
            
      # CW node
      cw_id = node_id_counter
      node_id_counter += 1
      cw_node = self.new_camera_wearer_node(node_id_counter)
      self.nodes[cw_id] = cw_node

      # Verb node 
      verb_id = node_id_counter
      node_id_counter += 1
      verb_node = self.new_main_verb_node(verb_id, verb, clip_feat)
      self.nodes[verb_id] = verb_node

      # CW -> verb edge
      self.edges.append(self.agent_edge(cw_id, verb_id))

      # verb -> direct_object relationship
      if direct_object_id is None:
         self.edges.append(self.rel_edge(verb_id, direct_object_id,rel_idx = self.rels["direct_object"]))
      
      # aux verb nodes (edges created later after objects exist)
      aux_verb_node_map = {}  # aux_verb_label -> node_id
      if aux_verb_idxs :
         for aux_verb in aux_verbs:
            aux_verb_id = node_id_counter
            node_id_counter += 1
            aux_verb_node = self.new_aux_verb_node(aux_verb_id, aux_verb, clip_feat=None)
            self.nodes[aux_verb_id] = aux_verb_node
            aux_verb_node_map[aux_verb] = aux_verb_id
         
      # object nodes + edges 
      obj_id_map: Dict[int, int] = {}  # obj_idx -> node_id
      attr_vecs = torch.zeros((len(self.objs), len(self.attrs)))
      for object_num, (obj_idx, orig_name, attributes) in enumerate(obj_data):
         if obj_idx  in obj_feats["objects"]:
            obj_feat = obj_feats["objects"][obj_idx]['feats']    
         else:
            feat_shape =  obj_feats["objects"][next(iter(obj_feats["objects"]))]['feats'].shape if next(iter(obj_feats["objects"])) else torch.Size([256]) 
            obj_feat = torch.zeros((feat_shape))
                 
         node_id = node_id_counter
         node_id_counter += 1
         
         # Process attributes
         for attr in attributes:
            attr_vecs[obj_idx, self.attrs[attr]] = 1.0
         
         if aux_direct_object_ids is None or not obj_idx in aux_direct_object_ids:
            obj_node = self.new_object_node(node_id, obj_idx, obj_feat, attributes, attr_vecs[obj_idx], self.verbs[verb])
         else:
            verb_related_to = [k for k, v in aux_direct_objects_map.items() if orig_name in v][0]
            obj_node = self.new_object_node(node_id, obj_idx, obj_feat, attributes, attr_vecs[obj_idx], self.verbs[verb_related_to])
            
         self.nodes[node_id] = obj_node
         obj_id_map[obj_idx] = node_id

         rels_vec = rels_vecs[obj_idx] 
         for rel_idx in torch.where(rels_vec > 0)[0].tolist():
            # rels_vecs of shape (n_objs, rels)
            self.edges.append(self.rel_edge(verb_id, node_id, rel_idx))

      # Now create aux verb -> object edges
      if aux_verb_idxs and aux_direct_objects_map:
         for aux_verb, aux_verb_node_id in aux_verb_node_map.items():
            if aux_direct_objects_map[aux_verb]:
               aux_obj = aux_direct_objects_map[aux_verb][0]
               aux_obj_idx = self.objs[aux_obj]
               if aux_obj_idx in obj_id_map:
                  self.edges.append(self.rel_edge(aux_verb_node_id, obj_id_map[aux_obj_idx], rel_idx=self.rels["aux_direct_object"]))
      
      self.rels_vecs = rels_vecs
      self.attr_vecs = attr_vecs
      
      return self

   def to_easg_tensors(self) -> Dict[str, torch.Tensor]:

      verb_node = [n for n in self.nodes.values() if n.node_type == "verb"]
      assert len(verb_node) == 1
      verb_node = verb_node[0]
      verb_idx_tensor = torch.tensor([verb_node.idx], dtype=torch.long)
      clip_feat_tensor = verb_node.feat 
      
      aux_verb_nodes = [n for n in self.nodes.values() if n.node_type == "aux_verb"]
      aux_verb_tensors = [torch.tensor([aux_verb_node.idx], dtype=torch.long) for aux_verb_node in aux_verb_nodes]
      aux_verb_tensors = torch.tensor(aux_verb_tensors)
      
      obj_nodes = [n for n in self.nodes.values() if n.node_type == "object"]
      
      obj_indices = torch.tensor([n.idx for n in obj_nodes], dtype=torch.long)
      obj_feats = torch.stack([n.feat for n in obj_nodes], dim=0)

      num_rels = len(self.rels)
      rels_vecs = torch.zeros((len(obj_nodes), num_rels), dtype=torch.float32)
      
      num_attrs = len(self.attrs)
      attr_vecs = torch.zeros((len(obj_nodes), num_attrs), dtype=torch.float32)
      
      for e in self.edges:
         if e.rel_idx is None:
               continue
         obj_node = self.nodes[e.dst]
         obj_row = obj_nodes.index(obj_node)
         rels_vecs[obj_row, e.rel_idx] = 1.0
         
      triplets = []
      
      for aux_verb in aux_verb_nodes:
         triplets.append((verb_node.idx, aux_verb.idx, self.rels["aux_verb"]))
         
      for obj_node in obj_nodes:
         obj_row = obj_nodes.index(obj_node)
         for attr in obj_node.attr:
            attr_vecs[obj_row,self.attrs[attr]] = 1.0
         for rel_idx in torch.where(rels_vecs[obj_row] > 0)[0]:
            verb_for_triplet = obj_node.related_to_verb if obj_node.related_to_verb is not None else verb_node.idx
            triplets.append((verb_for_triplet, obj_node.idx, rel_idx.item()))
      triplets = torch.tensor(triplets, dtype=torch.long) if len(triplets) > 0 else torch.zeros((0, 3), dtype=torch.long)
   
      return {
         "verb_idx": verb_idx_tensor,
         "aux_verb_idx": aux_verb_tensors,
         "clip_feat": clip_feat_tensor,
         "obj_indices": obj_indices,
         "attr_vecs" : attr_vecs,
         "obj_feats": obj_feats,
         "rels_vecs": rels_vecs,
         "triplets": triplets,
      }

def test():
   import os
   import json
   input_path = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
   sentence = "The camera wearer is washing hands in a sink while looking at his reflection in a mirror."
   subject = "The_camera_wearer"
   verb = "wash"
   direct_object = "hands"
   objects_atr_map = {
      "hands": {"base_object": "hand", "attributes": ['dirty', 'large']}, 
      "sink": {"base_object": "sink", "attributes": ['high', 'white', 'small']}, 
      "reflection": {"base_object": "reflection", "attributes": ['various', 'tall']}, 
      "mirror": {"base_object": "mirror", "attributes": ['textured', 'modern']}}

   rels_dict = [{'sink': 'in'}, {'reflection': 'at'}, {'mirror': 'in'}, {'hands' : 'with'}]
   aux_verbs = ['look']
   aux_direct_objects_map = {'look': ['reflection']}

   with open(os.path.join(input_path, "verbs.json"), "r") as f:
      verbs = json.load(f)
         
   with open(os.path.join(input_path, "objects.json"), "r") as f:
      objs = json.load(f)
         
   with open(os.path.join(input_path, "relationships.json"), "r") as f:
      rels = json.load(f)

   with open(os.path.join(input_path, "attributes.json"), "r") as f:
      attrs = json.load(f)   
      
   verbs = {k:v for k, v in sorted(verbs.items(), key=lambda x: x[1])}
   objs = {k:v for k, v in sorted(objs.items(), key=lambda x: x[1])} 
   rels = {k:v for k, v in sorted(rels.items(), key=lambda x: x[1])} 
   
   full_graph = FullActionGraph(verbs, objs, rels, attrs)
   graph = full_graph.create_graph(
         verb,
         direct_object,
         objects_atr_map,
         torch.zeros(100),
         [torch.zeros(25),torch.zeros(25),torch.zeros(25),torch.zeros(25)],
         rels_dict,
         aux_verbs = aux_verbs,
         aux_direct_objects_map = aux_direct_objects_map,
   )
   #output should be
   # Triplet 0: (wash, look, aux_verb)
   # Triplet 1: (wash, hand, direct_object)
   # Triplet 2: (wash, hand, with)
   # Triplet 3: (wash, sink, in)
   # Triplet 4: (look, reflection, at)
   # Triplet 5: (look, reflection, aux_direct_object)
   # Triplet 6: (wash, mirror, in)
   print(graph.to_easg_tensors())
   
if __name__ == "__main__":
   test()