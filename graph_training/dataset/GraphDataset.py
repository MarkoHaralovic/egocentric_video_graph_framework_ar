from typing import List
from ...graph_construction.graphs.full_graph import FullActionGraph
from torch.utils.data import Dataset
import os
import json
from ast import literal_eval
from tqdm import tqdm
import h5py
import cv2
import pandas as pd
from ...global_feature_training.data_loading.dataset_split import decode_label,map_or_skip_label, stratified_split
import pickle
import numpy as np
import torch

TRAIN_SIZE = 0.8
VAL_SIZE = 0.2

ignored_verbs = ['toast', 'fold', 'take', 'put', 'cut', 'play']
ignored_nouns = [
   'sink','washer','headphones','camera','watch','pan','tv','coffee maker','table','book','washer','sink',
   'cupboard','pants','present','macaroni','pizza','snack','banana','toast','bag','pancake','chip','newspaper',
   'magazine','soup','toast','treadmill','bag','kitchen','inside','house','door','floor','clothes','pan','cup',
   'fruit','ceiling','paper','note','room','carpet','table','floor'
]
noun_replacement="other"
skip_labels = {"na"}

DATASET_PATH = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
model_name = "dinov3h16+"
pooling="concat"

clips = [clip for clip in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, clip))]

def return_train_val_samples(input_folder=DATASET_PATH,clips=clips, model_name=model_name, num_frames = None, pooling=pooling, skip_labels=skip_labels, skip_verbs = ignored_verbs,
                             skip_nouns=ignored_nouns, noun_replacement="other",skip_na=True,val_ratio=VAL_SIZE):
   samples = collect_samples(
      input_folder,
      clips,
      model_name,
      pooling,
      num_frames,
      skip_labels,
      skip_verbs,
      skip_nouns,
      noun_replacement,
      skip_na
   )
   train_samples, val_samples = stratified_split(samples, val_ratio, seed=0)
   

   acts = sorted({s[3] for s in samples})
   activity_to_idx = {a:i for i,a in enumerate(acts)}
   
   return train_samples, val_samples, activity_to_idx

def collect_samples(input_folder, clip_names, model_name, pooling=None,num_frames=None,
                  skip_labels=set(), skip_verbs=set(), skip_nouns=set(), noun_replacement="other", skip_na=True):
   
   samples = []  # (clip_name, h5_path, block_idx, label_str)

   for clip_name in clip_names:
      clip_path = os.path.join(input_folder, clip_name)
      
      frames_path  = os.path.join(clip_path, "frames")
      annotations = pd.read_csv(os.path.join(clip_path, "annotations.csv"))
      parse_annotations = pd.read_csv(os.path.join(clip_path, "parse_annotation.csv"))
      object_features_path = os.path.join(clip_path, "object_features_dinoT.pkl")
      with open(object_features_path, 'rb') as f:
         object_features = pickle.load(f)
         
      if pooling is not None:
         h5_path = os.path.join(clip_path, f"activity_features_model_{model_name}_pooling_{pooling}.h5")
      elif num_frames is not None:
         h5_path = os.path.join(clip_path, f"activity_features_model_{model_name}_numframes_{num_frames}.h5")
      else:
         raise Exception(f"Define either num_frames or pooling.")
      
      if not os.path.exists(h5_path):
         continue

      with h5py.File(h5_path, "r") as f:
         labels = f["activity_labels"][:]
         for block_idx, raw_label in enumerate(labels):
            frame_anns = annotations[annotations["activity_block_id"] == block_idx]
            frame_idxs = list(frame_anns["frame_index"]) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            frame_clips = f['visual_features'][block_idx]      
            frame_clips = frame_clips.reshape(10,1280)     
            frame_parse_anns_list = [parse_annotations[parse_annotations["frame_id"] == f_idx] for f_idx in frame_idxs]
            frame_parse_anns = pd.concat(frame_parse_anns_list, ignore_index=False) if frame_parse_anns_list else pd.DataFrame()
            frame_object_features = [object_features[f"frame_{int(ind)}"] for ind in frame_idxs]
            lab = decode_label(raw_label)
            lab = map_or_skip_label(lab, skip_labels, skip_verbs, skip_nouns, noun_replacement, skip_na)
            if lab is not None:
               samples.append((clip_name, h5_path, block_idx, lab, frame_anns, frame_parse_anns, frame_clips, frame_object_features))

   return samples

class GraphDataset(Dataset):

   def __init__(self, input_path, samples):
      with open(os.path.join(input_path, "verbs.json"), "r") as f:
         self.verbs = json.load(f)
         
      with open(os.path.join(input_path, "objects.json"), "r") as f:
         self.objs = json.load(f)
            
      with open(os.path.join(input_path, "relationships.json"), "r") as f:
         self.rels = json.load(f)

      with open(os.path.join(input_path, "attributes.json"), "r") as f:
         self.attrs = json.load(f)   
      self.input_path = input_path
      
      self.samples = samples
      self.h5_paths = sorted(list({s[1] for s in samples}))
      self.h5_to_file_idx = {p: i for i, p in enumerate(self.h5_paths)}
      self.clip_names = [None] * len(self.h5_paths)
      for clip_name, h5_path, _, _, _, _, _,_ in samples:
         self.clip_names[self.h5_to_file_idx[h5_path]] = clip_name
      
      self.activity_to_idx = activity_to_idx or {a: i for i, a in enumerate(sorted({s[3] for s in samples}))}
      self.idx_to_activity = {v: k for k, v in self.activity_to_idx.items()}
      self.sample_index = [(self.h5_to_file_idx[s[1]], s[2], s[3],s[4], s[5],s[6], s[7]) for s in samples]
      
   def __len__(self):
      return len(self.full_graphs)

   def __getitem__(self, idx):
      """Each sample  is:
         h5  file index, block_idx, activity_label, per frame annotationsn, per frame paarsed annotoations, per frame clip feats, object features.
         They are then transformemd in 10 FullActionSceneGraphs
         Output is then:
            clip_name
            activity label
            activity name
            block_idx
            list of 10 FullActionGraphs
      """
      file_idx, block_idx, label_str, frame_anns, frame_parsed_anns, frame_feats, obj_feats = self.sample_index[idx]
      output = {'clip_name': self.clip_names[file_idx], 'block_idx': block_idx}
      output['activity_label'] = torch.tensor(self.activity_to_idx[label_str], dtype=torch.long)
      output['activity_name'] = label_str
      
      action_scene_graphs =  {}
      
      for i, frame_id in enumerate( frame_anns["frame_index"].tolist()):
         frame_parsed_ann = frame_parsed_anns[frame_parsed_anns["frame_id"] == frame_id]
         graph = FullActionGraph(self.verbs, self.objs, self.rels, self.attrs)
         
         verb = frame_parsed_ann["verb"].iloc[0]
         direct_object = frame_parsed_ann["direct_object"].iloc[0]
         objects_atr_map = literal_eval(frame_parsed_ann["all_objects"].iloc[0])
         rels_dict = literal_eval(frame_parsed_ann["preposition_object_pairs"].iloc[0])
         aux_verbs_str = frame_parsed_ann["aux_verbs"].iloc[0]
         aux_verbs = literal_eval(aux_verbs_str) if aux_verbs_str and aux_verbs_str != '[]' else None
         aux_obj_str = frame_parsed_ann["object_aux_verb"].iloc[0]
         aux_direct_objects_map = literal_eval(aux_obj_str) if aux_obj_str and aux_obj_str != '{}' else None
         
         graph =  graph.create_graph(
            verb = verb,
            direct_object = direct_object,
            objects_atr_map=objects_atr_map,
            clip_feat=frame_feats[i],
            obj_feats=obj_feats[i],
            rels_dict=rels_dict,
            aux_verbs=aux_verbs,
            aux_direct_objects_map=aux_direct_objects_map,
         )
         action_scene_graphs[i] = graph
      output["full_action_graphs"] = action_scene_graphs
      return output
   
train_samples, val_samples, activity_to_idx =  return_train_val_samples()
ds = GraphDataset(DATASET_PATH, val_samples)

out  =  ds.__getitem__(0)

print(out)