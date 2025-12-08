import spacy
from spacy.symbols import nsubj, VERB
import re
import os
import tqdm
import pandas as pd
import json

nlp = spacy.load("en_core_web_sm")

def get_subject_verb_pairs(t):
   doc = nlp(t)
   subject_verb_pairs = []
   for possible_subject in doc:
      if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
         subject_verb_pairs.append({possible_subject:possible_subject.head.lemma_})
   if len(subject_verb_pairs) > 0: 
      return subject_verb_pairs
   else: return None       

   
def get_preposition_object_pairs(t):
   doc = nlp(t)
   preposition_object_pairs = []
   for possible_object in doc:
      if possible_object.dep_ == 'pobj' and possible_object.head.dep_ == 'prep':
         preposition_object_pairs.append({possible_object:possible_object.head.lemma_})
   if len(preposition_object_pairs) > 0: 
      return preposition_object_pairs
   else: return None       

   
def check_verb(token):
   """Check verb type given spacy token"""
   if token.pos_ == 'VERB':
      indirect_object = False
      direct_object = False
      for item in token.children:
         if(item.dep_ == "iobj" or item.dep_ == "pobj"):
               indirect_object = True
         if (item.dep_ == "dobj" or item.dep_ == "dative"):
               direct_object = True
      if indirect_object and direct_object:
         return 'DITRANVERB'
      elif direct_object and not indirect_object:
         return 'TRANVERB'
      elif not direct_object and not indirect_object:
         return 'INTRANVERB'
      else:
         return 'VERB'
      
def check_dobj(token):
   if token.dep_ == 'dobj':
      return str(token)
   else:
      return None

def check_if_obj(token):
   if token.pos_ == "NOUN" or token.pos_ == "PROPN":
      if token.dep_ in ("dobj", "obj"):
         return str(token)
      if token.dep_ in ("iobj", "dative"):
         return str(token)
      if token.dep_ == "pobj" and token.head.dep_ == "prep":
         return str(token)
   return None

def extract_noun_compounds(token):
   compounds = []
   
   modifiers = []
   for child in token.children:
      if child.dep_ in ("compound", "amod", "nmod") and child.pos_ in ("NOUN", "ADJ", "PROPN"):
         modifiers.append(child)
   
   modifiers = sorted(modifiers, key=lambda x: x.i)
   
   if modifiers:
      compound_phrase = " ".join([str(mod) for mod in modifiers]) + " " + str(token)
      compounds.append(compound_phrase)
   
   return compounds

def standardize_narration(t):
   if t:
      t = t.replace("The camera wearer", "The_camera_wearer")
      t = t.replace("#Unsure","").replace("#unsure","").replace("#Sammary","").replace("#sammary","").replace("#Summary","")
      t = t.strip()
      # remove extra whitespaces
      t = re.sub(' +', ' ', t)
      t = re.sub('#\w','',t)
      t = t.strip()
      if t[0].islower(): t = t[0].upper()+t[1:]
      if t.endswith("."): t = t[:-1]
      return t.strip()
   else: return ""
   
def parse_annotate_action(action):
   standardized = standardize_narration(action)
   preposition_object_pairs = get_preposition_object_pairs(standardized)
   subject_verb_pairs = get_subject_verb_pairs(standardized)

   if subject_verb_pairs is None or len(subject_verb_pairs) == 0:
      return None, None, None, None, None, None, None, None, None, None

   subj_verb_dict = subject_verb_pairs[0]
   subject, verb = list(subj_verb_dict.keys())[0], list(subj_verb_dict.values())[0]

   direct_object = None
   all_objects = []
   pos_mask = None
   tag_mask = None
   dep_mask = None
   verb_type =  'VERB'
   phrasal_verb = None


   doc = nlp(standardized)
   verb_type = "VERB"
   prev_is_verb = False
   pos_toks, tag_toks, dep_toks = [],[],[]
   for token in doc:
      pos_toks.append(token.pos_)
      tag_toks.append(token.tag_)
      dep_toks.append(token.dep_)
      
      if token.lemma_ == verb:
         prev_is_verb = True
         verb_type = check_verb(token)

      if token.dep_ == 'prt' and prev_is_verb:
         phrasal_verb = str(verb) + "-" + str(token)
            
      res_dobj = check_dobj(token)
      if res_dobj is not None:
         direct_object =  res_dobj
         
      is_object = check_if_obj(token)
      if is_object is not None:
         all_objects.append(is_object)
         # Also extract noun compounds
         compounds = extract_noun_compounds(token)
         all_objects.extend(compounds)
         
   pos_mask = str(pos_toks)
   tag_mask = str(tag_toks)
   dep_mask= str(dep_toks)
   
   return subject, verb,verb_type,direct_object, all_objects, phrasal_verb, preposition_object_pairs, pos_mask, tag_mask, dep_mask

   
def parse_annotate_folder(input_path):
   clips = [clip for clip in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, clip))]
   objects = set()
   relationships = set()
   verbs = set()
   activities = set()
   
   for clip in tqdm.tqdm(clips, desc=f"Parsing annotations from {input_path}"):
      rows = []
      with open(os.path.join(input_path, clip, "actions.txt"), "r") as f:
         actions_list = [line.strip() for line in f.readlines()]
      
      with open(os.path.join(input_path, clip, "activities.txt"), "r") as f:
         activities_list = [line.strip() for line in f.readlines()]
      
      for act in activities_list:
         activities.add(act)
         
      for i, action in enumerate(actions_list):
         result = parse_annotate_action(action)
         if result[0] is not None:  
            subject, verb, verb_type, direct_object, all_objects, phrasal_verb, preposition_object_pairs, pos_mask, tag_mask, dep_mask = result
            
            if direct_object: objects.add(direct_object)
            if all_objects:objects.update(all_objects)
            verbs.add(verb)
            
            if preposition_object_pairs:
               for pdict in preposition_object_pairs:
                  for _, v in pdict.items():
                     relationships.add(v)
            
            rows.append({
               "frame_id": i,
               "subject": str(subject),
               "verb": verb,
               "verb_type": verb_type,
               "direct_object": direct_object,
               "all_objects": str(all_objects),
               "phrasal_verb": phrasal_verb,
               "preposition_object_pairs": str(preposition_object_pairs),
               "pos_mask": pos_mask,
               "tag_mask": tag_mask,
               "dep_mask": dep_mask
            })
      
      if rows:
         clip_dataframe = pd.DataFrame(rows)
         clip_dataframe.to_csv(os.path.join(input_path, clip, "parse_annotation.csv"), index=False)
   
   if objects:
      objects_enum = {obj: idx for idx, obj in enumerate(sorted(list(objects)))}
      with open(os.path.join(input_path, "objects.json"), "w") as f:
         json.dump(objects_enum, f, indent=2)
   
   if relationships:
      relationships_enum = {rel: idx for idx, rel in enumerate(sorted(list(relationships)))}
      with open(os.path.join(input_path, "relationships.json"), "w") as f:
         json.dump(relationships_enum, f, indent=2)
   
   if verbs:
      verbs_enum = {verb: idx for idx, verb in enumerate(sorted(list(verbs)))}
      with open(os.path.join(input_path, "verbs.json"), "w") as f:
         json.dump(verbs_enum, f, indent=2)

   if activities:
      activities_enum = {act: idx for idx, act in enumerate(sorted(list(activities)))}
      with open(os.path.join(input_path, "activities.json"), "w") as f:
         json.dump(activities_enum, f, indent=2)
         
def main():
   INPUT_DATASET_FOLDER = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
   parse_annotate_folder(INPUT_DATASET_FOLDER)
   
if __name__ == "__main__":
   main()