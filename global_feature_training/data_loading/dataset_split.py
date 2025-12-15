import os
import random
import json

TRAIN_SIZE = 0.8
VAL_SIZE = 0.2

DATASET_PATH = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
OUTPUT_PATH = f"/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf/training_data_split_train_{TRAIN_SIZE}_val_{VAL_SIZE}.json"
clips = [clip for clip in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, clip))]

location_script_train_test_split = {}
location_script_train_test_split["split_summary"] = {
   "train" :  [],
   "validation" :   []
}

loc_script_combinations = set()
for clip in clips:
   parts = clip.split("_")
   if len(parts) >= 2:
      loc_script_combinations.add(f"{parts[0]}_{parts[1]}")

for loc_script in loc_script_combinations:
   loc, script = loc_script.split("_")
   all_recs = [clip for clip in clips if clip.split("_")[0] == loc and clip.split("_")[1] == script]
   
   if len(all_recs) == 0:
      continue
   
   random.shuffle(all_recs)
   val_count = max(1, int(len(all_recs) * VAL_SIZE))
   train_count = len(all_recs) - val_count
   
   location_script_train_test_split[loc_script] = {
      "train": all_recs[:train_count],
      "validation": all_recs[train_count:]
   }

val_len = 0
for k,v in location_script_train_test_split.items(): 
   if k != "split_summary":
      location_script_train_test_split["split_summary"]["validation"].extend(v["validation"])
      val_len+=len(v["validation"])
print(val_len)

train_len = 0
for k,v in location_script_train_test_split.items():
   if k != "split_summary":
      location_script_train_test_split["split_summary"]["train"].extend(v["train"])
      train_len+=len(v["train"])
print(train_len)

print(f" train % : {train_len / (train_len+val_len)*100:.2f}%")
print(f" val % : {val_len / (train_len+val_len)*100:.2f}%")

with open(OUTPUT_PATH, "w") as f:
   json.dump(location_script_train_test_split, f, indent=2)