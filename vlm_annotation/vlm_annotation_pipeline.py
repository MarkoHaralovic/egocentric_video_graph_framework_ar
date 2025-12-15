import os
import random
from projectaria_tools.projects.aea import (
    AriaEverydayActivitiesDataProvider)
import numpy as np

from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import get_eyegaze_point_at_depth

import cv2
import tqdm
import pandas as pd

# VLM part
from PIL import Image
import torch

import torch.multiprocessing as mp

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig,LlavaNextProcessor, LlavaNextForConditionalGeneration

# hyperparameters 
VLM_ANNOTATOR = "/home/s3758869/models/llava-v1.6-34b-hf" #"/home/s3758869/models/LLaVA-NeXT-Video-34B-hf" # "/home/s3758869/models/llava-v1.6-mistral-7b-hf" # "/home/s3758869/models/llava-v1.6-34b-hf" #"llava-hf/llava-next-72b-hf" "/home/s3758869/models/llava-next-72b-hf" 
MODEL_CACHE_DIR = "/home/s3758869/models"  
NUM_GPUS = torch.cuda.device_count() 
EACH_NTH_FRAME_SEC = 3 # select one frame in this amount of seconds
N_FRAMES_FOR_ACTIVITY = 10 # number of consecutive action frames to be sent for activity annotation
IMAGE_SIZE_VLM_INPUT =  336 # image size for LLaVA annotation
LOCAL_WINDOW_FOR_ACTION_SIZE = 1 # amount of images at each timestamp sent for action recogntion. If eg 3, at second N, context is frame one second befor and after
FOUR_BIT_QUANTIZATION = True # whether to quantize LLaVA model to reduce VRAM usage
EIGHT_BIT_QUANTIZATION = False

# data location
INPUT_DATA_FOLDER = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/AriaEA"
OUTPUT_DATA_FOLDER = f"/home/s3758869/vlm_datasets/AriaEA_vlm_ann_{EACH_NTH_FRAME_SEC}_{N_FRAMES_FOR_ACTIVITY}_{VLM_ANNOTATOR.split('/')[-1]}"
os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)
RGB_STREAM_ID = StreamId("214-1")

def load_llava_model(
   model_name,
   device="cuda",
   four_bit=FOUR_BIT_QUANTIZATION,
   eight_bit=EIGHT_BIT_QUANTIZATION,
):
   if four_bit or eight_bit:
      bnb_config = BitsAndBytesConfig(
         load_in_4bit=four_bit,
         load_in_8bit=eight_bit,
         bnb_4bit_compute_dtype=torch.float16,
         bnb_4bit_use_double_quant=True,
         bnb_4bit_quant_type="nf4", 
      )

      device_map = {"": device} if isinstance(device, str) and "cuda" in device else "auto"

      model = LlavaNextForConditionalGeneration.from_pretrained(
         model_name,
         quantization_config=bnb_config,
         device_map=device_map,     
         low_cpu_mem_usage=True,
         cache_dir=MODEL_CACHE_DIR,
         local_files_only=True
      )
      
      model.gradient_checkpointing_enable()

   else:
      model = LlavaNextForConditionalGeneration.from_pretrained(
         model_name,
         torch_dtype=torch.float16,
         low_cpu_mem_usage=True,
         use_flash_attention_2=True,
         cache_dir=MODEL_CACHE_DIR,
         local_files_only=True
      ).to(device)

   processor = LlavaNextProcessor.from_pretrained(
      model_name,
      cache_dir=MODEL_CACHE_DIR,
      local_files_only=True,
      use_fast=True
   )
   model.eval()
   return model, processor

def parse_llava_output(llava_output):
   parsed = llava_output  # Default: return as-is
   
   if "[/INST]" in llava_output:
      parsed = llava_output.split("[/INST]")[-1].strip()
   elif "ASSISTANT: " in llava_output:
      parsed = llava_output.split("ASSISTANT: ")[-1].strip()
   elif "assistant/n" in llava_output:
      parsed = llava_output.split("assistant/n")[-1].strip()
   elif "assistant" in llava_output:
      parsed = llava_output.split("assistant")[-1].strip()
   return parsed
      
def stream_rgb_vrs_recording(aea_data_provider, rgb_stream_id):
   timestamps = aea_data_provider.vrs.get_timestamps_ns(
      rgb_stream_id, TimeDomain.DEVICE_TIME
   )

   images = []
   for idx in range(len(timestamps)):
      img = aea_data_provider.vrs.get_image_data_by_index(rgb_stream_id, idx)[0]
      images.append(img.to_numpy_array())

   return images, timestamps

def get_every_nth_second_frame(timestamps_ns, n_seconds=1):
   if not isinstance(timestamps_ns, np.ndarray):
      timestamps_ns = np.array(timestamps_ns)
      
   timestamps_sec = (timestamps_ns - timestamps_ns[0]) / 1e9
   
   max_time = int(timestamps_sec[-1])
   target_times = np.arange(0, max_time + n_seconds + 1, n_seconds)
   
   sampled_indices = []
   for target in target_times:
      indexes= np.argmin(np.abs(timestamps_sec - target))
      sampled_indices.append(indexes)
   sampled_indices = sorted(set(sampled_indices))
   
   total_frames = len(timestamps_ns)
   sampled_frames = len(sampled_indices)
   percentage = (sampled_frames / total_frames) * 100
   
   print(f"Total frames: {total_frames}, sampled frames: {sampled_frames}, sampling each {int(total_frames/sampled_frames)}th frame, pct : {percentage:.2f}%")
   return np.array(sampled_indices)

def recognize_action_single_frame(model, processor, image_np,image_size = IMAGE_SIZE_VLM_INPUT):
   """
   Takes a numpy array image (H, W, 3 – RGB)
   Returns a string with VLM-predicted action.
   """
   torch.cuda.empty_cache()
   
   img = Image.fromarray(image_np)
   
   img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
   
   conversation = [
      {
         "role": "user",
         "content": [
               {"type": "image"},
               {
                "type": "text",
                "text": """
               You will analyze ONE egocentric image and output the camera wearer's action.

               Output one sentence only, and it MUST follow this exact structure:

                  The camera wearer is <verb-ing> <object> <optional-context>.

              Rules:
               - Start with exactly: "The camera wearer is"
               - Use clear verb in -ing form (opening, closing, holding, cutting, washing, eating, pouring, using, pushing, pulling, taking, placing, walking, etc.) for the action
               - Define main object (door, phone, cup, bowl, laptop, sink, pan, chair, book, bag, etc.).
               - Context must be short and descriptive (location or surrounding objects).
               - Optional context can only include relevant objects and their characteristics in the frame scene
               - No adjectives unless needed to identify the object.
               - Include other people as objects as weel, direct if action done with them, indirect if it can not be concluded.
                     

               Examples you MUST imitate:
                  The camera wearer is playing tennis with another person in broad daylight.
                  The camera wearer is running with a group of people with lot of peaople at sidelines.
                  The camera wearer is playing table tennis with a girl.
                  The camera wearer is washing hands in a sink while looking at his reflection in a mirror.

               Return only the sentence. Do not explain reasoning.
               """
            }
         ],
      },
   ]
   prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
   
   inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
   
   with torch.inference_mode(): 
      output = model.generate(
         **inputs,
         max_new_tokens=100,
         do_sample=False  
      )

      result = processor.decode(output[0], skip_special_tokens=True)
   
   del inputs, output
   torch.cuda.empty_cache()
   
   return result

def recognize_action_local_window(model, processor, images_np,image_size = IMAGE_SIZE_VLM_INPUT):
   """
   Takes a numpy array images (N, H, W, 3 – RGB)
   Returns a string with VLM-predicted action.
   """
   torch.cuda.empty_cache()
   
   images = [Image.fromarray(img_np) for img_np in images_np]
   
   for img in images:
      img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
   
   content = []
   for _ in images:
      content.append({"type": "image"})
   
   content.append({
         "type": "text",
         "text": """
      You are given several egocentric images that belong to one short time span. 
      They are ordered in time from earliest to latest.

      Your task:
      1. Look at ALL the images together.
      2. Analyze images and output the camera wearer's action in the central frame.
      
      You will analyze ONE egocentric image and output the camera wearer's action.
      Output one sentence only, and it MUST follow this exact structure:

         The camera wearer is <verb-ing> <object> <optional-context>.

      Rules:
         - Start with exactly: "The camera wearer is"
         - Use clear verb in -ing form (opening, closing, holding, cutting, washing, eating, pouring, using, pushing, pulling, taking, placing, walking, etc.) for the action
         - Define main object (door, phone, cup, bowl, laptop, sink, pan, chair, book, bag, etc.).
         - Context must be short and descriptive (location or surrounding objects).
         - Optional context can only include relevant objects and their characteristics.
         - No adjectives unless needed to identify the object.
         

         Examples you MUST imitate:

            The camera wearer is opening a door in a long hallway full of people's portraits.
            The camera wearer is cutting vegetables on a brown board.
            The camera wearer is using a phone near a mirror in a bedroom.
            The camera wearer is taking a cup from the table with another person sitting at the table.
            The camera wearer is washing hands in a sink while looking at his reflection in a mirror.

         Return only the sentence. Do not explain reasoning.
      """
   })
   
   conversation = [
      {
         "role": "user",
         "content": content
      },
   ]
   
   prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
   
   inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
   
   with torch.inference_mode(): 
      output = model.generate(
         **inputs,
         max_new_tokens=100,
         do_sample=False
      )

      result = processor.decode(output[0], skip_special_tokens=True)
   
   del inputs, output
   torch.cuda.empty_cache()
   
   return result

def recognize_activity(model, processor, images_np, image_size = IMAGE_SIZE_VLM_INPUT, actions=None):
   """
   Takes a numpy array image (N, H, W, 3) -> N rgb images
   Returns a string with VLM-predicted activity across span of N images
   """
   torch.cuda.empty_cache()
   
   activity_image_size = min(IMAGE_SIZE_VLM_INPUT, image_size)
   
   images = [Image.fromarray(img_np) for img_np in images_np]
   
   for img in images:
      img.thumbnail((activity_image_size, activity_image_size), Image.Resampling.LANCZOS)
   
   content = []
   for _ in images:
      content.append({"type": "image"})
   
   content.append({
      "type": "text",
      "text": f"""
      You are given several egocentric images that belong to one short time span. 
      They are ordered in time from earliest to latest.

      Your task:
      1. Look at ALL the images together.
      2. Decide the SINGLE main activity the human (camera wearer) is performing across the whole sequence.
      3. Express this activity as one compact label in the format:

      <verb>_<object>

      Formatting rules:
      - Use exactly one verb and one object.
      - Verb: base form, lowercase, no tense (e.g., open, close, take, put, use, wash, cut, pour, eat, drink, look, walk, call).
      - Object: a single simple noun, lowercase, no adjectives (e.g., door, phone, cup, bed, plate, laptop, sink, window).
      - Use only lowercase letters and the underscore `_`. No spaces, commas, or extra words.

      Examples:
      - someone opening a door to go outside → open_door
      - someone scrolling on a smartphone → use_phone
      - someone eating the last bites of lunch → finish_lunch
      - someone running on a treadmill → exercise_treadmill
      - someone talking on a phone → talk_phone
      - someone streching -> exercise_stretch
      IMPORTANT:
      - Return ONLY the activity label token. 
      - Do NOT add explanations, sentences, or quotes.
      
      Previous actions based on single frame were : {actions}
      """
   })
   
   conversation = [
      {
         "role": "user",
         "content": content
      },
   ]
   
   prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
   
   inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
   
   try:
      with torch.inference_mode(): 
         output = model.generate(
            **inputs,
            max_new_tokens=20,  
            do_sample=False,
            use_cache=False 
         )

         result = processor.decode(output[0], skip_special_tokens=True).strip()
      
      del inputs, output
      torch.cuda.empty_cache()
      
   except torch.cuda.OutOfMemoryError:
      del inputs
      torch.cuda.empty_cache()
      return None
   
   del images, content, conversation
   torch.cuda.empty_cache()
   
   return result

def action_annotate(model, processor, images, local_window = LOCAL_WINDOW_FOR_ACTION_SIZE, image_size=IMAGE_SIZE_VLM_INPUT):
   actions = []
   prev, after = int(local_window // 2), int(local_window // 2)
   for i in tqdm.tqdm(range(len(images)), desc = "Action annotating clip"):
      images_for_act = images[max(0, i-prev):min(len(images), i+after+1)]
      images_for_act = [np.rot90(img, k=-1) for img in images_for_act]
      if LOCAL_WINDOW_FOR_ACTION_SIZE == 1:
         result = recognize_action_single_frame(model, processor, images_for_act[0], image_size) 
      else:
         result = recognize_action_local_window(model, processor, images_for_act, image_size) 
      
      action = parse_llava_output(result)
      actions.append(action)
      
      if (i + 1) % 10 == 0:
         torch.cuda.empty_cache()
   
   return actions


def activity_annotate(model, processor, images, N_frames_for_activity,image_size=IMAGE_SIZE_VLM_INPUT, actions = None):
   activities = []
   T = len(images)

   max_activities = (T + N_frames_for_activity - 1) // N_frames_for_activity  

   for i in tqdm.tqdm(range(max_activities), desc="Activity annotating clip"):
      torch.cuda.empty_cache()
      
      start = i * N_frames_for_activity
      end  = min(start + N_frames_for_activity, len(images)-1)

      images_for_act = images[start:end]
      
      if len(images_for_act) == 0:
         print(f"Warning: No images in block {i}, skipping")
         activities.append("not_annotated")
         continue
         
      images_for_act = [np.rot90(img, k=-1) for img in images_for_act]

      redo = True
      act_image_size = image_size
      act_images = images_for_act[:]
      
      while redo:
         # Ensure we always have at least one image
         if len(act_images) == 0:
            print(f"Error: Image list became empty in block {i}, skipping")
            result = None
            redo = False
            break
            
         try:
            result = recognize_activity(model, processor, act_images, act_image_size, actions=actions)
            redo = False
         except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(act_images) > 5 and act_image_size == image_size:
               current_num = len(act_images)
               new_num = max(5, current_num - 1)
               act_images = random.sample(act_images, new_num)
               print(f"OOM: Reducing image count to {new_num}")
            elif act_image_size == 336 and len(act_images) >= 5:
               act_image_size = 118
               print(f"OOM: Reducing image size to {act_image_size}")
            elif len(act_images) > 2 and act_image_size == 118:
               current_num = len(act_images)
               new_num = max(2, current_num - 1)
               act_images = random.sample(act_images, new_num)
               print(f"OOM: Reducing image count to {new_num}")
            elif len(act_images) == 2 and act_image_size == 118:
               # Try with just 1 image
               act_images = random.sample(act_images, 1)
               print(f"OOM: Reducing to 1 image")
            else:
               print("OOM: Cannot reduce further, skipping")
               result = None
               redo = False
            
      activity = parse_llava_output(result) if result is not None else "not_annotated"
      activities.append(activity)
      
      del act_images
      torch.cuda.empty_cache()

   return activities


def save_images_annotations(images_np, output_path, actions, activities, N_frames_for_activity, gazes=None):
   os.makedirs(output_path, exist_ok=True)
   img_dir = os.path.join(output_path, "frames")
   os.makedirs(img_dir, exist_ok=True)

   csv_records = []

   for i, img in enumerate(images_np):
      img = np.rot90(img, k=-1)
      frame_name = f"frame_{i:04d}.jpg"
      frame_path = os.path.join(img_dir, frame_name)

      cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

      block_idx = i // N_frames_for_activity
      
      h, w = img.shape[:2]
      gaze_x = gazes[i]["gaze_x"] if gazes and i < len(gazes) else None
      gaze_y = gazes[i]["gaze_y"] if gazes and i < len(gazes) else None
      
      if gaze_x is not None and gaze_y is not None:
         gaze_x_rot = h - gaze_y
         gaze_y_rot = gaze_x
      else:
         gaze_x_rot = None
         gaze_y_rot = None

      csv_records.append({
         "frame_index": i,
         "frame_file": frame_name,
         "action": actions[i] if i < len(actions) else "",
         "activity_block_id": block_idx,
         "activity": activities[block_idx] if block_idx < len(activities) else "",
         "gaze_x": gaze_x_rot,
         "gaze_y": gaze_y_rot
      })

   with open(os.path.join(output_path, "actions.txt"), "w") as f:
      for a in actions:
         f.write(a + "\n")

   with open(os.path.join(output_path, "activities.txt"), "w") as f:
      for act in activities:
         f.write(act + "\n")

   df = pd.DataFrame(csv_records)
   df.to_csv(os.path.join(output_path, "annotations.csv"), index=False)

   print(f"Saved:\n  {len(images_np)} frames\n  {len(actions)} actions\n  {len(activities)} activities")

def get_gaze_projection(aea_data_provider, device_time_ns, depth_m=1.0):
   rgb_stream_label = aea_data_provider.vrs.get_label_from_stream_id(RGB_STREAM_ID)
   device_calibration = aea_data_provider.vrs.get_device_calibration()
   rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

   eye_gaze = aea_data_provider.mps.get_general_eyegaze(device_time_ns)
   
   if eye_gaze is None:
      return None, None
   
   gaze_vector_in_cpf = get_eyegaze_point_at_depth(eye_gaze.yaw, eye_gaze.pitch, depth_m)
   T_device_CPF = device_calibration.get_transform_device_cpf()
   gaze_center_in_camera = (
      rgb_camera_calibration.get_transform_device_camera().inverse()
      @ T_device_CPF
      @ gaze_vector_in_cpf
   )
   gaze_projection = rgb_camera_calibration.project(gaze_center_in_camera)
   
   if gaze_projection is None:
      return None, None
   
   return gaze_projection[0], gaze_projection[1]
def get_saliency_map_gaze(gazes_x, gazes_y, H, W, sigma=15):
   saliency = np.zeros((H, W), dtype=np.float32)

   for x, y in zip(gazes_x, gazes_y):
      if 0 <= x < W and 0 <= y < H:
         saliency[int(y), int(x)] += 1.0

   saliency = cv2.GaussianBlur(saliency, (0, 0), sigma)
   saliency = saliency / saliency.max()
   saliency = (saliency * 255).astype(np.uint8)
   return saliency

def get_gazes_for_frames(aea_data_provider, timestamps_ns):
   gazes = []
   for ts in timestamps_ns:
      x, y = get_gaze_projection(aea_data_provider, ts)
      gazes.append({"gaze_x": x, "gaze_y": y})
   return gazes
   
      
def annotate_clip(model, processor, input_clip_path, input_clip_name, output_clip_path, stream_id=RGB_STREAM_ID, n_seconds=EACH_NTH_FRAME_SEC, local_window = LOCAL_WINDOW_FOR_ACTION_SIZE,N_frames_for_activity = N_FRAMES_FOR_ACTIVITY, image_size=IMAGE_SIZE_VLM_INPUT):
   actions_file = os.path.join(output_clip_path, "actions.txt")
   activities_file = os.path.join(output_clip_path, "activities.txt")
   
   actions_exist = os.path.exists(actions_file)
   activities_exist = os.path.exists(activities_file)
   
   aea_data_provider = AriaEverydayActivitiesDataProvider(os.path.join(input_clip_path,input_clip_name))
   
   rgb_images, rgb_timestamps = stream_rgb_vrs_recording(aea_data_provider, RGB_STREAM_ID)

   timestamps_ns = np.array(rgb_timestamps)
   samples = get_every_nth_second_frame(timestamps_ns, n_seconds=n_seconds)
   
   rgb_images_to_select = [rgb_images[k] for k in samples]
   timestamps_to_select = [rgb_timestamps[k] for k in samples]
   
   gazes = get_gazes_for_frames(aea_data_provider, timestamps_to_select)
   
   # Load or annotate actions
   if actions_exist:
      print(f"Loading existing actions for {input_clip_name}")
      with open(actions_file, 'r') as f:
         actions_in_a_clip = [line.strip() for line in f.readlines()]
   else:
      actions_in_a_clip = action_annotate(model, processor, rgb_images_to_select, local_window = local_window, image_size=image_size)
   
   # Load or annotate activities
   if activities_exist:
      print(f"Loading existing activities for {input_clip_name}")
      with open(activities_file, 'r') as f:
         activity_in_a_clip = [line.strip() for line in f.readlines()]
   else:
      activity_in_a_clip = activity_annotate(model, processor, rgb_images_to_select,N_frames_for_activity=N_frames_for_activity, image_size=image_size)
   
   save_images_annotations(rgb_images_to_select, output_clip_path, actions_in_a_clip, activity_in_a_clip, N_frames_for_activity, gazes)
   
def annotate_dataset(model, processor, input_path, output_path, stream_id=RGB_STREAM_ID, n_seconds=EACH_NTH_FRAME_SEC, local_window = LOCAL_WINDOW_FOR_ACTION_SIZE,N_frames_for_activity = N_FRAMES_FOR_ACTIVITY,image_size=IMAGE_SIZE_VLM_INPUT):
   clip_names = [clip for clip in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, clip))]
   for clip_name in tqdm.tqdm(clip_names, desc="Processing clips"):
      output_clip_path = os.path.join(output_path, clip_name)
      
      # Check if both annotations exist
      actions_file = os.path.join(output_clip_path, "actions.txt")
      activities_file = os.path.join(output_clip_path, "activities.txt")
      
      if os.path.exists(actions_file) and os.path.exists(activities_file):
         print(f"Skipping {clip_name}: already fully annotated")
         continue
      elif os.path.exists(actions_file) or os.path.exists(activities_file):
         print(f"Processing {clip_name}: partial annotations found, completing...")
      else:
         print(f"Processing clip: {clip_name}")
         
      try:
         annotate_clip(
            model, 
            processor, 
            input_path, 
            clip_name, 
            output_clip_path, 
            stream_id, 
            n_seconds, 
            local_window,
            N_frames_for_activity,
            image_size
         )
      except RuntimeError as e:
         print(e)
         continue
      
def gpu_worker(gpu_id, task_queue, result_queue, model_name, stream_id, n_seconds, local_window, N_frames_for_activity, image_size):
   device = f"cuda:{gpu_id}"
   torch.cuda.set_device(gpu_id)
   
   print(f"[GPU {gpu_id}] Loading model...")
   model, processor = load_llava_model(model_name, device)
   print(f"[GPU {gpu_id}] Model loaded on {model.device}")
   
   while True:
      try:
         task = task_queue.get(timeout=1)
         if task is None:
            break
            
         input_path, clip_name, output_path = task
         output_clip_path = os.path.join(output_path, clip_name)
         
         print(f"[GPU {gpu_id}] Processing {clip_name}")
         
         try:
            annotate_clip(
               model, 
               processor, 
               input_path, 
               clip_name, 
               output_clip_path, 
               stream_id, 
               n_seconds, 
               local_window,
               N_frames_for_activity,
               image_size
            )
            result_queue.put((clip_name, "success", None))
         except Exception as e:
            result_queue.put((clip_name, "error", str(e)))
            
      except Exception:
         continue
   
def annotate_dataset_mp(model_name,num_gpus, input_path, output_path, stream_id=RGB_STREAM_ID, n_seconds=EACH_NTH_FRAME_SEC, local_window = LOCAL_WINDOW_FOR_ACTION_SIZE,N_frames_for_activity = N_FRAMES_FOR_ACTIVITY,image_size=IMAGE_SIZE_VLM_INPUT):
   clip_names = [name for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name))]
   clip_names = [name for name in clip_names if not os.path.exists(os.path.join(output_path, name))]
   total_clips = len(clip_names)
   
   task_queue = mp.Queue()
   result_queue = mp.Queue()
   
   for clip_name in clip_names:
      task_queue.put((input_path, clip_name, output_path))
      
   for _ in range(num_gpus):
      task_queue.put(None)
      
   workers = []
   for gpu_id in range(num_gpus):
      p = mp.Process(
         target=gpu_worker,
         args=(gpu_id, task_queue, result_queue, model_name, stream_id, n_seconds, local_window, N_frames_for_activity, image_size)
      )
      p.start()
      workers.append(p)
   
   completed = 0
   errors = []
   with tqdm.tqdm(total=total_clips, desc="Total progress") as pbar:
      while completed < total_clips:
         clip_name, status, error = result_queue.get()
         completed += 1
         pbar.update(1)
         if status == "error":
            errors.append((clip_name, error))
            pbar.set_postfix({"errors": len(errors)})
   
   for p in workers:
      p.join()
      
   if errors:
      for clip_name, error in errors:
         print(f"  - {clip_name}: {error}")
      
def main():
   NUM_GPUS = torch.cuda.device_count() 
   quant = FOUR_BIT_QUANTIZATION or EIGHT_BIT_QUANTIZATION 

   if NUM_GPUS == 1 or quant:
      device="cuda:0" if NUM_GPUS == 1 else "cuda"
      model, processor = load_llava_model(VLM_ANNOTATOR, device)
      print(model.device)

      annotate_dataset(
         model, 
         processor, 
         INPUT_DATA_FOLDER, 
         OUTPUT_DATA_FOLDER, 
         RGB_STREAM_ID, 
         EACH_NTH_FRAME_SEC, 
         LOCAL_WINDOW_FOR_ACTION_SIZE,
         N_FRAMES_FOR_ACTIVITY,
         IMAGE_SIZE_VLM_INPUT
      )
   elif NUM_GPUS > 1:
      try:
         mp.set_start_method('spawn', force=True)
      except RuntimeError:
         pass 
      
      annotate_dataset_mp(
         VLM_ANNOTATOR,
         NUM_GPUS,
         INPUT_DATA_FOLDER, 
         OUTPUT_DATA_FOLDER, 
         RGB_STREAM_ID, 
         EACH_NTH_FRAME_SEC, 
         LOCAL_WINDOW_FOR_ACTION_SIZE,
         N_FRAMES_FOR_ACTIVITY,
         IMAGE_SIZE_VLM_INPUT
      )
   else:
      raise Exception("No GPU detected.")
   
if __name__ == "__main__":
   main()
      