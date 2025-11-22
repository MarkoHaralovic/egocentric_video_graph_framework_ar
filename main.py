import os
import subprocess
from projectaria_tools.projects.aea import (
    AriaEverydayActivitiesDataPathsProvider,
    AriaEverydayActivitiesDataProvider)
import numpy as np

import matplotlib.pyplot as plt

from projectaria_tools.core import calibration, mps
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.mps import get_eyegaze_point_at_depth

from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId

from segment_anything import sam_model_registry, SamPredictor
import cv2


# VLM part
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16)

print(model.device)

# SAM part
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_predictor = SamPredictor(sam)

sequence_name_rec1 = "loc5_script4_seq6_rec1"

sequence_path_rec1 = "output/" + sequence_name_rec1

# create AEA data provider
aea_data_provider_1 = AriaEverydayActivitiesDataProvider(sequence_path_rec1)

RGB_STREAM_ID = StreamId("214-1")

device_time_vec_1 = aea_data_provider_1.vrs.get_timestamps_ns(RGB_STREAM_ID, TimeDomain.DEVICE_TIME)

device_time_ns_1 = device_time_vec_1[100]

sentence = aea_data_provider_1.speech.get_sentence_data_by_timestamp_ns(
    device_time_ns_1, TimeQueryOptions.CLOSEST
)
print(f"Sentence from sequence 1 at device timestamp {device_time_ns_1} is : '{sentence}'")
 

def segment_gazed_object(aea_data_provider, device_time_ns, depth_m):
    rgb_stream_label = aea_data_provider.vrs.get_label_from_stream_id(RGB_STREAM_ID)
    device_calibration = aea_data_provider.vrs.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    image_data = aea_data_provider.vrs.get_image_data_by_time_ns(
        RGB_STREAM_ID, device_time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.BEFORE
    )
    image = image_data[0].to_numpy_array()

    eye_gaze = aea_data_provider.mps.get_general_eyegaze(device_time_ns, TimeQueryOptions.CLOSEST)
    if not eye_gaze:
        print("Eye gaze not available")
        return

    gaze_vector_in_cpf = mps.get_eyegaze_point_at_depth(eye_gaze.yaw, eye_gaze.pitch, depth_m)
    T_device_CPF = device_calibration.get_transform_device_cpf()
    gaze_center_in_camera = (
        rgb_camera_calibration.get_transform_device_camera().inverse()
        @ T_device_CPF
        @ gaze_vector_in_cpf
    )
    gaze_projection = rgb_camera_calibration.project(gaze_center_in_camera)

    if gaze_projection is None:
        print("Gaze projection out of camera plane")
        return

    # Segmentation part
    sam_predictor.set_image(image)
    x, y = gaze_projection.flatten()
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    fig, axes = plt.subplots(1, len(masks), figsize=(15,5))

    for i, (mask, ax) in enumerate(zip(masks, axes)):
        ax.imshow(image)
        ax.imshow(mask, alpha=0.5)
        ax.plot(x, y, 'r+', markersize=20)
        ax.set_title(f"Mask {i}")
        ax.axis('off')

    # First the masks wil show, and (for this image) the second mask is picked
    plt.show()

    bbox = mask_to_bbox(masks[2])
    if bbox is None:
        print("No valid segmentation area")
        return

    cropped = crop_from_bbox(image, bbox)

    recognize_action(cropped)

def recognize_action(image_np):
    """
    Takes a numpy array image (H, W, 3 – RGB)
    Returns a string with VLM-predicted action.
    """

    print(image_np.shape)

    img = Image.fromarray(image_np)

    print("test1")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text", 
                    "text": "This image shows only the object the human is fixating on in a video. Based on this object, what activity is the human likely performing? Answer with a short phrase."
                },
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)

    # This step might take a long while if running on CPU (which it will, since I did not add GPU support)
    output = model.generate(**inputs, max_new_tokens=100)

    result = processor.decode(output[0], skip_special_tokens=True)

    print(result)

def mask_to_bbox(mask):
    ys, xs = mask.nonzero()

    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    return x1, y1, x2, y2

def crop_from_bbox(image, bbox, padding=10):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image[y1:y2, x1:x2]


eye_gaze_depth = 1.0

segment_gazed_object(aea_data_provider_1, device_time_ns_1, eye_gaze_depth)
