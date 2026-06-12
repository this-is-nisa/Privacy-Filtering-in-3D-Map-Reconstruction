# This script will extract object labels from text input using an LLM and generate masks
# using code from grounded_sam2_tracking_demo.py

# using the grounded DINO local model

import os
from urllib import response
import cv2
import numpy as np
import supervision as sv
import torch
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from ollama import chat
from pydantic import BaseModel, Field, field_validator

###
# Hyper Params #
GROUNDING_DINO_CONFIG = "automatic_labeling/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "automatic_labeling/grounding_dino/checkpoint/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
VIDEO_PATH = "../data/table_video.mp4"
#LABEL_PROMPT = "" # MUST be in this format with dot at end # change for list for multiple obj remove
#TEXT_PROMPT = "Filter out the wallet from the video.
OUTPUT_VIDEO_PATH = "../data/table_video_tracked.mp4"
SOURCE_VIDEO_FRAME_DIR = "../data/table_video_frames"
SAVE_TRACKING_RESULTS_DIR = "../data/table_video_results"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
###




### Environment settings and model initialization for Grounding DINO, SAM2, and OLLAMA ###

# build grounding dino model from local path
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path = GROUNDING_DINO_CHECKPOINT,
    device = DEVICE
)

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_1.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint) # allows tracking -- the spatio-temporal memory bank to track the objects in video
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint) # loads nn structure
image_predictor = SAM2ImagePredictor(sam2_image_model) # locates object within specific frame (given frame and bounding box prompt)

# video stream information
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(video_info)
frame_generator = sv.get_video_frames_generator # frame generator func to read video frames one by one without loading entire video into memory

# saving video to img frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg") as sink:
    for frame in tdm(frame_generator, desc="Saving Video Frames..."):
        sink.save_image(frame)

# scan all the JPEG frame names in this directory 
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR) 
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0])) # sort the frame names in ascending order

# init video predictor state
inference_state = video_predictor.init_state(video_path=VIDEO_PATH) # intialize the video predictor's memory state for tracking objs
ann_frame_idx = 0 # the frame index to make initial mask





### Extracting the object label from the text prompt using Ollama and structuring it with pydantic for Grounding DINO ###

# text prompt input
text = input("Write prompt to filter out something from the video: ") # Ex: Filter out wallet from the room video.

TEXT_PROMPT = text # text prompt

# change this in future to extract multiple labels
class VideoFilterLabel(BaseModel):
    """Pydantic model to structure the response from the LLM for video object labels to filter.
    """
    object_label: str = Field(description="The exact name of the object to filter or remove from the video.")
    
    
    # validator to make ground_dino format (lowercase with period)
    @field_validator('object_label')
    @classmethod
    def format_obj_labels(cls, value:str) -> str:
        formatted_label = value.strip().lower().rstrip('.')
        return f"{formatted_label}."

# generate reponse in json schema format
response = chat(
model='gemma3',
messages=[{'role': 'user', 'content': TEXT_PROMPT}],
format = VideoFilterLabel.model_json_schema() # Use pydantic schema
)

# validate and parse the response to get the object label in the correct format for grounding dino
validated_output = VideoFilterLabel.model_validate_json(response.message.content) # put in validation format

print(validated_output.model_dump_json(indent=2))

LABEL_PROMPT = validated_output.object_label  # label output



### Prompt Grounding Dino locally ###

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx]) # path for initial frame
image_source, image = load_image(img_path) # load the image 

boxes, confidences, labels = predict(
    model= grounding_model, 
    image=image, 
    caption=LABEL_PROMPT,
    box_threshold=BOX_THRESHOLD, 
    text_threshold=TEXT_THRESHOLD
)

# process the box prompt for SAM2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h]) # convert to absolute coordinates
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy # convert to xyxy format for sam2
confidences = confidences.numpy().tolist()
class_names = labels

print(input_boxes)


# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(image_source) # set the image for sam predictor

OBJECTS = class_names
print(OBJECTS)

# speed up inference with autocast
torch.autocast(device=DEVICE, dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# prompt SAM 2 image predictor to get the mask for the object
masks, score, logits = image_predictor.predict( # all masks for all objects FOR FRAME 0
    point_coords=None, 
    point_labels=None, 
    box=input_boxes,
    multimask_output=False,
)

# convert the mask shape to (n, H, W))
if masks.ndim == 4:
    masks = masks.squeeze(0) # remove the extra dimension if only one box is provided



### Register each object's positive points to video predictor with seperate add_new_points call ### 
### Put the points/box/mask prompt into the video predictor's memory state for tracking in the video ###

# check if the prompt type for video is valid
assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "Invalid PROMPT_TYPE_FOR_VIDEO: SAM2 video predictor only support point/box/mask prompt"

# add the data to inference_state memory

# If using point prompts, uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from the mask for each object
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10) # random 10 points from mask (because easier to parse points than mask to video tracker)
    
    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):  # loops through each object and gives int id to each distinct entity in video
        labels = np.ones((points.shape[0]), dtype=np.int32)  # all mask points are positive (1)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box( # video predictor pushes the mask points data into SAM2's memory state (inference state)
            inference_state = inference_state, 
            frame_idx = ann_frame_idx, 
            obj_id = object_id, 
            points = points, 
            labels = labels
        )
        
# If using box prompts
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state, 
            frame_idx=ann_frame_idx,
            obj_id=object_id, 
            box=box 
        )
    
# If using mask prompts (more straightforward but more computationally heavy to parse))
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)  # all mask points are positive (1
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state, 
            frame_idx=ann_frame_idx, 
            obj_id=object_id, 
            mask=mask, 
           #labels = labels
        )

else:
    raise NotImplementedError("INVALID PROMPT TYPE: SAM 2 video predictor only supports point/box/mask prompts")    




### Propagate the video predictor to get the SEGMENTATION results for each frame ###
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    
    
    

### Visualize the segment results across the video and save them ###

# make directory to save if doesnt exist
if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
    os.makedirs(SAVE_TRACKING_RESULTS_DIR)
    
ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

# Put annotation on each frame and save the annotated frames
for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx])) # read the original frame image

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0) # (n_objects, H, W)
    
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks), # (n, 4))
        masks=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32)
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
    
    
    
    
    ### Convert annotated frames to video ###
    
    create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
        
        