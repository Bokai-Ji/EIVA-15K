import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time
import json
import base64
from openai import AzureOpenAI

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Build GPT4o client
AZURE_OPENAI_ENDPOINT = "https://safemo-gpt.openai.azure.com/"
AZURE_OPENAI_API_KEY = "1d7dbbc5eeb4402a9a9de3df9b72d34b"
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-03-15-preview"
)

image_root = "/home/jibokai/affordance/LISA/affordance_dataset/rgb"
proposals = json.load(open("cot_logs/72B/Initial_Proposal.json", "r"))

os.makedirs("cot_logs/72B/sam", exist_ok=True)
for i, (image_name, bbox) in enumerate(proposals.items()):
    image_path = os.path.join(image_root, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    input_box = np.array(bbox)
    mask, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    plt.figure()
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off'); plt.show(); plt.tight_layout(pad=0)
    plt.savefig(f"cot_logs/72B/sam/{image_name}")
    plt.close()