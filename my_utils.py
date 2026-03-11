import os

# Packages used for Qwen2.5-VL
import io
import ast
import json
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import ImageColor
from PIL import Image, ImageDraw, ImageFont

# Packages used for GPT4o api
import sys
import cv2
import time
import base64
import argparse
import numpy as np
from openai import AzureOpenAI
import matplotlib.pyplot as plt

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def build_qwen25vl_model(actor_path):
    actor = Qwen2_5_VLForConditionalGeneration.from_pretrained(actor_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
    processor = AutoProcessor.from_pretrained(actor_path)
    return actor, processor

def build_gpt4o_client():
    AZURE_OPENAI_ENDPOINT = "https://safemo-gpt.openai.azure.com/"
    AZURE_OPENAI_API_KEY = "1d7dbbc5eeb4402a9a9de3df9b72d34b"
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2023-03-15-preview"
    )
    return client

def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height, vis=False):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    # print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    # font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color)

    # Display the image
    if vis:
      img.show()
    
    return img, [int(abs_x1 / width * input_width), int(abs_y1 / height * input_height), int(abs_x2 / width * input_width), int(abs_y2 / height * input_height)]

def plot_points(im, text, input_width, input_height):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors
  xml_text = text.replace('```xml', '')
  xml_text = xml_text.replace('```', '')
  data = decode_xml_points(xml_text)
  if data is None:
    img.show()
    return
  points = data['points']
  description = data['phrase']

  # font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

  for i, point in enumerate(points):
    color = colors[i % len(colors)]
    abs_x1 = int(point[0])/input_width * width
    abs_y1 = int(point[1])/input_height * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color)
  
  img.show()

# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
  
def inference_qwen25vl(model, processor, img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):
  if isinstance(img_url, str):
    image = Image.open(img_url)
  elif isinstance(img_url, Image.Image):
    image = img_url
  else:
    raise ValueError("img_url should be a string or a PIL image")
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "image": img_url
        }
      ]
    }
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  # print("input:\n",text)
  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

  output_ids = model.generate(**inputs, max_new_tokens=1024)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  # print("output:\n",output_text[0])

  input_height = inputs['image_grid_thw'][0][1]*14
  input_width = inputs['image_grid_thw'][0][2]*14

  return output_text[0], input_height, input_width

def inference_gpt4o(client, system_prompt="You are a helpful assistant.", user_prompt="", images=[], fewshot_examples=[], max_retry=5):
    assert len(system_prompt) > 0 or len(user_prompt) > 0, "At least one prompt must be provided"
    # Build content (user prompt + images + few-shot examples)
    content = [{"type": "text", "text": user_prompt}]
    content = content + [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(open(image, 'rb').read()).decode('utf-8')}" if image.endswith(".png") else f"data:image/jpeg;base64,{base64.b64encode(open(image, 'rb').read()).decode('utf-8')}"
            }
        } for image in images
    ]
    # Build message
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    if len(fewshot_examples) > 0:
      for task, image_path, feedback in fewshot_examples.items():
          messages.append(
            {
              "role": "user",
              "content": [
                {"type": "text", "text": task},
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}" if image_path.endswith(".png") else f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"
                  }
                }
              ]
            }
          )
          messages.append({"role": "assistant", "content": feedback})
    # Request
    for i in range(max_retry):
        try:
            response = client.chat.completions.create(model="gpt-4o", messages=messages).choices[0].message.content
            return response
        except Exception as e:
            # pass # Mutely retry
            print(e)
        time.sleep(0.5)
        
    