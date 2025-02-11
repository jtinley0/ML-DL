#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


import os
HOME = os.getcwd()
print("HOME:", HOME)


# In[5]:


#import segment anything
get_ipython().run_line_magic('cd', '{HOME}')

import sys
get_ipython().system("{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'")


# In[6]:


get_ipython().system('pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision')


# Download SAM weights

# In[7]:


get_ipython().run_line_magic('cd', '{HOME}')
get_ipython().system('mkdir {HOME}/weights')
get_ipython().run_line_magic('cd', '{HOME}/weights')

get_ipython().system('wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')


# In[8]:


CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))


# Example from data

# In[10]:


image1= '/content/drive/Shareddrives/Computer_Vision/train/fdc07c32-6192c67d.jpg'


# Load model

# In[9]:


import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"


# In[11]:


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)


# Automated Mask Generation in SAM

# In[12]:


mask_generator = SamAutomaticMaskGenerator(sam)


# In[13]:


import cv2
import supervision as sv

image_bgr = cv2.imread(image1)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_rgb)


# In[14]:


#OUTPUT
print(sam_result[0].keys())


# SamAutomaticMaskGenerator returns a list of masks, where each mask is a dict containing various information about the mask:
# 
# - segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
# - area - [int] - the area of the mask in pixels
# - bbox - [List[int]] - the boundary box of the mask in xywh format
# - predicted_iou - [float] - the model's own prediction for the quality of the mask
# - point_coords - [List[List[float]]] - the sampled input point that generated this mask
# - stability_score - [float] - an additional measure of mask quality
# - crop_box - List[int] - the crop of the image used to generate this mask in xywh format
# 

# In[15]:


#visualiza the result
mask_annotator = sv.MaskAnnotator()

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)


# In[16]:


image2= '/content/drive/Shareddrives/Computer_Vision/train/b05160eb-73c0e8bb.jpg'


# In[18]:


image_bgr2 = cv2.imread(image2)
image_rgb2 = cv2.cvtColor(image_bgr2, cv2.COLOR_BGR2RGB)

sam_result2 = mask_generator.generate(image_rgb2)


# In[19]:


#visualiza the result
mask_annotator = sv.MaskAnnotator()

detections2 = sv.Detections.from_sam(sam_result=sam_result2)

annotated_image2 = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections2)

sv.plot_images_grid(
    images=[image_bgr2, annotated_image2],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

