#!/usr/bin/env python
# coding: utf-8

# ## YOLOv3

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
net = cv2.dnn.readNet("/System/Volumes/Data/Users/jean622/darknet/yolov3.weights", "/System/Volumes/Data/Users/jean622/darknet/cfg/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Load image
image_path = "/Users/jean622/Desktop/img.jpg"
image = cv2.imread(image_path)
height, width, channels = image.shape

# Detect objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Information showing how confident the model is about the located object
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Non-maximum suppression

# Load labels
with open("/Users/jean622/darknet/data/coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Draw bounding boxes
for i in range(len(boxes)):
    if i in indexes:
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

# Convert BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.axis('off') # To remove the axes
plt.show()


# In[ ]:


#pip install opencv-python


# ## YOLOv8

# In[ ]:


get_ipython().system("yolo predict model=yolov8n.pt source='img.jpg'")


# In[ ]:


import webbrowser
import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to the results directory
results_path = "runs/detect/predict"
image_filename = "img.jpg"

# Full path to the image
image_path = os.path.join(results_path, image_filename)

# Open the image
image = Image.open(image_path)

# Display the image
plt.imshow(image)
plt.axis('off') # To hide the axis
plt.show()

