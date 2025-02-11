# Autonomous driving: Multi-label image classification and segmentation


## Abstract
A critical challenge in autonomous driving is developing robust algorithms that enable road objects to be detected accurately in dynamic conditions. Using a small randomly generated subset of the open-source Berkeley DeepDrive BDD100K dataset, I trained a custom convolutional neural network and used transfer learning with a ResNet50 model to do multi-label image classification. I then trained a YOLOv8 and Segment Anything model (SAM) to do instance segmentation. Given our small training subset of BDD100K (chosen to fit our compute capacity), our models struggled to learn the nuances between classes and there is significant room for improvement in multi-label image classification. We experienced greater success instance segmentation, where both our YOLO and SAM models were able to accurately detect different objects.

## Computer Vision Problems
### > Multi-label Image Classification: (i.e., classify the images with different labels)
### > Instance Segmentation (i.e., segment the images to improve object detection)

## Approach
### Multi-label Image Classification
➢ Custom CNN
➢ Transfer learning with ResNet50
### Instance Segmentation
➢ SAM
➢ YOLO v8


