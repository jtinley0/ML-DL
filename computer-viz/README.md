# Autonomous driving: Multi-label image classification and instance segmentation


## Abstract
A critical challenge in autonomous driving is developing robust algorithms that enable road objects to be detected accurately in dynamic conditions. Using a small randomly generated subset of the open-source Berkeley DeepDrive BDD100K dataset, I trained a custom convolutional neural network and used transfer learning with a ResNet50 model to do multi-label image classification. A YOLOv8 and Segment Anything model (SAM) was trained to do instance segmentation. Given the small training subset of BDD100K (chosen to fit compute capacity), the models struggled to learn the nuances between classes and there is significant room for improvement in multi-label image classification.  Greater success was achieved in instance segmentation, where both the YOLO and SAM models were able to accurately detect different objects.

## Computer Vision Problems
### > Multi-label Image Classification
### > Instance Segmentation

## Approaches
### Multi-label Image Classification
- Custom CNN
- Transfer learning with ResNet50
### Instance Segmentation
- SAM
- YOLO v8

## Dataset Summary
### Berkeley DeepDrive BDD100K Dataset
 - Images were derived from 100K short road videos (~40s - 720p - 30 fps) → ~120M
 - Annotations/keyframes (taken at 10 second of every video) includes:
   - image tagging
   - object bounding boxes
   - full-frame image segmentation

### BDD100K version used: a subset with approximately 80K jpg images total
- BDD100K version used: a subset with approximately 80K jpg images total
- Train folder (~70K jpg images )
- Test folder (10K jpg images)
- train.json → labels
- test.json → labels
- Missing values: 137 labels in train.json
- Original Image size: 1280x720
- 12 object classes (e.g., car, truck, person)
- Multiple cities, weather conditions, and times of day

Berkeley DeepDrive. Accessed August 11, 2023. https://bdd-data.berkeley.edu/

## Exploratory Data Analysis - Subsetting and Class Imbalance
### Class balance
- Bus, bike, rider, motor, and train classes were underrepresented in data (n<500)
- Car, driveable area, and lane are over-represented (ns>2500)
- Addressing class imbalance leads to better generalization to new data; improve model sensitivity to minority class

### Color Distribution
 - All color channels for sampled images are positively skewed
 - Dawn, dusk, and night have high pixel spikes in all three channels
 - Traffic lights and brake lights appear to cause intense spikes
 - Equalized the RGB distributions of images using Contrast Limited Adaptive Histogram Equalization (CLAHE)

### Image Preprocessing and Feature Engineering
- Preprocessing
 - Subsetting:
   - randomly downsampled to fit our compute capacity (train n = 3000; test n = 1000)
   - We removed one severely underrepresented class (i.e., train n=5)
  - Resizing
   - All images resized to 224x224
 - Normalization
  - Images pixel values normalized from [0,255] to [0,1]
  - Data Augmentation
   - Randomly applied the below augmentation techniques:
   - Rotation between -30 and 30 degrees
   - Shift width and height by 20%
   - Shear transformation
   - Flip images horizontally
   - Fill any newly created pixels after rotation or shift 13

## Multi-label Image Classification (Model Architectures)
- Custom CNN Model Architecture
  - Input layer
    - 224x224 with 3 RGB color channels
    - Max Pooling - pool size (2,2)
  - Hidden Convolutional Layer 1
    - 64 filters; kernel size of (3,3)
    - ReLU Activation function
    - Batch Normalization
  - Hidden Convolutional Layer 2
   - 64 filters; kernel size of (3,3)
   - ReLU Activation function
   - Max Pooling - pool size (2,2)
 - Flatten layer
- Fully connected layer 1
  - 128 neurons
  - Activation function - ReLU
  - Dropout - rate 0.2
- Output layer
  - Activation function - Softmax 15

## ResNet Model Architecture
- ResNet50
- Freeze layer
  - Freezes all layers except the last 10 layers
- Add new top layers
  - 2 Fully connected layers
- Fully connected layer 1
  - 256 neurons
  - Activation function - ReLU
  - Dropout - rate 0.3
- Fully connected layer 2
  - 128 neurons
  - Activation function - ReLU
  - Dropout - rate 0.3

## Takeaways
 - Multi-label Image Classification
 - - Models had mediocre performance in predicting majority classes, and performed poorly with minority classes
   - Resnet50 performed better than custom CNN with with test accuracy = 45%; loss = 0.4
 - Instance segmentation
 - - Both SAM and YOLO can effectively detect different objects
 - Key Limitation:
 - - Dataset size (i.e., train n=3000): affected our models’ test accuracy especially in multi-label classification. With more training data our models could have better learned the nuances between object classes






