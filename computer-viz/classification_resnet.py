#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[10]:


train_path = '/content/drive/Shareddrives/Computer_Vision/train'
test_path = '/content/drive/Shareddrives/Computer_Vision/test'


# In[11]:


import os

#check number of images in each file
train_num = os.listdir(train_path)
test_num = os.listdir(test_path)
train_len= len(train_num)
test_len= len(test_num)

print(f"Number of files in the train folder: {train_len}")
print(f"Number of files in the train folder: {test_len}")


# ## Train data

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image

from sklearn.metrics import classification_report

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras import callbacks, models, layers, optimizers, regularizers

import warnings
warnings.filterwarnings("ignore")


# ### one hot encoding label

# In[13]:


df_train = pd.read_json('/content/drive/Shareddrives/Computer_Vision/train30.json')
df_train.head()


# In[14]:


df_train.shape


# In[15]:


def clean_labels(example):
  example_df = pd.DataFrame.from_records(example)
  example_df = example_df['category'].unique().tolist()
  return ','.join(example_df)


# In[16]:


df_train['clean_labels'] = df_train['labels'].map(clean_labels)
df_train


# In[17]:


#create target_list, convert to set, convert to list
target_list = ",".join(df_train.clean_labels).split(",")
target_list = list(set(target_list))
target_list


# In[18]:


#create copy of df, for loop to search if clean_labels contains target_list, convert to int
data_train = df_train.copy()
for target in target_list:
    data_train[target] = data_train['clean_labels'].str.contains(target)
    data_train[target] = data_train[target].astype(int)


# In[19]:


#drop attributes and timestamp columns, not neccessary for our model
columns = ['clean_labels','attributes', 'timestamp','labels']
data_train = data_train.drop(columns = columns)

#review columns have been dropped
data_train


# In[20]:


#retreiving labels
labels = list(data_train.columns.values)
labels = labels[3:]
print(labels)

#creating dataframe
counts = []
for label in labels:
    counts.append((label, data_train[label].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'Occurrence'])
df_stats = df_stats.sort_values(['Occurrence']).reset_index(drop=True)
df_stats


# In[21]:


# Set seaborn style and plt size
sns.set_style("darkgrid")
plt.figure(figsize=(12, 10))

# Create ascending order
order = ['train', 'motor', 'rider', 'bike', 'bus', 'truck', 'person', 'traffic light',
         'traffic sign', 'lane', 'drivable area', 'car']

# Create seaborn visual
ax = sns.barplot(x=order, y=data_train[order].sum().values)

# Title and labels
plt.title("Image Classification Labels", fontsize=22)
plt.ylabel('Total Occurrences', fontsize=20)
plt.xlabel('Labels', fontsize=20)

# Adding the text labels
rects = ax.patches
labels = data_train[order].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{label}', ha='center', va='bottom', fontsize=16)

plt.xticks(rotation=45, fontsize=16, ha='right')
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()


# In[22]:


#remove 'train', since it has too few of labels
columns = ['train']
data_train = data_train.drop(columns = columns)

data_train


# ## Test data

# In[23]:


df_test = pd.read_json('/content/drive/Shareddrives/Computer_Vision/test10.json')
df_test.head()


# In[24]:


df_test.shape


# In[25]:


df_test['clean_labels'] = df_test['labels'].map(clean_labels)
df_test


# In[26]:


#create target_list, convert to set, convert to list
target_list = ",".join(df_test.clean_labels).split(",")
target_list = list(set(target_list))
target_list


# In[27]:


#create copy of df_val, for loop to search if clean_labels contains target_list, convert to int
data_test = df_test.copy()
for target in target_list:
    data_test[target] = data_test['clean_labels'].str.contains(target)
    data_test[target] = data_test[target].astype(int)


# In[28]:


#drop attributes and timestamp columns, not neccessary for our model
columns = ['clean_labels','attributes', 'timestamp','labels']
data_test = data_test.drop(columns = columns)

data_test


# In[29]:


#retreiving labels
labels = list(data_test.columns.values)
labels = labels[3:]
print(labels)

#creating dataframe
counts = []
for label in labels:
    counts.append((label, data_test[label].sum()))
df_stats_2 = pd.DataFrame(counts, columns=['Labels', 'Occurrence'])
df_stats_2 = df_stats_2.sort_values(['Occurrence']).reset_index(drop=True)
df_stats_2


# In[30]:


# Set seaborn style and plt size
sns.set_style("darkgrid")
plt.figure(figsize=(12, 10))

# Create ascending order
order = ['train', 'motor', 'rider', 'bike', 'bus', 'truck', 'person', 'traffic light',
         'traffic sign', 'lane', 'drivable area', 'car']

# Create seaborn visual
ax = sns.barplot(x=order, y=data_test[order].sum().values, order=order)

# Title and labels
plt.title("Image Classification Labels", fontsize=22)
plt.ylabel('Total Occurrences', fontsize=20)
plt.xlabel('Labels', fontsize=20)

# Adding the text labels
rects = ax.patches
labels = data_test[order].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{label}', ha='center', va='bottom', fontsize=16)

plt.xticks(rotation=60, fontsize=16, ha='right')
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()


# In[31]:


#remove 'train', since there is too few labels
columns = ['train']
data_test = data_test.drop(columns = columns)

data_test


# In[32]:


#update target list
target_list= ['bus', 'person', 'drivable area', 'traffic light', 'motor','lane', 'traffic sign', 'bike', 'rider', 'car', 'truck']


# ## ImageDataGenerator

# In[33]:


IMG_SIZE= (224,224)
train_datagen = ImageDataGenerator(
    rotation_range=30,      # Random rotation between -30 and 30 degrees
    width_shift_range=0.2,  # xRandomly shift the width by 20%
    height_shift_range=0.2, # Randomly shift the height by 20%
    shear_range=0.2,        # Randomly apply shear transformation
    zoom_range=0.2,         # Randomly zoom by 20%
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill any newly created pixels after rotation or shift
)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[34]:


train_set = train_datagen.flow_from_dataframe(data_train, directory=train_path,
                                              x_col ='name', y_col = target_list,
                                              class_mode = "multi_output", seed= 24,
                                                batch_size=3000, target_size = IMG_SIZE)

test_set = test_datagen.flow_from_dataframe(data_test, directory=test_path,
                                          x_col ='name', y_col = target_list,
                                          class_mode = "multi_output", seed= 24,
                                          batch_size=1000, target_size = IMG_SIZE)


# In[35]:


#create datasets for train, test
train_images, train_labels = next(train_set)
test_images, test_labels = next(test_set)


# In[36]:


len(train_labels)


# In[37]:


#reshape labels
train_labels_ = np.array(train_labels).reshape(len(train_labels), -1).T
test_labels_ = np.array(test_labels).reshape(len(test_labels), -1).T


# In[38]:


#reshape images
train_img = train_images.reshape(train_images.shape[0], -1)
test_img = test_images.reshape(test_images.shape[0], -1)


# In[39]:


#explore dataset, checking number of samples and shape
m_train = train_img.shape[0]
m_test = test_img.shape[0]

print ("Number of training samples: " + str(m_train))
print ("Number of testing samples: " + str(m_test))
print ("train_images shape: " + str(train_images.shape))
print ("train_img shape: " + str(train_img.shape))
print ("train_labels_ shape: " + str(train_labels_.shape))
print ("test_images shape: " + str(test_images.shape))
print ("test_img shape: " + str(test_img.shape))
print ("test_labels_ shape: " + str(test_labels_.shape))


# ## Check for images

# In[ ]:


#observing images are viewable
array_to_img(train_images[0])


# In[ ]:


#observing labels match the image
pd.Series(train_labels_[0], index=target_list)


# ## Transfer Learning

# In[ ]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


def plot_training(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # summarize history for accuracy
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'test'], loc='upper left')

    plt.show()


# ## Model 1: Base model

# In[ ]:


# Load the pre-trained ResNet50 model
res_model1 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


# Freeze some layers in the pre-trained model
for layer in res_model1.layers[:-10]:
    layer.trainable = False

# Add new top layers for multi-label classification
x = res_model1.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(target_list), activation='sigmoid')(x)

model_transfer1 = Model(inputs=res_model1.input, outputs=predictions)

# Compile the model with appropriate optimizer and loss function
optimizer = Adam(learning_rate=0.001)
model_transfer1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# In[ ]:


# Train the model using the generators with adjusted batch size and more epochs
history_res1 = model_transfer1.fit(train_images, train_labels_,
                                 epochs=10, batch_size=64,
                                 validation_data=(test_images, test_labels_),
                                 verbose=1)


# In[ ]:


plot_training(history_res1)


# In[ ]:


model_transfer1.save('/content/drive/Shareddrives/Computer_Vision/base_resnet_model.h5')


# ## Model 2: Add dropout

# In[ ]:


# Load the pre-trained ResNet50 model
res_model2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


# Freeze some layers in the pre-trained model
for layer in res_model2.layers[:-10]:
    layer.trainable = False

# Add new top layers for multi-label classification
x = res_model2.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(target_list), activation='sigmoid')(x)

model_transfer2 = Model(inputs=res_model2.input, outputs=predictions)

# Compile the model with appropriate optimizer and loss function
optimizer = Adam(learning_rate=0.001)
model_transfer2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# In[ ]:


# Train the model using the generators with adjusted batch size and more epochs
history_res2 = model_transfer2.fit(train_images, train_labels_,
                                 epochs=10, batch_size=64,
                                 validation_data=(test_images, test_labels_),
                                 callbacks=[early_stopping],
                                 verbose=1)


# In[ ]:


plot_training(history_res2)


# In[ ]:


model_transfer2.save('/content/drive/Shareddrives/Computer_Vision/drop_resnet_model.h5')


# ## Model 3: Add Batch normalization

# In[ ]:


# Load the pre-trained ResNet50 model
res_model3 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


# Freeze some layers in the pre-trained model
for layer in res_model3.layers[:-10]:
    layer.trainable = False

# Add new top layers for multi-label classification
x = res_model3.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(target_list), activation='sigmoid')(x)

model_transfer3 = Model(inputs=res_model3.input, outputs=predictions)

# Compile the model with appropriate optimizer and loss function
optimizer = Adam(learning_rate=0.001)
model_transfer3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# In[ ]:


# Train the model using the generators with adjusted batch size and more epochs
history_res3 = model_transfer3.fit(train_images, train_labels_,
                                 epochs=10, batch_size=64,
                                 validation_data=(test_images, test_labels_),
                                 callbacks=[early_stopping],
                                 verbose=1)



# In[ ]:


plot_training(history_res3)


# ## Prediction & Evaluation

# In[8]:


# Model 2 is the final model

from keras.models import load_model
loaded_model = load_model('/content/drive/Shareddrives/Computer_Vision/drop_resnet_model.h5')


# In[ ]:


loaded_model.summary()


# In[40]:


from sklearn.metrics import multilabel_confusion_matrix, classification_report
import numpy as np

# Predict the class labels for test_images
predictions = loaded_model.predict(test_images)

predicted_labels = (predictions > 0.5).astype(int)

# Compute the multi-label confusion matrix
mcm = multilabel_confusion_matrix(test_labels_, predicted_labels)

for i, matrix in enumerate(mcm):
    print(f"Confusion matrix for label {i}:")
    print(matrix)

# classification report
report = classification_report(test_labels_, predicted_labels, target_names=target_list)
print(report)


# In[47]:


# Create a 2x5 grid layout for plotting
num_rows = 3
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Plot the confusion matrices in the grid with numbers
for i, (matrix, label) in enumerate(zip(mcm, target_list)):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]

    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix for {label}')
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Annotate the matrix values
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(matrix[r, c]), va='center', ha='center', color='white', fontsize=12)


# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[41]:


# Model Predict Function
np.random.seed(123)
def model_predict(model, test_img):

  pred_test = model.predict(test_img)
  pred_test = pred_test.round()
  return pred_test


# In[42]:


# Pred vs True Function
def pred_vs_true(i, model):

  print("Image")
  print('-----------------')
  x = array_to_img(test_images[i])
  newsize = (112,112)
  x = x.resize(newsize)
  display(x)
  print('-----------------')
  pred_test = model_predict(model, test_images)
  df = pd.DataFrame({'Model Prediction': pred_test[i], 'True Labels':test_labels_[i]},
                index = target_list)
  return df


# In[ ]:


pred_vs_true(30,loaded_model)


# In[43]:


pred_vs_true(72,loaded_model)


# In[48]:


pred_vs_true(120,loaded_model)

