#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#connect to google drive
#from google.colab import drive
#drive.mount('/content/drive')


# # Read train and test data
# 

# In[1]:


train_path = 'train'
test_path = 'test'


# In[2]:


import os


# In[3]:


#check number of images in each file
train_num = os.listdir(train_path)
test_num = os.listdir(test_path)
train_len= len(train_num)
test_len= len(test_num)

print(f"Number of files in the train folder: {train_len}")
print(f"Number of files in the train folder: {test_len}")


# In[4]:


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
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout, BatchNormalization
from tensorflow.keras import callbacks, models, layers, optimizers, regularizers

import warnings
warnings.filterwarnings("ignore")


# ### Read Train data

# In[5]:


# load label of images
df_train = pd.read_json('bdd100k/train30.json')
df_train.head()


# In[6]:


df_train.shape


# #### one hot encoding label

# In[7]:


#created function to extract 'category' from labels' column in each row 
def clean_labels(example): 
  example_df = pd.DataFrame.from_records(example)
  example_df = example_df['category'].unique().tolist()
    
  return ','.join(example_df)


# In[8]:


df_train['clean_labels'] = df_train['labels'].map(clean_labels)
df_train


# In[9]:


#create target_list, convert to set, convert to list 
target_list = ",".join(df_train.clean_labels).split(",")
target_list = list(set(target_list))
target_list


# In[10]:


#create copy of df, for loop to search if clean_labels contains target_list, convert to int 
data_train = df_train.copy()
for target in target_list:
    data_train[target] = data_train['clean_labels'].str.contains(target)
    data_train[target] = data_train[target].astype(int)
 


# In[11]:


#drop attributes and timestamp columns, not neccessary for our model 
columns = ['clean_labels','attributes', 'timestamp','labels']
data_train = data_train.drop(columns = columns)

#review columns have been dropped 
data_train 


# In[13]:


#retreiving labels 
labels = list(data_train.columns.values)
labels = labels[1:]
print(labels)

#creating dataframe
counts = []
for label in labels:
    counts.append((label, data_train[label].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'Occurrence'])
df_stats = df_stats.sort_values(['Occurrence']).reset_index(drop=True)
df_stats


# In[14]:


#retreiving sum of label counts
rowSums = data_train.iloc[:,1:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[1:]
label_count = pd.DataFrame(multiLabel_counts, columns = ['Total # of images']).rename_axis('# of Labels', axis=1)
label_count


# In[15]:


#remove 'train', since there's too little data with those label
columns = ['train']
data_train = data_train.drop(columns = columns)

data_train


# ### Test dataset

# In[16]:


df_test = pd.read_json('bdd100k/test10.json')
df_test.head()


# In[17]:


df_test.shape


# In[18]:


df_test['clean_labels'] = df_test['labels'].map(clean_labels)
df_test


# In[19]:


#create target_list, convert to set, convert to list 
target_list = ",".join(df_test.clean_labels).split(",")
target_list = list(set(target_list))
target_list


# In[20]:


#create copy of df_val, for loop to search if clean_labels contains target_list, convert to int
data_test = df_test.copy()
for target in target_list:
    data_test[target] = data_test['clean_labels'].str.contains(target)
    data_test[target] = data_test[target].astype(int)


# In[21]:


#drop attributes and timestamp columns, not neccessary for our model 
columns = ['clean_labels','attributes', 'timestamp','labels']
data_test = data_test.drop(columns = columns)

data_test 


# In[22]:


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


# In[23]:


#retreiving sum of label counts
rowSums = data_test.iloc[:,3:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[1:]
label_count = pd.DataFrame(multiLabel_counts, columns = ['Total # of images']).rename_axis('# of Labels', axis=1)
label_count


# In[24]:


#remove 'train', since there's too little data with those label
columns = ['train']
data_test = data_test.drop(columns = columns)

data_test


# In[25]:


#update target list
target_list= [ 'motor','drivable area','lane','bike','truck','traffic sign','person','bus','car','rider','traffic light']


# ### ImageDataGenerator

# In[26]:


IMG_SIZE= (224,224)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Random rotation between -30 and 30 degrees
    width_shift_range=0.2,  # xRandomly shift the width by 20%
    height_shift_range=0.2, # Randomly shift the height by 20%
    shear_range=0.2,        # Randomly apply shear transformation
    zoom_range=0.2,         # Randomly zoom by 20%
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill any newly created pixels after rotation or shift
)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[27]:


train_set = train_datagen.flow_from_dataframe(data_train, directory=train_path, 
                                              x_col ='name', y_col = target_list, 
                                              class_mode = "multi_output", seed= 24,
                                                batch_size=3000, target_size = IMG_SIZE)
test_set = test_datagen.flow_from_dataframe(data_test, directory=test_path, 
                                          x_col ='name', y_col = target_list, 
                                          class_mode = "multi_output", seed= 24,
                                          batch_size=1000, target_size = IMG_SIZE)


# In[28]:


#create datasets for train, test
train_images, train_labels = next(train_set)
test_images, test_labels = next(test_set)


# In[29]:


#reshape labels 
train_labels_ = np.array(train_labels).reshape(len(train_labels), -1).T
test_labels_ = np.array(test_labels).reshape(len(test_labels), -1).T


# In[30]:


#reshape images 
train_img = train_images.reshape(train_images.shape[0], -1)
test_img = test_images.reshape(test_images.shape[0], -1)


# In[31]:


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


# ### Check for images

# In[32]:


#observing images are viewable
array_to_img(train_images[0])


# In[33]:


#observing labels match the image
pd.Series(train_labels_[0], index=target_list)


# # Multiclass classification CNN model
# ## Model1: Base Model

# In[34]:


num_classes=11


# In[35]:


# Create a Sequential model
model_cnn = Sequential()

model_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224,224, 3)))
model_cnn.add(MaxPooling2D((2, 2)))

model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D((2, 2)))
model_cnn.add(Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D((2, 2)))

# Flatten the output and add fully connected layers
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5))  # Add dropout for regularization
model_cnn.add(Dense(num_classes, activation='softmax'))

# Compile the model
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[36]:


model_cnn.summary()


# In[37]:


# Train the model
history_cnn = model_cnn.fit(train_images, train_labels_, epochs=10, batch_size=64, validation_data=(test_images, test_labels_), verbose=1)


# In[38]:


# Make predictions
predictions = model_cnn.predict(test_images)  


# Model evaluation

# In[39]:


def plot_loss_and_accuracy(history):
    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Plot loss during training
    axs[1].plot(history.history['loss'], label='train')
    axs[1].plot(history.history['val_loss'], label='test')
    axs[1].set_title('Loss')
    axs[1].legend()

    # Plot accuracy during training
    axs[0].plot(history.history['accuracy'], label='train')
    axs[0].plot(history.history['val_accuracy'], label='test')
    axs[0].set_title('Accuracy')
    axs[0].legend()

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # Display the plots
    plt.show()


# In[40]:


plot_loss_and_accuracy(history_cnn)


# In[78]:


from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Predict the class labels for test_images
predictions = model_cnn.predict(test_images)

predicted_labels = (predictions > 0.5).astype(int)

# Compute the multi-label confusion matrix
mcm = multilabel_confusion_matrix(test_labels_, predicted_labels)

for i, matrix in enumerate(mcm):
    print(f"Confusion matrix for label {i}:")
    print(matrix)

# classification report
report = classification_report(test_labels_, predicted_labels, target_names=target_list)
print(report)


# In[122]:


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


# ## Model2: Change hidden layers

# In[42]:


#instantiate model
model_2 = Sequential()

#input layer
model_2.add(layers.Conv2D(32,(3, 3), activation='relu',padding='same', input_shape=(224,224,3)))
model_2.add(layers.MaxPooling2D((2, 2)))

#hidden layer
#first hidden layer
model_2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_2.add(layers.MaxPooling2D((2, 2)))

#second hidden layer
model_2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_2.add(layers.MaxPooling2D((2, 2)))

#output layer  
model_2.add(Flatten())
model_2.add(Dense(128, activation='relu'))
model_2.add(Dropout(0.5))  # Add dropout for regularization
model_2.add(Dense(num_classes, activation='softmax'))

#compile model 
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[43]:


#summary
model_2.summary()


# In[44]:


# Train the model
history_cnn2 = model_2.fit(train_images, train_labels_, epochs=10, batch_size=64, validation_data=(test_images, test_labels_), verbose=1)


# In[45]:


plot_loss_and_accuracy(history_cnn2)


# ### Model3: Add BatchNormalization layer

# In[41]:


#instantiate model
model_3 = Sequential()

#input layer
model_3.add(layers.Conv2D(32,(3, 3), activation='relu',padding='same', input_shape=(224,224,3)))
model_3.add(layers.MaxPooling2D((2, 2)))

#hidden layer
model_3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_3.add(BatchNormalization())
model_3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_3.add(layers.MaxPooling2D((2, 2)))

#output layer  
model_3.add(Flatten())
model_3.add(Dense(128, activation='relu'))
model_3.add(Dropout(0.2))  # Add dropout for regularization
model_3.add(Dense(num_classes, activation='softmax'))

#compile model 
model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[46]:


history_cnn3 = model_3.fit(train_images, train_labels_, epochs=10, batch_size=64, validation_data=(test_images, test_labels_), verbose=1)


# In[47]:


plot_loss_and_accuracy(history_cnn3)


# In[123]:


# Predict the class labels for test_images
predictions = model_cnn.predict(test_images)

predicted_labels = (predictions > 0.5).astype(int)

# Compute the multi-label confusion matrix
mcm = multilabel_confusion_matrix(test_labels_, predicted_labels)

for i, matrix in enumerate(mcm):
    print(f"Confusion matrix for label {i}:")
    print(matrix)

# classification report
report = classification_report(test_labels_, predicted_labels, target_names=target_list)
print(report)


# In[124]:


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


# ### Model4: Add L1/L2 layer

# In[48]:


from keras.regularizers import l1_l2


# In[49]:


#instantiate model
model_4 = Sequential()

#input layer
model_4.add(layers.Conv2D(32,(3, 3), activation='relu',padding='same', input_shape=(224,224,3)))
model_4.add(layers.MaxPooling2D((2, 2)))

#hidden layer
model_4.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_4.add(BatchNormalization())
model_4.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_4.add(layers.MaxPooling2D((2, 2)))

#output layer  
model_4.add(Flatten())
model_4.add(Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
model_4.add(Dropout(0.2))  # Add dropout for regularization
model_4.add(Dense(num_classes, activation='softmax'))

#compile model 
model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[125]:


model_4.summary()


# In[50]:


history_cnn4 = model_4.fit(train_images, train_labels_, epochs=10, batch_size=64, validation_data=(test_images, test_labels_), verbose=1)


# In[51]:


plot_loss_and_accuracy(history_cnn4)


# ### Model Prediction

# In[69]:


def model_predict(model, test_img):
    
    pred_test = model.predict(test_img)
    #pred_test = pred_test.round()
    return pred_test


# In[70]:


def model_prediction(i, model):

  print("Image")
  print('-----------------')
  x = array_to_img(test_images[i])
  newsize = (224,224)
  x = x.resize(newsize)
  display(x)
  print('-----------------')
  pred_test = model_predict(model, test_images)
  df = pd.DataFrame({'Model Prediction': pred_test[i], 'True Labels':test_labels_[i]},
                index = target_list)
  return df 


# In[71]:


#image1 = 'Documents/UChicago/2023_Summer/Deep_Learning/project/bdd100k/images/100k/test/cabc9045-cd422b81.jpg'
model_prediction(79, model_3)


# ### Save the best performance model

# In[79]:


model_3.save("cnn_model.h5")


# In[99]:


import keras.utils as image

path = '/Users/jasmine19970120/Documents/UChicago/2023_Summer/Deep_Learning/project/bdd100k/images/100k/test/cabc9045-cd422b81.jpg'
img = image.load_img(path, target_size=(224,224))
image_array = np.array(img) / 255.0

#image_array = image.img_to_array(img)
predictions = model_3.predict(np.expand_dims(image_array, axis=0))
predictions


# ### Model5: add more hidden layer

# In[102]:


#instantiate model
model_5 = Sequential()

#input layer
model_5.add(layers.Conv2D(32,(3, 3), activation='relu',padding='same', input_shape=(224,224,3)))
model_5.add(layers.MaxPooling2D((2, 2)))

#hidden layer
model_5.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_5.add(BatchNormalization())
model_5.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_5.add(layers.MaxPooling2D((2, 2)))

model_5.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_5.add(BatchNormalization())
model_5.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_5.add(layers.MaxPooling2D((2, 2)))

#output layer  
model_5.add(Flatten())
model_5.add(Dense(128, activation='relu'))
model_5.add(Dropout(0.2))  # Add dropout for regularization
model_5.add(Dense(num_classes, activation='softmax'))

#compile model 
model_5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[103]:


history_cnn5 = model_5.fit(train_images, train_labels_, epochs=10, batch_size=64, validation_data=(test_images, test_labels_), verbose=1)


# In[104]:


plot_loss_and_accuracy(history_cnn5)


# In[ ]:




