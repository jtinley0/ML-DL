import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained models
loaded_models = {
    'ResNet50 Model': load_model('drop_resnet_model.h5'),
    'Full CNN Model': load_model('cnn_model.h5')
}

# Preprocessing the images
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert('RGB')
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predicting using the selected model
def predict_image(model, image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)[0]
    return predictions

st.title('Self-Driving Car: Multi-label Classification')

# Set Seaborn style and font sizes
sns.set(style="whitegrid", font_scale=1.2)

# Dropdown menu to select model
selected_model = st.selectbox('Select a model', list(loaded_models.keys()))

# Uploading image through Streamlit
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Displaying the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    selected_loaded_model = loaded_models[selected_model]
    class_probs = predict_image(selected_loaded_model, uploaded_image)
    
    # Defining class names
    class_names = ['motor', 'drivable area', 'lane', 'bike', 'truck', 'traffic sign', 'person', 'bus', 'car', 'rider', 'traffic light']
    
    # Format probabilities with 3 decimal places
    formatted_probs = [format(prob, '.3f') for prob in class_probs]
    
    # Creating a bar plot of probabilities with different colors
    plt.figure(figsize=(10, 6))
    color_palette = sns.color_palette("pastel")
    sns.barplot(x=class_probs, y=class_names, palette=color_palette)
    plt.xlabel('Probability')
    plt.ylabel('Class')
    plt.title('Class Probabilities')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    st.pyplot(plt)
    
    # Displaying class probabilities in a table
    class_prob_df = pd.DataFrame(zip(class_names, formatted_probs), columns=['Class', 'Probability'])
    st.table(class_prob_df)
