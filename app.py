# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load the trained model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("wasteclassifier.h5")

# model = load_model()

# # Define class labels
# class_labels = ["Organic", "Recyclable"]

# # Streamlit UI
# st.title("Waste Classification App ♻️")
# st.write("Upload an image to classify it as **Organic** or **Recyclable**.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image
#     img = image.resize((224, 224))  # Adjust size based on your model
#     img_array = np.array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Make prediction
#     prediction = model.predict(img_array)
#     predicted_class = class_labels[np.argmax(prediction)]

#     # Show result
#     st.write(f"### Prediction: **{predicted_class}**")

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import gdown
# import os
# from PIL import Image

# # Google Drive File ID of your model
# FILE_ID = "1SAH0sj2ww9baG_S2eitQP_8dSl4t-YTz"
# MODEL_PATH = "model.h5"
# IMG_SIZE = (224, 224)  # Adjust as per your model's input size

# # Function to download model from Google Drive
# def download_model():
#     if not os.path.exists(MODEL_PATH):
#         url = f"https://drive.google.com/uc?id={FILE_ID}"
#         gdown.download(url, MODEL_PATH, quiet=False)

# # Load the model
# @st.cache_resource
# def load_model():
#     download_model()
#     return tf.keras.models.load_model(MODEL_PATH)

# model = load_model()

# # Streamlit UI
# st.title("Waste Classification App ♻️")
# st.write("Upload an image and let the AI predict it as **Organic** or **Recyclable**.")

# # File uploader for image input
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     # Display uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess image
#     image = image.resize(IMG_SIZE)
#     image = np.array(image) / 255.0  # Normalize if required
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Predict
#     if st.button("Predict"):
#         prediction = model.predict(image)
#         st.success(f"Prediction: {prediction[0]}")  # Adjust as per your model output

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import gdown
# import os
# from PIL import Image

# # Google Drive File ID of your model
# FILE_ID = "1SAH0sj2ww9baG_S2eitQP_8dSl4t-YTz"
# MODEL_PATH = "model.h5"
# IMG_SIZE = (224, 224)  # Adjust as per your model's input size

# # Define class labels (update with your model's classes)
# CLASS_NAMES = ["Class A", "Class B", "Class C"]  # Modify as needed

# # Function to download model from Google Drive
# def download_model():
#     if not os.path.exists(MODEL_PATH):
#         url = f"https://drive.google.com/uc?id={FILE_ID}"
#         gdown.download(url, MODEL_PATH, quiet=False)

# # Load the model
# @st.cache_resource
# def load_model():
#     download_model()
#     return tf.keras.models.load_model(MODEL_PATH)

# model = load_model()

# # Streamlit UI
# st.title("Image Classification App 📷🚀")
# st.write("Upload an image and let the AI predict!")

# # File uploader for image input
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     # Display uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess image
#     image = image.resize(IMG_SIZE)
#     image = np.array(image) / 255.0  # Normalize if required
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Predict
#     if st.button("Predict"):
#         prediction = model.predict(image)
        
#         # Check prediction output shape
#         st.write(f"Raw Prediction Output: {prediction}")

#         if prediction.shape[-1] == len(CLASS_NAMES):  # Classification case
#             predicted_label = CLASS_NAMES[np.argmax(prediction)]
#             confidence = np.max(prediction) * 100
#             st.success(f"Predicted: {predicted_label} ({confidence:.2f}% confidence)")
#         else:
#             st.success(f"Prediction: {prediction[0]}")  # For regression models

import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive File ID of your model
FILE_ID = "1SAH0sj2ww9baG_S2eitQP_8dSl4t-YTz"
MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)  # Adjust to your model's input size

# Define class labels (Binary Classification)
CLASS_NAMES = ["Organic", "Recyclable"]  # Modify if needed

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Streamlit UI
st.title("Waste Classification App ♻️")
st.write("Upload an image to classify it as **Organic** or **Recyclable**.")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Normalize if required
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    if st.button("Predict"):
        prediction = model.predict(image)
        predicted_label = CLASS_NAMES[np.argmax(prediction)]
        st.success(f"Prediction: **{predicted_label}** ✅")
