import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import keras

# Load the model outside Streamlit for performance
try:
  MODEL = tf.keras.models.load_model("/home/ansh/Jupyter/potato-disease/potato disease.keras")
  class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
except Exception as e:
  st.error(f"An error occurred loading the model: {e}")
  st.stop()

def preprocess_image(img, model_input_shape=(224, 224, 3)):
  """
  Preprocesses an image for model prediction.

  Args:
      img: PIL image object
      model_input_shape: Expected input shape of the model (channels last)

  Returns:
      A preprocessed NumPy array
  """
  img = img.resize(model_input_shape[:2])
  img_array = np.array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

def predict(model, image):
  img_array = preprocess_image(image, model.input_shape[1:])  # Use model input shape
  predictions = model.predict(img_array)

  predicted_class = class_names[np.argmax(predictions[0])]
  confidence = round(100 * (np.max(predictions[0])), 2)
  return predicted_class, confidence

# Streamlit app
st.title("Potato Disease Prediction")

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Analyze'):
        predicted_class, confidence = predict(MODEL, image)
        try:
            if predicted_class:
                st.write("Predicted Class:", predicted_class)
                st.write("Confidence:", f"{confidence:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
