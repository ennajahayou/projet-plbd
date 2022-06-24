import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import joblib
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential

from tensorflow import keras
from keras import layers,models
from keras.models import Sequential
from keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
@st.cache(allow_output_mutation=True)
def load_model():
  model = joblib.load("mchin.joblib")
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
def app():
 

 st.write("""
         #  Classification du maladie des plantes 
         """
         )

 file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png","jpeg"])
 import cv2
 from PIL import Image, ImageOps
 import numpy as np
 st.set_option('deprecation.showfileUploaderEncoding', False)




 from PIL import Image
 def import_and_predict(image_data, model):    

        dataset20=[]
        SIZE = 256    
        image = image_data.resize((SIZE, SIZE))
        dataset20.append(np.array(image))
        dataset20=np.array(dataset20)
        dataset20=dataset20.reshape(dataset20.shape[0],-1)  
        prediction = model.predict(dataset20)
        
        return prediction
 if file is None:
    st.text("Please upload an image file")
 else:
    image = Image.open(file)
    st.image(Image.open(file), use_column_width=True)
    predictions = import_and_predict(image, model)
   # score = predictions[0]
   # st.write(predictions)
  #  st.write(score)
    class_names=['irriguer', 'non irriguer']
    #print(
    #"This image most likely belongs to {} with a {:.2f} percent confidence."
   # .format(class_names[np.argmax(score)], 100 * np.max(score))
    #)
    st.write("predicted label:",class_names[predictions[0]])