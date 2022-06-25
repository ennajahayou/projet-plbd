import streamlit as st
import pandas as pd 
import datetime

st.set_page_config(layout='wide')
def app10():
 st.title('projet learning by doing group 15')
 today = datetime.date.today()
 df=pd.read_csv('base_od_donne.csv')
 st.write(df)
 start_date = st.date_input('Start date', today)
 st.write('la valeur minimal de ET est',(min(df['ET'][df['Year']==str(start_date)])))
 st.write('l heure ou on va irriger est',df['hour1'][df['ET']==(min(df['ET'][df['Year']==str(start_date)]))].to_list()[0])
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import joblib
import tensorflow as tf

from keras.models import Sequential
@st.cache(allow_output_mutation=True)
def load_model():
   model1=tf.keras.models.load_model('model leaf deasis 97.11.h5')  
   return model1
with st.spinner('Model is being loaded..'):
  model1=load_model()

def app20():
 

 st.write("""
         #  Classification du maladie des plantes 
         """
         )
 
 file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
 import cv2
 from PIL import Image, ImageOps
 import numpy as np
 st.set_option('deprecation.showfileUploaderEncoding', False)
 def import_and_predict(image_data, model):
    
        dataset20=[]
        size = (256,256)    
        image = Image.fromarray(image_data)
        image = image.resize(size)
        dataset20.append(np.array(image)) 
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        #img_reshape = img[np.newaxis,...] 
        dataset20=np.stack(dataset20,axis=0)
        dataset20=dataset20/255.
        prediction = model.predict(dataset20)
        
        return prediction
 if file is None:
    st.text("Please upload an image file")
 else:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.image(Image.open(file), use_column_width=True)
    predictions = import_and_predict(opencv_image, model1)
   # score = predictions[0]
    score = tf.nn.softmax(predictions)
    st.write(predictions)
    class_names=['Early_Blight','Early_Blight','Healthy']
    print(
    predictions
    )
    st.write("predicted label:",class_names[np.argmax(predictions)])
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import joblib
import tensorflow as tf

from keras.models import Sequential
def load_model():
  model = joblib.load("mchin.joblib")  
  return model   
with st.spinner('Model is being loaded..'):
  model=load_model()
def app30():
 

 st.write("""
         #  Classificationzone irriguer 
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

st.header('groupe plbd 15')
page=st.sidebar.selectbox('choisir une page',["base de donnée","maladie des plantes",'zone irriguer'])
if page=="base de donnée":
    app10()
if page=="maladie des plantes":
    app20()
if page=='zone irriguer':
    app30()

