
import streamlit as st  #Web App
import torch
import numpy as np #Image Processing 
import cv2
import imutils
from PIL import Image #Image Processing
from gtts import gTTS
import IPython.display as ipd
import matplotlib.pyplot as plt
import easyocr as ocr  #OCR
# %matplotlib inline

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#title
st.title("Bus route number detector")

#subtitle
st.markdown("## Optical Character Recognition - \n ## Using `YOLO v5`,`easyocr`, `streamlit`")

st.markdown("")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

@st.cache
def load_detector_model(): 
    model = torch.hub.load('yolov5', 'custom', path='./models/best.pt', source='local') 
    return model 

@st.cache
def load_OCR_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='./models')
    return reader 

model = load_detector_model() #load ML model

reader = load_OCR_model() #load OCR model

if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image
    input_image = np.array(input_image)

    with st.spinner("🤖 AI is at Work! "):
        
        resultOfML = model(input_image)

        bounding_box = resultOfML.pandas().xyxy[0]  # img predictions (pandas)

        # xmin
        x_min = int(bounding_box['xmin'][1])
        # xmax
        x_max = int(bounding_box['xmax'][1])
        # ymin
        y_min = int(bounding_box['ymin'][1])
        # ymax
        y_max = int(bounding_box['ymax'][1])

        # use numpy slicing to crop the region of interest
        roi = input_image[y_min:y_max,x_min:x_max]

        inverted_image = cv2.bitwise_not(roi)
        cv2.imwrite("./temp/inverted.jpg", inverted_image)

        inocr_img = cv2.imread('./temp/inverted.jpg')

        result = reader.readtext(np.array(inocr_img))

        if(len(result) == 0):
            result_text = "Route number not detected"
        else:
            result_text = result[0][1]

        st.write(result_text)
        #st.success("Here you go!")
        st.balloons()

        audio_text = result_text
        language = 'en'
        audio_obj = gTTS(text=audio_text, lang=language, slow=False) 
        audio_obj.save("./out-audio/bus_number.mp4") 

        st.audio('./out-audio/bus_number.mp4', format="audio/mp4", start_time=0)

else:
    st.write("Upload an Image")

st.caption("Made with ❤️ by @PW_26")





