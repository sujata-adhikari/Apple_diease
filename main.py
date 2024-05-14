import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.set_page_config(
    page_title="Apple Disease Recognition System",
    layout="wide",
    initial_sidebar_state="expanded",
)


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("APPLE DISEASE RECOGNITION SYSTEM")
    image_path = "applemain.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""

    Welcome to the Apple Disease Recognition System!

    ## About

    Our system is dedicated to helping apple growers identify and manage common diseases affecting their crops. With advanced image recognition technology, we provide accurate diagnosis and effective treatment recommendations for the following diseases:

    ### Apple Scab


    **Symptoms:**
    - Dark, olive-green, or black lesions on leaves, fruit, and twigs.
    - Velvety texture on the underside of leaves.
    - Premature leaf drop.

    **Treatment:**
    - Apply fungicides containing active ingredients like myclobutanil or sulfur.
    - Prune infected branches and remove fallen leaves to reduce overwintering spores.
    - Plant resistant apple varieties.

    ### Black Rot

    **Symptoms:**
    - Circular, sunken lesions on fruit with concentric rings.
    - Brown, shriveled lesions on leaves.
    - Twig cankers and leaf drop.

    **Treatment:**
    - Apply fungicides containing active ingredients like captan.
    - Remove infected fruit and prune affected branches during dormant season.
    - Promote good air circulation and sanitation practices.

    ### Cedar Apple Rust


    **Symptoms:**
    - Yellow-orange spots or lesions on leaves, which enlarge and form yellow-orange spore horns.
    - Galls on twigs.
    - Premature leaf drop.

    **Treatment:**
    - Apply fungicides containing active ingredients like myclobutanil or mancozeb.
    - Remove nearby cedar or juniper trees, which serve as alternate hosts.
    - Plant resistant apple varieties.

    ### Healthy Apple Trees


    **Symptoms:**
    - Vigorous growth with no visible signs of disease.
    - Lush green leaves and abundant fruit production.

    **Preventive Measures:**
    - Implement good orchard management practices, including proper irrigation and fertilization.
    - Monitor for early signs of disease and promptly treat if detected.
    - Practice crop rotation and maintain overall tree health.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of images of healthy and diseased apple leaves which are categorized into 4 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                
                #### Content
                1. train - Used to train the model
                2. test - Used to test the model
                3. validation - used for validation of the model

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

