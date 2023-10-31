

import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
st.title("Dog breed Image Classification")

st.write("Predict the dog breed that is being represented in the image.")

model = load_model("model.h5")
labels = ['Afghan_hound',
          'African_hunting_dog',
          'Airedale',
          'American_Staffordshire_terrier',
          'Appenzeller',
          'Australian_terrier',
          'Bedlington_terrier',
          'Bernese_mountain_dog',
          'Blenheim_spaniel',
          'Beagle',
          'Border_terrier',
          'Boston_bull',
          'Bouvier_des_Flandres',
          'Brabancon_griffon',
          'Brittany_spaniel',
          'Cardigan',
          'Chesapeake_Bay_retriever',
          'Chihuahua',
          'Dandie_Dinmont',
          'Doberman',
          'English_foxhound',
          'English_setter',
          'English_springer',
          'EntleBucher',
          'Eskimo_dog',
          'French_bulldog',
          'German_shepherd',
          'German_short',
          'Gordon_setter',
          'Great_Dane',
          'Great_Pyrenees',
          'Greater_Swiss_Mountain_dog',
          'Ibizan_hound',
          'Irish_setter',
          'Irish_terrier',
          'Irish_water_spaniel',
          'Irish_wolfhound',
          'Italian_greyhound',
          'Japanese_spaniel',
          'Kerry_blue_terrier',
          'Labrador_retriever',
          'Lakeland_terrier',
          'Leonberg',
          'Lhasa',
          'Maltese_dog',
          'Mexican_hairless',
          'Newfoundland',
          'Norfolk_terrier',
          'Norwegian_elkhound',
          'Norwich_terrier',
          'Old_English_sheepdog',
          'Pekinese',
          'Pembroke',
          'Pomeranian',
          'Rhodesian_ridgeback',
          'Rottweiler',
          'Saint_Bernard',
          'Saluki',
          'Samoyed',
          'Scotch_terrier',
          'Scottish_deerhound',
          'Sealyham_terrier',
          'Shetland_sheepdog',
          'Shih',
          'Siberian_husky',
          'Staffordshire_bullterrier',
          'Sussex_spaniel',
          'Tibetan_mastiff',
          'Tibetan_terrier',
          'Walker_hound',
          'Weimaraner',
          'Welsh_springer_spaniel',
          'West_Highland_white_terrier',
          'Yorkshire_terrier',
          'affenpinscher',
          'basenji',
          'basset',
          'beagle',
          'black',
          'bloodhound',
          'bluetick',
          'borzoi',
          'boxer',
          'briard',
          'bull_mastiff',
          'cairn',
          'chow',
          'clumber',
          'cocker_spaniel',
          'collie',
          'curly',
          'dhole',
          'dingo',
          'flat',
          'giant_schnauzer',
          'golden_retriever',
          'groenendael',
          'keeshond',
          'kelpie',
          'komondor',
          'kuvasz',
          'malamute',
          'malinois',
          'miniature_pinscher',
          'miniature_poodle',
          'miniature_schnauzer',
          'otterhound',
          'papillon',
          'pug',
          'redbone',
          'schipperke',
          'silky_terrier',
          'soft',
          'standard_poodle',
          'standard_schnauzer',
          'toy_poodle',
          'toy_terrier',
          'vizsla',
          'whippet',
          'wire']


uploaded_file = st.file_uploader(
    "Upload an image of a Dog:", type='jpg'
)
predictions = -1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1 = image.smart_resize(image1, (224, 224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label = labels[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("test_cricket.jpg")
    image1 = image.smart_resize(image1, (224, 224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label = labels[np.argmax(predictions)]
    image1 = Image.open("test_cricket.jpg")
    st.image(image1, caption="Uploaded Image", use_column_width=True)
    st.markdown(
        f"<h2 style='text-align: center;'>{label}</h2>",
        unsafe_allow_html=True,
    )
