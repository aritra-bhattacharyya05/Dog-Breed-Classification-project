import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow_hub import KerasLayer

st.set_page_config(
    page_title="🐶 Dog Breed Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

BREEDS = [
    "affenpinscher","afghan_hound","african_hunting_dog","airedale",
    "american_staffordshire_terrier","appenzeller","australian_terrier",
    "basenji","basset","beagle","bedlington_terrier","bernese_mountain_dog",
    "black-and-tan_coonhound","blenheim_spaniel","bloodhound","bluetick",
    "border_collie","border_terrier","borzoi","boston_bull",
    "bouvier_des_flandres","boxer","brabancon_griffon","briard",
    "brittany_spaniel","bull_mastiff","cairn","cardigan",
    "chesapeake_bay_retriever","chihuahua","chow","clumber",
    "cocker_spaniel","collie","curly-coated_retriever","dandie_dinmont",
    "dhole","dingo","doberman","english_foxhound","english_setter",
    "english_springer","entlebucher","eskimo_dog","flat-coated_retriever",
    "french_bulldog","german_shepherd","german_short-haired_pointer",
    "giant_schnauzer","golden_retriever","gordon_setter","great_dane",
    "great_pyrenees","greater_swiss_mountain_dog","groenendael",
    "ibizan_hound","irish_setter","irish_terrier","irish_water_spaniel",
    "irish_wolfhound","italian_greyhound","japanese_spaniel","keeshond",
    "kelpie","kerry_blue_terrier","komondor","kuvasz",
    "labrador_retriever","lakeland_terrier","leonberg","lhasa",
    "malamute","malinois","maltese_dog","mexican_hairless",
    "miniature_pinscher","miniature_poodle","miniature_schnauzer",
    "newfoundland","norfolk_terrier","norwegian_elkhound",
    "norwich_terrier","old_english_sheepdog","otterhound","papillon",
    "pekinese","pembroke","pomeranian","pug","redbone",
    "rhodesian_ridgeback","rottweiler","saint_bernard","saluki",
    "samoyed","schipperke","scotch_terrier","scottish_deerhound",
    "sealyham_terrier","shetland_sheepdog","shih-tzu",
    "siberian_husky","silky_terrier","soft-coated_wheaten_terrier",
    "staffordshire_bullterrier","standard_poodle",
    "standard_schnauzer","sussex_spaniel","tibetan_mastiff",
    "tibetan_terrier","toy_poodle","toy_terrier","vizsla",
    "walker_hound","weimaraner","welsh_springer_spaniel",
    "west_highland_white_terrier","whippet",
    "wire-haired_fox_terrier","yorkshire_terrier",
]


@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(
            os.path.dirname(__file__),
            '20260502-090800-full-image-set-mobilenetv2-Adam-3.h5'
        )

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': KerasLayer}
        )

        return model

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.write("📂 Files in directory:", os.listdir(os.path.dirname(__file__)))
        return None

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.title("🐶 Dog Breed Classifier")
st.markdown("Upload a dog photo and I'll identify the breed!")
st.divider()

model = load_model()

if model is None:
    st.stop()

uploaded_file = st.file_uploader(
    "📸 Upload a dog image",
    type=["jpg", "jpeg", "png", "gif", "webp"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Result")

        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img, verbose=0)

        confidence_scores = predictions[0]
        top_idx = np.argmax(confidence_scores)
        top_breed = BREEDS[top_idx].replace('_', ' ').title()
        top_confidence = float(confidence_scores[top_idx]) * 100

        st.success("✅ Prediction Complete!")
        st.markdown(f"## {top_breed}")
        st.markdown(f"**Confidence: {top_confidence:.1f}%**")
        st.progress(top_confidence / 100.0)

    st.divider()
    st.subheader("🏆 Top 5 Predictions")

    top_5_idx = np.argsort(confidence_scores)[-5:][::-1]

    predictions_list = []
    for rank, idx in enumerate(top_5_idx, 1):
        predictions_list.append({
            "Rank": rank,
            "Breed": BREEDS[idx].replace('_', ' ').title(),
            "Confidence": f"{float(confidence_scores[idx])*100:.1f}%"
        })

    df = pd.DataFrame(predictions_list)
    st.table(df)

else:
    st.info("Upload an image to get started.")

st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;'>MobileNetV2 Transfer Learning Model</div>",
    unsafe_allow_html=True
)
