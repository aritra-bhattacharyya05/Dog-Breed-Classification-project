import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import streamlit as st
    import numpy as np
    from PIL import Image
    import pandas as pd
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    import streamlit as st
    st.error("⚠️ TensorFlow not available locally")
    st.info("This app will work perfectly when deployed to Streamlit Cloud. For local testing, please install Visual C++ Redistributable: https://support.microsoft.com/en-us/help/2977003/")
    st.stop()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🐶 Dog Breed Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# BREED NAMES (All 120 breeds)
# ============================================================
BREEDS = [
    "affenpinscher",
    "afghan_hound",
    "african_hunting_dog",
    "airedale",
    "american_staffordshire_terrier",
    "appenzeller",
    "australian_terrier",
    "basenji",
    "basset",
    "beagle",
    "bedlington_terrier",
    "bernese_mountain_dog",
    "black-and-tan_coonhound",
    "blenheim_spaniel",
    "bloodhound",
    "bluetick",
    "border_collie",
    "border_terrier",
    "borzoi",
    "boston_bull",
    "bouvier_des_flandres",
    "boxer",
    "brabancon_griffon",
    "briard",
    "brittany_spaniel",
    "bull_mastiff",
    "cairn",
    "cardigan",
    "chesapeake_bay_retriever",
    "chihuahua",
    "chow",
    "clumber",
    "cocker_spaniel",
    "collie",
    "curly-coated_retriever",
    "dandie_dinmont",
    "dhole",
    "dingo",
    "doberman",
    "english_foxhound",
    "english_setter",
    "english_springer",
    "entlebucher",
    "eskimo_dog",
    "flat-coated_retriever",
    "french_bulldog",
    "german_shepherd",
    "german_short-haired_pointer",
    "giant_schnauzer",
    "golden_retriever",
    "gordon_setter",
    "great_dane",
    "great_pyrenees",
    "greater_swiss_mountain_dog",
    "groenendael",
    "ibizan_hound",
    "irish_setter",
    "irish_terrier",
    "irish_water_spaniel",
    "irish_wolfhound",
    "italian_greyhound",
    "japanese_spaniel",
    "keeshond",
    "kelpie",
    "kerry_blue_terrier",
    "komondor",
    "kuvasz",
    "labrador_retriever",
    "lakeland_terrier",
    "leonberg",
    "lhasa",
    "malamute",
    "malinois",
    "maltese_dog",
    "mexican_hairless",
    "miniature_pinscher",
    "miniature_poodle",
    "miniature_schnauzer",
    "newfoundland",
    "norfolk_terrier",
    "norwegian_elkhound",
    "norwich_terrier",
    "old_english_sheepdog",
    "otterhound",
    "papillon",
    "pekinese",
    "pembroke",
    "pomeranian",
    "pug",
    "redbone",
    "rhodesian_ridgeback",
    "rottweiler",
    "saint_bernard",
    "saluki",
    "samoyed",
    "schipperke",
    "scotch_terrier",
    "scottish_deerhound",
    "sealyham_terrier",
    "shetland_sheepdog",
    "shih-tzu",
    "siberian_husky",
    "silky_terrier",
    "soft-coated_wheaten_terrier",
    "staffordshire_bullterrier",
    "standard_poodle",
    "standard_schnauzer",
    "sussex_spaniel",
    "tibetan_mastiff",
    "tibetan_terrier",
    "toy_poodle",
    "toy_terrier",
    "vizsla",
    "walker_hound",
    "weimaraner",
    "welsh_springer_spaniel",
    "west_highland_white_terrier",
    "whippet",
    "wire-haired_fox_terrier",
    "yorkshire_terrier",
]

# ============================================================
# LOAD MODEL (Cached for efficiency)
# ============================================================
@st.cache_resource
def load_model():
    """Load model once and cache it"""
    try:
        # Get the directory where app.py is located
        model_path = os.path.join(os.path.dirname(__file__), '20260502-090800-full-image-set-mobilenetv2-Adam-3.h5')
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.error("Make sure your .h5 file is in the same directory as app.py")
        st.error("Also make sure tensorflow-hub is installed: pip install tensorflow-hub")
        return None

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(image):
    """Convert PIL image to model-ready tensor"""
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ============================================================
# MAIN APP
# ============================================================

# Title
st.title("🐶 Dog Breed Classifier")
st.markdown("Upload a dog photo and I'll identify the breed!")
st.divider()

# Load model
model = load_model()

if model is None:
    st.stop()

# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📸 Upload a dog image",
    type=["jpg", "jpeg", "png", "gif", "webp"],
    help="JPG, PNG, GIF, or WebP format. Clear photos work best."
)

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Left column: Display image
    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
    
    # Right column: Display prediction
    with col2:
        st.subheader("Result")
        
        # Preprocess image
        processed_img = preprocess_image(image)
        
        # Make prediction
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.write("🔍 Analyzing image...")
        progress_bar.progress(33)
        
        predictions = model.predict(processed_img, verbose=0)
        
        progress_bar.progress(66)
        
        confidence_scores = predictions[0]
        
        # Get top prediction
        top_idx = np.argmax(confidence_scores)
        top_breed = BREEDS[top_idx].replace('_', ' ').title()
        top_confidence = float(confidence_scores[top_idx]) * 100
        
        progress_bar.progress(100)
        progress_text.empty()
        progress_bar.empty()
        
        # Display result
        st.success("✅ Prediction Complete!")
        st.markdown(f"## {top_breed}")
        st.markdown(f"**Confidence: {top_confidence:.1f}%**")
        st.progress(top_confidence / 100.0)
    
    # ============================================================
    # TOP 5 PREDICTIONS
    # ============================================================
    
    st.divider()
    st.subheader("🏆 Top 5 Predictions")
    
    # Get top 5
    top_5_idx = np.argsort(confidence_scores)[-5:][::-1]
    
    # Display as table
    predictions_list = []
    for rank, idx in enumerate(top_5_idx, 1):
        breed_name = BREEDS[idx].replace('_', ' ').title()
        confidence = float(confidence_scores[idx]) * 100
        predictions_list.append({
            "Rank": rank,
            "Breed": breed_name,
            "Confidence": f"{confidence:.1f}%"
        })
    
    # Create and display table
    df = pd.DataFrame(predictions_list)
    st.table(df)

else:
    # Show tips when no image uploaded
    st.info("""
    ### 💡 Tips for Best Results
    - **Clear photos**: Well-lit images work best
    - **Full view**: Show face or full body
    - **Purebreds**: Works best with purebred dogs
    - **Mixed breeds**: May show multiple matches with similar confidence
    
    ### 📊 Model Info
    - **Training Dataset**: 10,000 images
    - **Dog Breeds**: 120 different breeds
    - **Model**: MobileNetV2 with custom layers
    """)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Made with ❤️ using TensorFlow & Streamlit | 
Dog Breed Classification Model
</div>
""", unsafe_allow_html=True)