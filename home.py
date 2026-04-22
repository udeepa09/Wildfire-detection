import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests # type: ignore
from PIL import Image
import os
from glob import glob
from numpy import random
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    # Predict objects using the model
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )
    
    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}
    
    # Count the number of occurrences for each class
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    # Generate prediction text
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'
        if v > 1:
            prediction_text += 's'
        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    # Calculate inference latency
    latency = sum(res[0].speed.values())  
    latency = round(latency / 1000, 2)
    prediction_text += f' in {latency} seconds.'

    # Convert the result image to RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
    
    return res_image, prediction_text

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",
    )
    
    # --- ADDED IMAGE PART BACK ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logos = glob('dalle-logos/*.png')
        if logos:
            logo = random.choice(logos)
            st.image(logo, use_column_width=True, caption="Wildfire Detection System")
    
    st.markdown("<h1 style='text-align: center;'>Wildfire Detection</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # --- SETTINGS HARDCODED IN BACKGROUND ---
    model_type = "Fire Detection"
    selected_model = "fire_n"
    models_dir = "fire-models"
    model_path = os.path.join(models_dir, selected_model + ".pt")
    
    # Load the selected model
    model = load_model(model_path)

    # --- THRESHOLDS SECTION ---
    st.write("### Detection Settings")
    col_a, col_b = st.columns(2)
    with col_a:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)
    with col_b:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)

    st.markdown("---")

    # Image selection
    st.write("### Step 1: Provide an Image")
    image = None
    image_source = st.radio("Select image source:", ("Upload from Computer", "Enter URL"))
    
    if image_source == "Upload from Computer":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Error loading image from URL.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Detection Output
    if image:
        st.markdown("---")
        st.write("### Step 2: Detection Result")
        with st.spinner("Analyzing image..."):
            prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
            st.image(prediction, caption="AI Detection Result", use_column_width=True)
            st.success(text)
        
        # Download Option
        prediction_pil = Image.fromarray(prediction)
        image_buffer = io.BytesIO()
        prediction_pil.save(image_buffer, format='PNG')

        st.download_button(
            label='Download Result Image',
            data=image_buffer.getvalue(),
            file_name='wildfire_detection_result.png',
            mime='image/png'
        )

if __name__ == "__main__":
    main()