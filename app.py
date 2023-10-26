# Python In-built packages
from pathlib import Path
import PIL
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

large_animal_list = [19, 20, 21, 23, 24, 25]

# Setting page layout
st.set_page_config(
    page_title="Object Detection",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using DEtection TRansformer (DETR)")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

# Detection Options
detection_type = st.sidebar.radio(
    "Select Task", ['All', 'Deer'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model()
    processor = helper.load_processor()

except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

# If image is selected
if source_radio == settings.IMAGE:
    data = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'),
        accept_multiple_files=True)

    col1, col2 = st.columns(2)

    if len(data) > 0:

        for source_img in data:

            with col1:
                try:
                    if source_img is not None:
                        uploaded_image = PIL.Image.open(source_img)
                        st.image(source_img, caption="Uploaded Image",
                                    use_column_width=True)
                except Exception as ex:
                    st.error("Error occurred while opening the image.")
                    st.error(ex)

            with col2:
                if source_img is not None:
                    inputs = processor(images=uploaded_image, return_tensors="pt")
                    outputs = model(**inputs)

                    target_sizes = [uploaded_image.size[::-1]]
                    res = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=(confidence))[0]
# filter the results to just the animals of interest
                    if detection_type == 'Deer':
                        animal_of_interest = np.where(np.isin(res['labels'], large_animal_list))[0]
                        labels = [f"Confidence: {round(x*100)}" for x in res['scores'][animal_of_interest].tolist()]
                    else:
                        animal_of_interest = [True for x in res['labels']]
                        labels = [f"{model.config.id2label[x]}: conf. {round(score, 3)}" for x, score in zip(res['labels'].tolist(),res['scores'].tolist())]
                    im = to_pil_image(
                        draw_bounding_boxes(
                            pil_to_tensor(uploaded_image),
                            res['boxes'][animal_of_interest],
                            colors="red",
                            width=int(target_sizes[0][1]/100),
                            font_size=int(target_sizes[0][1]/25),
                            font='arial',
                            labels = labels
                                      
                                    )
                                )
            

                    st.image(im, caption='Detected Image',
                                use_column_width=True)
                    # try:
                    #     with st.expander("Detection Results"):
                    #         for box in res['boxes']:
                    #             st.write(box.data)
                    # except Exception as ex:
                    #     # st.write(ex)
                    #     st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")
