# Python In-built packages
from pathlib import Path
import PIL
import numpy as np
import tempfile
import cv2

# External packages
import streamlit as st

# Local Modules
import settings
import helper

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from sys import platform
if platform == "linux" or platform == "linux2":
    font = 'DejaVuSans'
elif platform == "win32":
    font = 'arial'

# Setting page layout
st.set_page_config(
    page_title="Object Detection",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load Pre-trained ML Model
try:
    model = helper.load_model()
    processor = helper.load_processor()

except Exception as ex:
    st.error(f"Unable to load model.")
    st.error(ex)

# Main page heading
st.title("Object Detection using DEtection TRansformer (DETR)")

# Sidebar
st.sidebar.header("ML Model Config")

# Detection Options
detection_type = st.sidebar.radio(
    "Select Detection Type", ['All', 'Deer', 'Custom'])

if detection_type == "Custom":
    options = st.sidebar.multiselect(
        'What do you want to find?',
        list(model.config.label2id.keys()),
        )

    detection_list = [model.config.label2id[x] for x in options]
elif detection_type == "Deer":
    detection_list = [19, 20, 21, 23, 24, 25]
else:
    detection_list = [x for x in model.config.label2id.values()]


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

data = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp', 'mp4'),
    accept_multiple_files=True)

col1, col2 = st.columns(2)

if len(data) > 0:

    for source_img in data:

        with col1:
            try:
                if source_img is not None:
                            # check if the file is a video
                    if 'video' in source_img.type:

                        with open('temp_vid.mp4', mode='wb') as f:
                            f.write(source_img.read()) # save video to disk

                        vidcap = cv2.VideoCapture('temp_vid.mp4')

                        success, frame = vidcap.read() # get next frame from video
                        uploaded_image = PIL.Image.fromarray(frame[:,:,[2,1,0]]) # convert opencv frame (with type()==numpy) into PIL Image
                        
                        vidcap.release()
                        cv2.destroyAllWindows()
                    else:
                        uploaded_image = PIL.Image.open(source_img)

                    st.image(uploaded_image, caption="Uploaded Image",
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
                    item_of_interest = np.where(np.isin(res['labels'], detection_list))[0]
                    labels = [f"Confidence: {round(x*100)}" for x in res['scores'][item_of_interest].tolist()]
                else:
                    item_of_interest = np.where(np.isin(res['labels'], detection_list))[0]
                    labels = [f"{model.config.id2label[x]}: conf. {round(score, 3)}" for x, score in zip(res['labels'][item_of_interest].tolist(), res['scores'][item_of_interest].tolist())]
                
                im = to_pil_image(
                    draw_bounding_boxes(
                        pil_to_tensor(uploaded_image),
                        res['boxes'][item_of_interest],
                        colors="red",
                        width=int(target_sizes[0][1]/100),
                        font_size=int(target_sizes[0][1]/25),
                        font=font,
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
