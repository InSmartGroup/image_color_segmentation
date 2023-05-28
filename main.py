import cv2
import numpy as np
import streamlit as st
from io import BytesIO
import PIL

# GUI layout
st.set_page_config(layout="wide")
st.title("Image Color Segmentation")
st.write("Download an image and use sliders to segment the image by colors")

column_1, column_2 = st.columns([1, 1])

column_slider_1, column_slider_2, column_slider_3, \
column_slider_4, column_slider_5, column_slider_6 = st.columns(6)

#
source_image = st.file_uploader("Download an image", type=["JPG", "JPEG", "PNG"])

if source_image is not None:
    image_rawbytes = np.asarray(bytearray(source_image.read()), dtype=np.uint8)
    image_input = cv2.imdecode(image_rawbytes, cv2.IMREAD_COLOR)
    image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    image_output = image_input.copy()

    image_output = cv2.cvtColor(image_output, cv2.COLOR_RGB2HSV)

    hue_min = column_slider_1.slider("Hue Minimum", 0, 180, 0)
    hue_max = column_slider_2.slider("Hue Maximum", 0, 180, 180)
    saturation_min = column_slider_3.slider("Saturation Minimum", 0, 255, 0)
    saturation_max = column_slider_4.slider("Saturation Maximum", 0, 255, 255)
    value_min = column_slider_5.slider("Value Minimum", 0, 255, 0)
    value_max = column_slider_6.slider("Value Maximum", 0, 255, 255)

    lower_boundary = np.array([hue_min, saturation_min, value_min])
    upper_boundary = np.array([hue_max, saturation_max, value_max])

    image_mask = cv2.inRange(image_output, lower_boundary, upper_boundary)

    image_output = cv2.bitwise_and(image_input, image_input, mask=image_mask)

    column_1.image(image_input)
    column_2.image(image_output)
