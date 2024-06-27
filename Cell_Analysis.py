import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cellCount
from streamlit_image_comparison import image_comparison
import pandas as pd
from Parameters import parameters

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer")
st.write("Upload an image and select parameters")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif"])
st.divider()

params = parameters()

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    params['minimum_area'] = params['minimum_area'] * params['scaling'] ** 2
    params['average_cell_area'] = params['average_cell_area'] * params['scaling'] ** 2
    params['connected_cell_area'] = params['connected_cell_area'] * params['scaling'] ** 2
    height, width = image.shape[:2]
    overall_area = (height * width) / params['scaling'] ** 2

    normalized, morphed, mask, overlay, count, total_area, threshold_area, avg_area = cellCount.cell_detection(
        image, **params)

    st.divider()
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Cell Count", count)
    with cols[1]:
        st.metric("Total Area of Picture", f"{round(overall_area)} µm\u00b2")
    with cols[2]:
        st.metric("Total Cell Area (by contours)", f"{total_area} µm\u00b2")
    with cols[3]:
        st.metric("Total Cell Area (by threshold)", f"{threshold_area} µm\u00b2")
    with cols[4]:
        if count > 0:
            st.metric("Average Cell Area", f"{round(total_area / count, 2)} µm\u00b2")
        else:
            st.metric("Average Cell Area", f"{0} µm\u00b2")

    st.divider()
    cls = st.columns(3)
    with cls[0]:
        st.image(image, caption='Original Image', use_column_width=True)
        st.image(mask, caption="Masked Image (Using intensity thresholds defined above)", use_column_width=True)
    with cls[1]:
        copy = image.copy()
        height, width = copy.shape[:2]

        if width / np.sqrt(params['minimum_area'] / np.pi) > 75:
            x1, y1 = 0, 0
            x2, y2 = width // 2, height // 2
            cropped = copy[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

            cv2.circle(resized, (int(width * 0.25), int(height * 0.5)),
                       int(np.sqrt(params['minimum_area'] / np.pi) * 2), (255, 200, 0), 4)
            cv2.circle(resized, (int(width * 0.5), int(height * 0.5)),
                       int(np.sqrt(params['average_cell_area'] / np.pi) * 2), (0, 0, 255), 4)
            cv2.circle(resized, (int(width * 0.75), int(height * 0.5)),
                       int(np.sqrt(params['connected_cell_area'] / np.pi) * 2), (255, 0, 0), 4)
            st.image(resized, use_column_width=True)
            st.caption(
                'Original Image with :orange[minimum area], :blue[average area], and :red[max cell size] displayed for reference')
        else:
            cv2.circle(copy, (int(width * 0.25), int(height * 0.5)),
                       int(np.sqrt(params['minimum_area'] / np.pi)), (255, 200, 0), 4)
            cv2.circle(copy, (int(width * 0.5), int(height * 0.5)),
                       int(np.sqrt(params['average_cell_area'] / np.pi)), (0, 0, 255), 4)
            cv2.circle(copy, (int(width * 0.75), int(height * 0.5)),
                       int(np.sqrt(params['connected_cell_area'] / np.pi)), (255, 0, 0), 4)
            st.image(copy, use_column_width=True)
            st.caption(
                'Original Image with :green[minimum area], :blue[average area], and :red[max cell size] displayed for reference')

        if params['morph_checkbox']:
            st.image(morphed, caption='Morphed Image (fills in holes and borders)', use_column_width=True)
        else:
            st.info("Morph Options Unselected")

    with cls[2]:
        st.image(normalized, caption='Processed Image (Image made from edge detection algorithm)',
                 use_column_width=True)
        st.image(overlay, caption='Overlayed Image (green is area counted, red (if there is any) are detected holes)',
                 use_column_width=True)

    st.divider()
    with st.container():
        st.write("Slide to compare")
        image_comparison(image, overlay)
