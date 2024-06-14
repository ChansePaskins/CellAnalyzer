import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cellCount
from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer")
st.write("Upload an image and select parameters")

# Define default values for sliders
minimum_area = 200
average_cell_area = 400
connected_cell_area = 1000
lower_intensity = 0
upper_intensity = 60
block_size = 100
scaling = 3.75
morph_checkbox = True
kernel_size = 3
opening = True
closing = True
eroding = False
dilating = False
open_iter = 1
close_iter = 1
erode_iter = 1
dilate_iter = 1


# Create a form for user input
with st.container():
    # File uploader for image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif"])

    # Sliders for parameter selection
    columns = st.columns(2)
    with columns[0]:
        minimum_area = st.slider("Minimum Area to Be Consider Cell", min_value=1, max_value=2000, value=minimum_area, step=10)
        average_cell_area = st.slider("Average Size of Single Cell", min_value=1, max_value=4000, value=average_cell_area, step=10)
        connected_cell_area = st.slider(
            "Max Size of Cell (for cases where cells are stuck together)", min_value=1, max_value=5000, value=connected_cell_area, step=20)
        lower_intensity, upper_intensity = st.select_slider("Intensity Thresholds", options=list(range(101)),
                                                            value=(lower_intensity, upper_intensity))
        scaling = st.number_input("Scaling Between Pixels and Centimeters", value=3.75)
    with columns[1]:

        image_method = st.selectbox("Processing Method (I recommend using Sobel)",
                                    ("Sobel", "Canny", "Canny Channel", "Block Segmentation", "Histogram"))

        morph_checkbox = st.checkbox("Apply Morphological Transformations?", value=True)

        if image_method == "Block Segmentation":
            block_size = st.slider("Block Size", min_value=50, max_value=200, value=block_size, step=10)

        if morph_checkbox:
            kernel_size = st.slider("Kernel Size (must be odd number)", min_value=1, max_value=11, value=3, step=2)
            cl = st.columns(4)
            with cl[0]:
                eroding = st.checkbox("Perform Erosion?", value=True)
            with cl[1]:
                dilating = st.checkbox("Perform Dilation?", value=True)
            with cl[2]:
                opening = st.checkbox("Perform Opening?", value=True)
            with cl[3]:
                closing = st.checkbox("Perform Closing?", value=True)

            if eroding:
                erode_iter = st.slider("Erode Iterations", min_value=1, max_value=10, value=1)
            if dilating:
                dilate_iter = st.slider("Dilate Iterations",min_value=1, max_value=10, value=1)
            if opening:
                open_iter = st.slider("Opening Iterations", min_value=1, max_value=10, value=1)
            if closing:
                close_iter = st.slider("Closing Iterations",min_value=1, max_value=10, value=1)


if uploaded_file is not None:
    # Read the image
    # Read the uploaded file as a byte array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode the byte array to an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    normalized, morphed, mask, overlay, count, total_area, avg_area = cellCount.cell_detection(
        image, lower_intensity, upper_intensity, image_method,
        block_size, morph_checkbox, minimum_area, average_cell_area,
        connected_cell_area, scaling, kernel_size, opening, closing,
        eroding, dilating, open_iter, close_iter, erode_iter, dilate_iter
    )

    st.divider()

    cols = st.columns(3)

    with cols[0]:
        st.metric("Total Cell Count", count)
    with cols[1]:
        st.metric("Total Cell Area", f"{total_area} µm\u00b2")
    with cols[2]:
        st.metric("Average Cell Area", f"{average_cell_area} µm\u00b2")

    st.divider()

    cls = st.columns(3)

    # Display the original image
    with cls[0]:
        st.image(image, caption='Original Image', use_column_width=True)
        copy = image.copy()
        height, width = copy.shape[:2]

        # Define coordinates for cropping (you can adjust these values)
        x1, y1 = 0, 0  # Top-left corner of the cropped region
        x2, y2 = width//3, height//3  # Bottom-right corner of the cropped region

        # Crop the image using numpy slicing
        original_with_circles = copy[y1:y2, x1:x2]

        # Draw circles representing the areas
        cv2.circle(original_with_circles, (100, 100), int(np.sqrt(minimum_area / np.pi)), (0, 255, 0),
                   2)  # Green circle for minimum area
        cv2.circle(original_with_circles, (200, 100), int(np.sqrt(average_cell_area / np.pi)), (0, 0, 255),
                   2)  # Blue circle for average cell area
        cv2.circle(original_with_circles, (300, 100), int(np.sqrt(connected_cell_area / np.pi)), (255, 0, 0),
                   2)  # Red circle for connected cell area
        st.image(original_with_circles, use_column_width=True)
        'Original Image with :green[selected minimum area], :blue[average area], and :red[threshold for max cell size] displayed for reference'

    with cls[1]:
        st.image(normalized, caption='Processed Image', use_column_width=True)
        st.image(morphed, caption='Morphed Image', use_column_width=True) if morph_checkbox else None

    with cls[2]:
        st.image(overlay, caption='Overlayed Image', use_column_width=True)
        st.image(mask, caption="Masked Image", use_column_width=True)

    st.divider()

    with st.container():
        st.write("Slide to compare")
        image_comparison(image, overlay)
