import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cellCount
from streamlit_image_comparison import image_comparison
import os

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer Pipeline (Not finished)")
st.write("Once you have found settings that work for your batch, simply upload the folder to process all images")

# Define default values for parameters
minimum_area = 150  # lower threshold of area for what is considered a 'cell' (vs debris and noise)
average_cell_area = 550  # average size of cells (used for determining how many cells are in clusters)
connected_cell_area = 1000  # threshold for what is considered more than one cell
lower_intensity = 0  # lower brightness threshold used to differentiate between cell and background
upper_intensity = 60  # upper brightness threshold used to differentiate between cell and background
block_size = 100  # used for block segmentation method (outdated)
scaling = 3.75  # scaling between pixels and cm^2
fluorescence = False  # for when using fluorescence
morph_checkbox = True  # applies selectable morph operations
kernel_size = 3  # kernel size for morph operations
opening = True  # useful in removing noise - essentially is erosion followed by dilation
closing = True  # helps remove small holes in an object - essentially is dilation followed by erosion
eroding = False  # erodes away boundaries of an object
dilating = False  # thickens boundaries of an object
open_iter = 1  # how many times the open morph is applied
close_iter = 1  # how many times the close morph is applied
erode_iter = 1  # how many times the erode morph is applied
dilate_iter = 1  # how many times the dilate morph is applied
hole_size = connected_cell_area

# Create a form for user input
with st.expander("Parameters", expanded=True):
    # File uploader
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png", "bmp", "tif"], accept_multiple_files=True)

    st.divider()

    # various parameter selections
    with st.container():
        st.subheader("Cell Characteristics")
        columns = st.columns(2)
        with columns[0]:
            minimum_area = st.slider("Minimum Area to Be Consider Cell (µm\u00b2)", min_value=1, max_value=8000, value=minimum_area, step=2)
            average_cell_area = st.slider("Average Size of Single Cell (µm\u00b2)", min_value=1, max_value=8000, value=average_cell_area, step=2)

        with columns[1]:
            connected_cell_area = st.slider(
                "Max Size of Cell (µm\u00b2) (for cases where cells are stuck together)", min_value=1, max_value=8000, value=connected_cell_area, step=10)
            lower_intensity, upper_intensity = st.select_slider("Intensity Thresholds", options=list(range(101)),
                                                                value=(lower_intensity, upper_intensity))
            scaling = st.number_input("Scaling Between Pixels and Centimeters", value=0.595)

        st.divider()

    st.subheader("Image Processing")
    cl = st.columns(2)
    with cl[0]:
        image_method = st.selectbox("Processing Method (I recommend using Sobel, Canny, or Laplace)",
                                    ("Sobel", "Canny", "Laplace", "Block Segmentation", "Histogram (Not Working)", None))

        col = st.columns(2)
        with col[0]:
            morph_checkbox = st.checkbox("Apply Morphological Transformations?", value=True)
        with col[1]:
            fluorescence = st.checkbox("Use Fluorescence?")

        hole_size = st.slider("Minimum hole size for subtraction (µm\u00b2)", min_value=1, max_value=8000, value=minimum_area, step=2)

        if image_method == "Block Segmentation":
            block_size = st.slider("Block Size", min_value=50, max_value=200, value=block_size, step=10)

    with cl[1]:
        if morph_checkbox:
            kernel_size = st.slider("Kernel Size (must be odd number)", min_value=1, max_value=11, value=3, step=2)
            cl = st.columns(4)
            with cl[0]:
                opening = st.checkbox("Remove Noise? (Open Transform)", value=True)
            with cl[1]:
                closing = st.checkbox("Fill In Holes? (Close Transform)", value=True)
            with cl[2]:
                eroding = st.checkbox("Thin Borders? (Erode Transform)", value=False)
            with cl[3]:
                dilating = st.checkbox("Thicken Borders? (Dilate Transform)", value=False)
            if opening:
                open_iter = st.slider("Noise Removal Iterations", min_value=1, max_value=10, value=1)
            if closing:
                close_iter = st.slider("Hole Sealing Iterations", min_value=1, max_value=10, value=1)
            if eroding:
                erode_iter = st.slider("Border Thinning Iterations", min_value=1, max_value=10, value=1)
            if dilating:
                dilate_iter = st.slider("Border Thickening Iterations", min_value=1, max_value=10, value=1)

# Run test button
if st.button("Run Test"):
    if uploaded_files:
        uploaded_file = uploaded_files[0]

        if uploaded_file is not None:
            # turns the image file into an array that OpenCV can understand and decode
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            # Decode the byte array to an image
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # change thresholds back to pixels from um2
            min_area_px = minimum_area * scaling ** 2
            avg_cell_area_px = average_cell_area * scaling ** 2
            conn_cell_area_px = connected_cell_area * scaling ** 2
            height, width = image.shape[:2]
            overall_area = (height * width) / scaling**2

            # Calls master function to perform all operations using selected parameters
            normalized, morphed, mask, overlay, count, total_area, threshold_area, avg_area = cellCount.cell_detection(
                image, lower_intensity, upper_intensity, fluorescence, image_method,
                block_size, hole_size, morph_checkbox, min_area_px, avg_cell_area_px,
                conn_cell_area_px, scaling, kernel_size, opening, closing,
                eroding, dilating, open_iter, close_iter, erode_iter, dilate_iter
            )

            # Display metrics
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
                    st.metric("Average Cell Area", f"{round(total_area/count, 2)} µm\u00b2")
                else:
                    st.metric("Average Cell Area", f"{0} µm\u00b2")

            st.divider()

            # Image comparison slider
            with st.container():
                st.write("Slide to compare")
                image_comparison(image, overlay)

# Run batch button
if st.button("Run Batch"):
    if uploaded_files:
        metrics = []

        for uploaded_file in uploaded_files:
            # turns the image file into an array that OpenCV can understand and decode
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            # Decode the byte array to an image
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # change thresholds back to pixels from um2
            min_area_px = minimum_area * scaling ** 2
            avg_cell_area_px = average_cell_area * scaling ** 2
            conn_cell_area_px = connected_cell_area * scaling ** 2
            height, width = image.shape[:2]
            overall_area = (height * width) / scaling**2

            # Calls master function to perform all operations using selected parameters
            normalized, morphed, mask, overlay, count, total_area, threshold_area, avg_area = cellCount.cell_detection(
                image, lower_intensity, upper_intensity, fluorescence, image_method,
                block_size, hole_size, morph_checkbox, min_area_px, avg_cell_area_px,
                conn_cell_area_px, scaling, kernel_size, opening, closing,
                eroding, dilating, open_iter, close_iter, erode_iter, dilate_iter
            )

            # Store metrics
            metrics.append({
                "Filename": uploaded_file.name,
                "Total Cell Count": count,
                "Total Area of Picture (µm²)": round(overall_area),
                "Total Cell Area (by contours) (µm²)": total_area,
                "Total Cell Area (by threshold) (µm²)": threshold_area,
                "Average Cell Area (µm²)": round(total_area/count, 2) if count > 0 else 0
            })

            # Save overlay image
            output_dir = "output_images"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            overlay_image_path = os.path.join(output_dir, f"{uploaded_file.name}_overlay.png")
            cv2.imwrite(overlay_image_path, overlay)

            # Display images
            st.divider()
            cls = st.columns(3)
            with cls[0]:
                st.image(image, caption='Original Image', use_column_width=True)
                st.image(mask, caption="Masked Image", use_column_width=True)

            with cls[1]:
                copy = image.copy()
                height, width = copy.shape[:2]
                cv2.circle(copy, (int(width * 0.25), int(height * 0.5)), int(np.sqrt(min_area_px / np.pi)), (255, 200, 0), 4)
                cv2.circle(copy, (int(width * 0.5), int(height * 0.5)), int(np.sqrt(avg_cell_area_px / np.pi)), (0, 0, 255), 4)
                cv2.circle(copy, (int(width * 0.75), int(height * 0.5)), int(np.sqrt(conn_cell_area_px / np.pi)), (255, 0, 0), 4)
                st.image(copy, use_column_width=True)
                st.caption('Original Image with :green[minimum area], :blue[average area], and :red[max cell size] displayed for reference')
                if morph_checkbox:
                    st.image(morphed, caption='Morphed Image', use_column_width=True)
                else:
                    st.info("Morph Options Unselected")

            with cls[2]:
                st.image(normalized, caption='Processed Image', use_column_width=True)
                st.image(overlay, caption='Overlayed Image', use_column_width=True)

            st.divider()
            st.write("Slide to compare")
            image_comparison(image, overlay)

        # Create DataFrame and download link
        df = pd.DataFrame(metrics)
        st.write(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Metrics as CSV", data=csv, file_name='metrics.csv', mime='text/csv')

        st.write("Overlay images are saved in the output_images directory.")
