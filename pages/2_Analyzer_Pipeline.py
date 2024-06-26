import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cellCount
from streamlit_image_comparison import image_comparison
import os
import zipfile
import io
from datetime import datetime

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer Pipeline")
st.write(
    "Upload as many files as you need. Run 'Test' to check your settings on the first file. After you find the settings you like, run 'Batch' to iterate through all files.")

# Define default values for parameters
minimum_area = 150  # lower threshold of area for what is considered a 'cell' (vs debris and noise)
average_cell_area = 550  # average size of cells (used for determining how many cells are in clusters)
connected_cell_area = 1000  # threshold for what is considered more than one cell
lower_intensity = 0  # lower brightness threshold used to differentiate between cell and background
upper_intensity = 60  # upper brightness threshold used to differentiate between cell and background
block_size = 100  # used for block segmentation method (outdated)
scaling = 3.75  # scaling between pixels and cm^2
fluorescence = False  # for when using fluorescence
image_method = 0
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
param_bool = True


with st.expander("Upload Settings", expanded=False):
    settings_upload = st.file_uploader("Upload Previous Settings Here", type="csv")


if settings_upload:
    param_bool = False
    settings_df = pd.read_csv(settings_upload, index_col="Parameter")
    # Define default values for parameters
    minimum_area = int(settings_df.loc["Minimum Area", "Value"])
    average_cell_area = int(settings_df.loc["Average Cell Area", "Value"])
    connected_cell_area = int(settings_df.loc["Connected Cell Area", "Value"])
    lower_intensity = int(settings_df.loc["Lower Intensity", "Value"])
    upper_intensity = int(settings_df.loc["Upper Intensity", "Value"])
    block_size = int(settings_df.loc["Block Size", "Value"])
    scaling = float(settings_df.loc["Scaling", "Value"])
    fluorescence = settings_df.loc["Fluorescence", "Value"]
    image_method = int(settings_df.loc["Image Method", "Value"])
    morph_checkbox = settings_df.loc["Morphological Transformations", "Value"]
    kernel_size = int(settings_df.loc["Kernel Size", "Value"])
    opening = settings_df.loc["Noise Removal (Open Transform)", "Value"]
    closing = settings_df.loc["Hole Filling (Close Transform)", "Value"]
    eroding = settings_df.loc["Border Thinning (Erode Transform)", "Value"]
    dilating = settings_df.loc["Border Thickening (Dilate Transform)", "Value"]
    open_iter = int(settings_df.loc["Noise Removal Iterations", "Value"])
    close_iter = int(settings_df.loc["Hole Sealing Iterations", "Value"])
    erode_iter = int(settings_df.loc["Border Thinning Iterations", "Value"])
    dilate_iter = int(settings_df.loc["Border Thickening Iterations", "Value"])
    hole_size = int(settings_df.loc["Hole Size", "Value"])

# File uploader
uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png", "bmp", "tif"],
                                  accept_multiple_files=True)

st.divider()

# Create a form for user input
with st.expander("Parameters", expanded=param_bool):

    # various parameter selections
    with st.container():
        st.subheader("Cell Characteristics")
        columns = st.columns(2)
        with columns[0]:
            minimum_area = st.number_input("Minimum Area to Be Consider Cell (µm\u00b2)", value=minimum_area)
            average_cell_area = st.number_input("Average Size of Single Cell (µm\u00b2)", value=average_cell_area)

        with columns[1]:
            connected_cell_area = st.number_input(
                "Max Size of Cell (µm\u00b2) (for cases where cells are stuck together)", value=connected_cell_area)
            lower_intensity, upper_intensity = st.select_slider("Intensity Thresholds", options=list(range(101)),
                                                                value=(lower_intensity, upper_intensity))
            scaling = st.number_input("Scaling Between Pixels and Micrometers", value=0.595)

        st.divider()

    st.subheader("Image Processing")
    cl = st.columns(2)
    with cl[0]:
        image_method = st.selectbox("Processing Method (I recommend using Sobel, Canny, or Laplace)",
                                    (
                                    "Sobel", "Canny", "Laplace", "Block Segmentation", "Histogram (Not Working)"),
                                    index=image_method)

        col = st.columns(2)
        with col[0]:
            morph_checkbox = st.checkbox("Apply Morphological Transformations?", value=True)
        with col[1]:
            fluorescence = st.checkbox("Use Fluorescence?")

        hole_size = st.slider("Minimum hole size for subtraction (µm\u00b2)", min_value=1, max_value=8000,
                              value=minimum_area, step=2)

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
            overall_area = (height * width) / scaling ** 2

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
                    st.metric("Average Cell Area", f"{round(total_area / count, 2)} µm\u00b2")
                else:
                    st.metric("Average Cell Area", f"{0} µm\u00b2")

            st.divider()

            # Displays images
            cls = st.columns(3)
            with cls[0]:
                # Displays original image
                st.image(image, caption='Original Image', use_column_width=True)

                st.image(mask, caption="Masked Image (Using intensity thresholds defined above)", use_column_width=True)

            with cls[1]:
                #####################################################################################
                # Displays original image along with contours for minimum cell area, average cell size, and max cell size
                copy = image.copy()
                height, width = copy.shape[:2]

                if width / np.sqrt(minimum_area / np.pi) > 75:

                    # Cropping parameters (to better visualize cell sizing)
                    x1, y1 = 0, 0  # Top-left corner of the cropped region
                    x2, y2 = width // 2, height // 2  # Bottom-right corner of the cropped region
                    cropped = copy[y1:y2, x1:x2]  # Slices image

                    # Resize the cropped image back to the original size
                    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

                    # Draw circles representing the areas to scale
                    cv2.circle(resized, (int(width * 0.25), int(height * 0.5)), int(np.sqrt(minimum_area / np.pi) * 2),
                               (255, 200, 0),
                               4)  # Green circle for minimum area
                    cv2.circle(resized, (int(width * 0.5), int(height * 0.5)),
                               int(np.sqrt(average_cell_area / np.pi) * 2), (0, 0, 255),
                               4)  # Blue circle for average cell area
                    cv2.circle(resized, (int(width * 0.75), int(height * 0.5)),
                               int(np.sqrt(connected_cell_area / np.pi) * 2), (255, 0, 0),
                               4)  # Red circle for connected cell area
                    st.image(resized, use_column_width=True)
                    st.caption(
                        'Original Image with :orange[minimum area], :blue[average area], and :red[max cell size] displayed for reference')

                else:
                    # Draw circles representing the areas to scale
                    cv2.circle(copy, (int(width * 0.25), int(height * 0.5)), int(np.sqrt(minimum_area / np.pi)),
                               (255, 200, 0),
                               4)  # Green circle for minimum area
                    cv2.circle(copy, (int(width * 0.5), int(height * 0.5)), int(np.sqrt(average_cell_area / np.pi)),
                               (0, 0, 255),
                               4)  # Blue circle for average cell area
                    cv2.circle(copy, (int(width * 0.75), int(height * 0.5)), int(np.sqrt(connected_cell_area / np.pi)),
                               (255, 0, 0),
                               4)  # Red circle for connected cell area
                    st.image(copy, use_column_width=True)
                    st.caption(
                        'Original Image with :green[minimum area], :blue[average area], and :red[max cell size] displayed for reference')
                ####################################################################################

                # Displays image after morphological operations
                if morph_checkbox:
                    st.image(morphed, caption='Morphed Image (fills in holes and borders)', use_column_width=True)
                else:
                    st.info("Morph Options Unselected")
                #######################################################################################

            with cls[2]:
                # Displays image after edge detection processing
                st.image(normalized, caption='Processed Image (Image made from edge detection algorithm)',
                         use_column_width=True)

                # Displays original image with calculated contours overlayed
                st.image(overlay,
                         caption='Overlayed Image (green is area counted, red (if there is any) are detected holes)',
                         use_column_width=True)

            st.divider()

            # Image comparison slider
            with st.container():
                st.write("Slide to compare")
                image_comparison(image, overlay)

# Run batch button
if st.button("Run Batch"):
    if uploaded_files:
        metrics = []
        images_dict = {}
        overlays_dict = {}
        morph_dict = {}

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
            overall_area = (height * width) / scaling ** 2

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
                "Average Cell Area (µm²)": round(total_area / count, 2) if count > 0 else 0
            })

            # Save overlay image
            images_dict[f"{uploaded_file.name}"] = image
            overlays_dict[f"{uploaded_file.name}"] = overlay
            morph_dict[f"{uploaded_file.name} morphed.tif"] = morphed

        image_cols = st.columns(2)

        with image_cols[0]:
            for item in images_dict:
                st.caption(f"{item}")
                image_comparison(images_dict[item], overlays_dict[item], width=500)

        with image_cols[1]:
            for item in morph_dict:
                st.caption(f"{item}")
                st.image(morph_dict[item], width=450)

        # Create DataFrame and download link
        df = pd.DataFrame(metrics)
        st.write(df)

        # Store selected settings
        settings = {
            "Minimum Area": minimum_area,
            "Average Cell Area": average_cell_area,
            "Connected Cell Area": connected_cell_area,
            "Lower Intensity": lower_intensity,
            "Upper Intensity": upper_intensity,
            "Block Size": block_size,
            "Scaling": scaling,
            "Fluorescence": fluorescence,
            "Image Method": image_method,
            "Morphological Transformations": morph_checkbox,
            "Kernel Size": kernel_size,
            "Noise Removal (Open Transform)": opening,
            "Hole Filling (Close Transform)": closing,
            "Border Thinning (Erode Transform)": eroding,
            "Border Thickening (Dilate Transform)": dilating,
            "Noise Removal Iterations": open_iter,
            "Hole Sealing Iterations": close_iter,
            "Border Thinning Iterations": erode_iter,
            "Border Thickening Iterations": dilate_iter,
            "Hole Size": hole_size
        }
        # Create a DataFrame from the settings dictionary
        settings_df = pd.DataFrame(list(settings.items()), columns=['Parameter', 'Value'])

        csv = df.to_csv(index=False).encode('utf-8')
        settings_csv = settings_df.to_csv(index=False).encode('utf-8')

        # Get the current date
        current_date = datetime.now()
        # Format the current date as "06June2024"
        formatted_date = current_date.strftime("%d%b%Y")
        # Use the formatted date in a filename
        zip_filename = f"Batch {formatted_date}.zip"  # Example filename

        # Create a zip file containing all images
        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, 'w') as zf:
            for filename, image_data in overlays_dict.items():
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                img_bytes = cv2.imencode('.tif', image_data)[1].tobytes()
                zf.writestr(filename, img_bytes)
            for filename, image_data in morph_dict.items():
                img_bytes = cv2.imencode('.tif', image_data)[1].tobytes()
                zf.writestr(filename, img_bytes)
            zf.writestr("metrics.csv", csv)
            zf.writestr("settings.csv", settings_csv)

        # Offer zip file for download
        st.download_button(label="Download Data", data=zip_bytes.getvalue(), file_name=zip_filename,
                           mime='application/zip')
