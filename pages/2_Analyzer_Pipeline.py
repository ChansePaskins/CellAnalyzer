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
from Parameters import parameters, csv_bool_error

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer Pipeline")
st.write(
    "Upload as many files as you need. Run 'Test' to check your settings on the first file. After you find the settings you like, run 'Batch' to iterate through all files.")


# File uploader
uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png", "bmp", "tif"],
                           accept_multiple_files=True)

params = parameters()

st.divider()

# Run test button
if st.button("Run Test"):
    if uploaded_files:
        uploaded_file = uploaded_files[0]

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

                if width / np.sqrt(params['minimum_area'] / np.pi) > 75:

                    # Cropping parameters (to better visualize cell sizing)
                    x1, y1 = 0, 0  # Top-left corner of the cropped region
                    x2, y2 = width // 2, height // 2  # Bottom-right corner of the cropped region
                    cropped = copy[y1:y2, x1:x2]  # Slices image

                    # Resize the cropped image back to the original size
                    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

                    # Draw circles representing the areas to scale
                    cv2.circle(resized, (int(width * 0.25), int(height * 0.5)), int(np.sqrt(params['minimum_area'] / np.pi) * 2),
                               (255, 200, 0),
                               4)  # Green circle for minimum area
                    cv2.circle(resized, (int(width * 0.5), int(height * 0.5)),
                               int(np.sqrt(params['average_cell_area'] / np.pi) * 2), (0, 0, 255),
                               4)  # Blue circle for average cell area
                    cv2.circle(resized, (int(width * 0.75), int(height * 0.5)),
                               int(np.sqrt(params['connected_cell_area'] / np.pi) * 2), (255, 0, 0),
                               4)  # Red circle for connected cell area
                    st.image(resized, use_column_width=True)
                    st.caption(
                        'Original Image with :orange[minimum area], :blue[average area], and :red[max cell size] displayed for reference')

                else:
                    # Draw circles representing the areas to scale
                    cv2.circle(copy, (int(width * 0.25), int(height * 0.5)), int(np.sqrt(params['minimum_area'] / np.pi)),
                               (255, 200, 0),
                               4)  # Green circle for minimum area
                    cv2.circle(copy, (int(width * 0.5), int(height * 0.5)), int(np.sqrt(params['average_cell_area'] / np.pi)),
                               (0, 0, 255),
                               4)  # Blue circle for average cell area
                    cv2.circle(copy, (int(width * 0.75), int(height * 0.5)), int(np.sqrt(params['connected_cell_area'] / np.pi)),
                               (255, 0, 0),
                               4)  # Red circle for connected cell area
                    st.image(copy, use_column_width=True)
                    st.caption(
                        'Original Image with :green[minimum area], :blue[average area], and :red[max cell size] displayed for reference')
                ####################################################################################

                # Displays image after morphological operations
                if params['morph_checkbox']:
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

            params['minimum_area'] = params['minimum_area'] * params['scaling'] ** 2
            params['average_cell_area'] = params['average_cell_area'] * params['scaling'] ** 2
            params['connected_cell_area'] = params['connected_cell_area'] * params['scaling'] ** 2
            height, width = image.shape[:2]
            overall_area = (height * width) / params['scaling'] ** 2

            normalized, morphed, mask, overlay, count, total_area, threshold_area, avg_area = cellCount.cell_detection(
                image, **params)

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
                st.image(morph_dict[item], width=485)

        # Create DataFrame and download link
        df = pd.DataFrame(metrics)
        st.write(df)

        # Store selected settings
        settings = {
            "Minimum Area": params['minimum_area'] / params['scaling'] ** 2,
            "Average Cell Area": params['average_cell_area'] / params['scaling'] ** 2,
            "Connected Cell Area": params['connected_cell_area'] / params['scaling'] ** 2,
            "Lower Intensity": params['lower_intensity'],
            "Upper Intensity": params['upper_intensity'],
            "Block Size": params['block_size'],
            "Scaling": params['scaling'],
            "Fluorescence": params['fluorescence'],
            "Image Method": params['image_method'],
            "Morphological Transformations": params['morph_checkbox'],
            "Kernel Size": params['kernel_size'],
            "Noise Removal (Open Transform)": params['opening'],
            "Hole Filling (Close Transform)": params['closing'],
            "Border Thinning (Erode Transform)": params['eroding'],
            "Border Thickening (Dilate Transform)": params['dilating'],
            "Noise Removal Iterations": params['open_iter'],
            "Hole Sealing Iterations": params['close_iter'],
            "Border Thinning Iterations": params['erode_iter'],
            "Border Thickening Iterations": params['dilate_iter'],
            "Hole Size": params['hole_size']
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
