import streamlit as st
import cv2
import numpy as np
import pandas as pd
import masterFunction
from streamlit_image_comparison import image_comparison
import os
import zipfile
import io
from datetime import datetime
from parameters import parameters, csv_bool_error

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer Pipeline")
st.write(
    "Upload as many files as you need. Run 'Test' to check your settings on the first file. After you find the settings you like, run 'Batch' to iterate through all files."
)

# File uploader
uploaded_files = st.file_uploader(
    "Choose images",
    type=["jpg", "jpeg", "png", "bmp", "tif"],
    accept_multiple_files=True,
)

# Load parameters from external file
params = parameters()

st.divider()

# Run test button
if st.button("Run Test"):
    if uploaded_files:
        uploaded_file = uploaded_files[0]  # Test the first uploaded file

        if uploaded_file is not None:
            # Convert the uploaded file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Adjust parameters based on scaling factor
            params['minimum_area'] = params['minimum_area'] * params['scaling'] ** 2
            params['average_cell_area'] = params['average_cell_area'] * params['scaling'] ** 2
            params['connected_cell_area'] = params['connected_cell_area'] * params['scaling'] ** 2

            # Calculate the overall area of the image
            height, width = image.shape[:2]
            overall_area = (height * width) / params['scaling'] ** 2

            # Run cell detection algorithm
            normalized, morphed, mask, overlay, count, total_area, threshold_area, avg_area, fluorescence_intensities = masterFunction.cell_detection(
                image, **params
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

            # Display images
            cls = st.columns(3)
            with cls[0]:
                st.image(image, caption='Original Image', use_column_width=True)
                st.image(mask, caption="Masked Image (Using intensity thresholds defined above)", use_column_width=True)
            with cls[1]:
                # Create a copy of the image to draw contours
                copy = image.copy()
                height, width = copy.shape[:2]

                # If the image is large enough, crop and resize for better visualization
                if width / np.sqrt(params['minimum_area'] / np.pi) > 75:
                    x1, y1 = 0, 0
                    x2, y2 = width // 2, height // 2
                    cropped = copy[y1:y2, x1:x2]
                    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

                    # Draw circles representing cell size thresholds
                    cv2.circle(
                        resized,
                        (int(width * 0.25), int(height * 0.5)),
                        int(np.sqrt(params['minimum_area'] / np.pi) * 2),
                        (255, 200, 0),
                        4,
                    )  # Orange circle for minimum area
                    cv2.circle(
                        resized,
                        (int(width * 0.5), int(height * 0.5)),
                        int(np.sqrt(params['average_cell_area'] / np.pi) * 2),
                        (0, 0, 255),
                        4,
                    )  # Blue circle for average cell area
                    cv2.circle(
                        resized,
                        (int(width * 0.75), int(height * 0.5)),
                        int(np.sqrt(params['connected_cell_area'] / np.pi) * 2),
                        (255, 0, 0),
                        4,
                    )  # Red circle for connected cell area
                    st.image(resized, use_column_width=True)
                    st.caption(
                        'Original Image with :orange[minimum area], :blue[average area], and :red[max cell size] displayed for reference'
                    )
                else:
                    # Draw circles on the original image for smaller images
                    cv2.circle(
                        copy,
                        (int(width * 0.25), int(height * 0.5)),
                        int(np.sqrt(params['minimum_area'] / np.pi)),
                        (255, 200, 0),
                        4,
                    )  # Orange circle for minimum area
                    cv2.circle(
                        copy,
                        (int(width * 0.5), int(height * 0.5)),
                        int(np.sqrt(params['average_cell_area'] / np.pi)),
                        (0, 0, 255),
                        4,
                    )  # Blue circle for average cell area
                    cv2.circle(
                        copy,
                        (int(width * 0.75), int(height * 0.5)),
                        int(np.sqrt(params['connected_cell_area'] / np.pi)),
                        (255, 0, 0),
                        4,
                    )  # Red circle for connected cell area
                    st.image(copy, use_column_width=True)
                    st.caption(
                        'Original Image with :green[minimum area], :blue[average area], and :red[max cell size] displayed for reference'
                    )

                # Display morphed image if morphological transformations are enabled
                if params['morph_checkbox']:
                    st.image(morphed, caption='Morphed Image (fills in holes and borders)', use_column_width=True)
                else:
                    st.info("Morph Options Unselected")

            with cls[2]:
                # Display processed image and overlay
                st.image(normalized, caption='Processed Image (Image made from edge detection algorithm)', use_column_width=True)
                st.image(overlay, caption='Overlayed Image (green is area counted, red (if there is any) are detected holes)', use_column_width=True)

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
        process_dict = {}

        # Adjust parameters based on scaling factor
        params['minimum_area'] = params['minimum_area'] * params['scaling'] ** 2
        params['average_cell_area'] = params['average_cell_area'] * params['scaling'] ** 2
        params['connected_cell_area'] = params['connected_cell_area'] * params['scaling'] ** 2

        with st.spinner("Running Batch..."):
            for uploaded_file in uploaded_files:
                # Convert the uploaded file to an OpenCV image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                height, width = image.shape[:2]
                overall_area = (height * width) / params['scaling'] ** 2

                # Run cell detection algorithm
                normalized, morphed, mask, overlay, count, total_area, threshold_area, avg_area, fluorescence_intensities = masterFunction.cell_detection(
                    image, **params
                )

                # Store metrics for each image
                metrics.append({
                    "Filename": uploaded_file.name,
                    "Total Cell Count": count,
                    "Total Area of Picture (µm²)": round(overall_area),
                    "Total Cell Area (by contours) (µm²)": total_area,
                    "Total Cell Area (by threshold) (µm²)": threshold_area,
                    "Average Cell Area (µm²)": round(total_area / count, 2) if count > 0 else 0,
                })

                # Store images for each processing step
                images_dict[f"{uploaded_file.name}"] = image
                overlays_dict[f"{uploaded_file.name}"] = overlay
                morph_dict[f"{uploaded_file.name} morphed.tif"] = morphed
                process_dict[f"{uploaded_file.name} process.tif"] = normalized

        # Display images and comparison
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
            "Remove Noise? (Median Filter)": params['noise'],
            "Remove Irregularities? (Open Transform)": params['opening'],
            "Hole Filling (Close Transform)": params['closing'],
            "Border Thinning (Erode Transform)": params['eroding'],
            "Border Thickening (Dilate Transform)": params['dilating'],
            "Noise Removal Iterations": params['open_iter'],
            "Hole Sealing Iterations": params['close_iter'],
            "Border Thinning Iterations": params['erode_iter'],
            "Border Thickening Iterations": params['dilate_iter'],
            "Hole Size": params['hole_size'],
            "Hole Threshold": params['hole_threshold'],
        }
        # Create a DataFrame from the settings dictionary
        settings_df = pd.DataFrame(list(settings.items()), columns=['Parameter', 'Value'])

        csv = df.to_csv(index=False).encode('utf-8')
        settings_csv = settings_df.to_csv(index=False).encode('utf-8')

        # Get the current date
        current_date = datetime.now()
        # Format the current date as "06June2024"
        formatted_date = current_date.strftime("%d%b%Y")
        formatted_time = current_date.strftime("%X")
        # Use the formatted date in a filename
        zip_filename = f"Batch {formatted_date}_{formatted_time}.zip"  # Example filename

        # Create a zip file containing all images organized into folders
        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, 'w') as zf:
            # Create folders
            zf.writestr('images/original/', '')  # Empty folder
            zf.writestr('images/overlay/', '')   # Empty folder
            zf.writestr('images/morphed/', '')   # Empty folder
            zf.writestr('images/processed/', '') # Empty folder

            for filename, image_data in images_dict.items():
                img_bytes = cv2.imencode('.tif', image_data)[1].tobytes()
                zf.writestr(f'images/original/{filename}', img_bytes)
            for filename, image_data in overlays_dict.items():
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                img_bytes = cv2.imencode('.tif', image_data)[1].tobytes()
                zf.writestr(f'images/overlay/{filename}', img_bytes)
            for filename, image_data in morph_dict.items():
                img_bytes = cv2.imencode('.tif', image_data)[1].tobytes()
                zf.writestr(f'images/morphed/{filename}', img_bytes)
            for filename, image_data in process_dict.items():
                img_bytes = cv2.imencode('.tif', image_data)[1].tobytes()
                zf.writestr(f'images/processed/{filename}', img_bytes)
            zf.writestr("metrics.csv", csv)
            zf.writestr("settings.csv", settings_csv)

        # Offer zip file for download
        st.download_button(label="Download Data", data=zip_bytes.getvalue(), file_name=zip_filename, mime='application/zip')
