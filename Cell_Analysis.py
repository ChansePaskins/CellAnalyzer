import streamlit as st
import cv2
import numpy as np
import masterFunction
from streamlit_image_comparison import image_comparison
import pandas as pd
from parameters import parameters
from imageManipulation import plot_fluorescent_histogram

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer")
st.write("Upload an image and select parameters")

# File uploader to allow users to upload image files
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif"])
st.divider()

# Load and initialize parameters
params = parameters()

# Process the uploaded image if available
if uploaded_file is not None:
    # Read the uploaded image file as a byte array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Scale parameters based on the scaling factor
    params['minimum_area'] = params['minimum_area'] * params['scaling'] ** 2
    params['average_cell_area'] = params['average_cell_area'] * params['scaling'] ** 2
    params['connected_cell_area'] = params['connected_cell_area'] * params['scaling'] ** 2

    # Calculate image dimensions and overall area
    height, width = image.shape[:2]
    overall_area = (height * width) / params['scaling'] ** 2

    # Perform cell detection and analysis
    normalized, morphed, mask, overlay, count, total_area, threshold_area, avg_area, fluorescence_intensities = masterFunction.cell_detection(
        image, **params)

    # Display metrics related to cell counting and areas
    st.divider()
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Cell Count", count)
    with cols[1]:
        st.metric("Total Cell Area (by contours)", f"{total_area} µm\u00b2")
    with cols[2]:
        st.metric("Total Cell Area (by threshold)", f"{threshold_area} µm\u00b2")
    with cols[3]:
        st.metric("Percent Area of Image (by contours)", f"{round(100 * (total_area / overall_area), 2)}%")
    with cols[4]:
        st.metric("Percent Area of Image (by threshold)", f"{round(100 * (threshold_area / overall_area), 2)}%")

    # Display images and visual references
    st.divider()
    cls = st.columns(3)
    with cls[0]:
        st.image(image, caption='Original Image', use_column_width=True)
        st.image(mask, caption="Masked Image (Using intensity thresholds defined above)", use_column_width=True)
    with cls[1]:
        copy = image.copy()
        height, width = copy.shape[:2]

        # Display circles of minimum, average, and maximum cell sizes on the image
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

        # Display morphed image if the option is selected
        if params['morph_checkbox']:
            st.image(morphed, caption='Morphed Image (fills in holes and borders)', use_column_width=True)
        else:
            st.info("Morph Options Unselected")

    with cls[2]:
        st.image(normalized, caption='Processed Image (Image made from edge detection algorithm)',
                 use_column_width=True)
        st.image(overlay, caption='Overlayed Image (green is area counted, red (if there is any) are detected holes)',
                 use_column_width=True)

    # Image comparison slider
    st.divider()
    with st.container():
        st.write("Slide to compare")
        image_comparison(image, overlay)

    # Display fluorescence scoring if the option is selected
    if params['fluorescence_scoring']:
        st.divider()
        plt = plot_fluorescent_histogram(fluorescence_intensities)
        st.plotly_chart(plt)
        fl_cols = st.columns(2)
        with fl_cols[0]:
            st.metric("Average Intensity Across Contours", round(np.mean(fluorescence_intensities)), 2)
        with fl_cols[1]:
            st.metric("Intensity Standard Deviation Across Contours", round(np.std(fluorescence_intensities)), 2)

    # Store selected settings in a dictionary
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
    settings_csv = settings_df.to_csv(index=False).encode('utf-8')

    # Offer settings as a CSV file for download
    st.download_button(label="Save Settings", data=settings_csv, file_name="Settings.csv")
