import streamlit as st
import pandas as pd
import cv2


# Defines a function to get default parameters and handle parameter uploads
def parameters():
    # Define mapping for image methods (used for the file upload portion)
    image_method_mapping = {
        'Sobel': 0,
        'Canny': 1,
        'Laplace': 2,
        'Block Segmentation': 3,
        'Histogram Equalization': 4
    }

    # Define default values for parameters
    params = pd.Series({
        'minimum_area': 800,
        'average_cell_area': 1000,
        'connected_cell_area': 2000,
        'lower_intensity': 100,
        'upper_intensity': 255,
        'block_size': 100,
        'scaling': 0.59,
        'fluorescence': False,
        'morph_checkbox': True,
        'kernel_size': 3,
        'noise': True,
        'opening': True,
        'closing': True,
        'eroding': False,
        'dilating': False,
        'open_iter': 1,
        'close_iter': 1,
        'erode_iter': 1,
        'dilate_iter': 1,
        'hole_size': 800,
        'hole_threshold': 200,
        'image_method': 0,
        'fluorescence_scoring': False,
    })

    # Option to upload parameters
    with st.expander("Upload Settings", expanded=False):
        settings_upload = st.file_uploader("Upload Previous Settings Here", type="csv")
        expanded_bool = True

    # If a settings file is uploaded, read it and update parameters
    if settings_upload is not None:
        try:
            expanded_bool = False
            settings_df = pd.read_csv(settings_upload, index_col="Parameter")
            params['minimum_area'] = int(float(settings_df.loc["Minimum Area", "Value"]))
            params['average_cell_area'] = int(float(settings_df.loc["Average Cell Area", "Value"]))
            params['connected_cell_area'] = int(float(settings_df.loc["Connected Cell Area", "Value"]))
            params['lower_intensity'] = int(settings_df.loc["Lower Intensity", "Value"])
            params['upper_intensity'] = int(settings_df.loc["Upper Intensity", "Value"])
            params['block_size'] = int(settings_df.loc["Block Size", "Value"])
            params['scaling'] = float(settings_df.loc["Scaling", "Value"])
            params['fluorescence'] = csv_bool_error(settings_df.loc["Fluorescence", "Value"])
            params['image_method'] = image_method_mapping.get(settings_df.loc["Image Method", "Value"], 0)
            params['morph_checkbox'] = csv_bool_error(settings_df.loc["Morphological Transformations", "Value"])
            params['kernel_size'] = int(settings_df.loc["Kernel Size", "Value"])
            params['noise'] = csv_bool_error(settings_df.loc["Remove Noise? (Median Filter)", "Value"])
            params['opening'] = csv_bool_error(settings_df.loc["Remove Irregularities? (Open Transform)", "Value"])
            params['closing'] = csv_bool_error(settings_df.loc["Hole Filling (Close Transform)", "Value"])
            params['eroding'] = csv_bool_error(settings_df.loc["Border Thinning (Erode Transform)", "Value"])
            params['dilating'] = csv_bool_error(settings_df.loc["Border Thickening (Dilate Transform)", "Value"])
            params['open_iter'] = int(settings_df.loc["Noise Removal Iterations", "Value"])
            params['close_iter'] = int(settings_df.loc["Hole Sealing Iterations", "Value"])
            params['erode_iter'] = int(settings_df.loc["Border Thinning Iterations", "Value"])
            params['dilate_iter'] = int(settings_df.loc["Border Thickening Iterations", "Value"])
            params['hole_size'] = int(float(settings_df.loc["Hole Size", "Value"]))
            params['hole_threshold'] = int(settings_df.loc["Hole Threshold", "Value"])
            params['fluorescence_scoring'] = csv_bool_error(settings_df.loc["Score Fluorescence by Intensity?", "Value"])

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    # Create a form for manual user input of parameters
    with st.expander("Parameters", expanded=expanded_bool):

        with st.container():
            st.subheader("Cell Characteristics")
            columns = st.columns(2)
            with columns[0]:
                # Input for minimum cell area
                params['minimum_area'] = st.number_input("Minimum Area to Be Consider Cell (µm\u00b2)", value=params['minimum_area'])
                # Input for average cell area
                params['average_cell_area'] = st.number_input("Average Size of Single Cell (µm\u00b2)", value=params['average_cell_area'])

            with columns[1]:
                # Input for maximum connected cell area
                params['connected_cell_area'] = st.number_input("Max Size of Cell (µm\u00b2) (for cases where cells are stuck together)", value=params['connected_cell_area'])
                # Slider for intensity thresholds
                params['lower_intensity'], params['upper_intensity'] = st.select_slider("Intensity Thresholds", options=list(range(256)), value=(params['lower_intensity'], params['upper_intensity']))
                # Input for scaling factor
                params['scaling'] = st.number_input("Scaling Between Pixels and Micrometers", value=params['scaling'])

            st.divider()

        st.subheader("Image Processing")
        cl = st.columns(2)
        with cl[0]:
            # Select box for image processing method
            params['image_method'] = st.selectbox("Processing Method (I recommend using Sobel, Canny, or Laplace)",
                                                  ("Sobel", "Canny", "Laplace", "Prewitt", "Roberts Cross", "Scharr", "Frei Chen", "Block Segmentation", "Histogram (Not Working)", "None"), index=params['image_method'])

            col = st.columns(3)
            with col[0]:
                # Checkbox for applying morphological transformations
                params['morph_checkbox'] = st.checkbox("Apply Morphological Transformations?", value=params['morph_checkbox'])
            with col[1]:
                # Checkbox for using fluorescence
                params['fluorescence'] = st.checkbox("Use Fluorescence?", value=params['fluorescence'])
            with col[2]:
                # Checkbox for ignoring holes
                hole_checkbox = st.checkbox("Ignore Holes?", value=False)
            if not hole_checkbox:
                # Slider for minimum hole size
                params['hole_size'] = st.slider("Minimum hole size for subtraction (µm\u00b2)", min_value=1, max_value=8000, value=params['minimum_area'], step=2)
                # Slider for hole brightness threshold
                params['hole_threshold'] = st.slider("Brightness threshold for holes", min_value=0, max_value=255, value=params['hole_threshold'])
            else:
                params['hole_size'] = 1e9

            # Additional input for block segmentation method
            if params['image_method'] == "Block Segmentation":
                params['block_size'] = st.slider("Block Size", min_value=50, max_value=200, value=params['block_size'], step=10)

            # Additional input for morphological transformations
            if params['morph_checkbox']:
                params['kernel_size'] = st.slider("Kernel Size (must be odd number)", min_value=1, max_value=11, value=params['kernel_size'], step=2)
            if params['fluorescence']:
                params['fluorescence_scoring'] = st.checkbox("Score Fluorescence by Intensity?", value=params['fluorescence_scoring'])
        with cl[1]:
            if params['morph_checkbox']:

                cl = st.columns(3)
                with cl[0]:
                    # Checkbox for noise removal
                    params['noise'] = st.checkbox("Remove Noise? (Median Filter)", value=params['noise'])
                with cl[1]:
                    # Checkbox for irregularities removal
                    params['opening'] = st.checkbox("Remove Irregularities? (Open Transform)", value=params['opening'])
                with cl[2]:
                    # Checkbox for hole filling
                    params['closing'] = st.checkbox("Fill In Holes? (Close Transform)", value=params['closing'])
                cl2 = st.columns(3)
                with cl2[0]:
                    # Checkbox for border thinning
                    params['eroding'] = st.checkbox("Thin Borders? (Erode Transform)", value=params['eroding'])
                with cl2[1]:
                    # Checkbox for border thickening
                    params['dilating'] = st.checkbox("Thicken Borders? (Dilate Transform)", value=params['dilating'])

                if params['opening']:
                    # Slider for noise removal iterations
                    params['open_iter'] = st.slider("Noise Removal Iterations", min_value=1, max_value=10, value=params['open_iter'])
                if params['closing']:
                    # Slider for hole sealing iterations
                    params['close_iter'] = st.slider("Hole Sealing Iterations", min_value=1, max_value=10, value=params['close_iter'])
                if params['eroding']:
                    # Slider for border thinning iterations
                    params['erode_iter'] = st.slider("Border Thinning Iterations", min_value=1, max_value=10, value=params['erode_iter'])
                if params['dilating']:
                    # Slider for border thickening iterations
                    params['dilate_iter'] = st.slider("Border Thickening Iterations", min_value=1, max_value=10, value=params['dilate_iter'])

    return params

# Helper function to handle boolean values from CSV
def csv_bool_error(value):
    bool_value = value.lower() == "true"
    return bool_value
