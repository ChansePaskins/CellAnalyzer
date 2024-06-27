import streamlit as st
import pandas as pd
import cv2


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
        'lower_intensity': 0,
        'upper_intensity': 100,
        'block_size': 100,
        'scaling': 0.59,
        'fluorescence': False,
        'morph_checkbox': True,
        'kernel_size': 3,
        'opening': True,
        'closing': True,
        'eroding': False,
        'dilating': False,
        'open_iter': 1,
        'close_iter': 1,
        'erode_iter': 1,
        'dilate_iter': 1,
        'hole_size': 800,
        'image_method': 0,
    })

    # Option to upload parameters
    with st.expander("Upload Settings", expanded=False):
        settings_upload = st.file_uploader("Upload Previous Settings Here", type="csv")
        expanded_bool = True

    if settings_upload is not None:
        try:
            expanded_bool = False
            settings_df = pd.read_csv(settings_upload, index_col="Parameter")
            params['minimum_area'] = int(settings_df.loc["Minimum Area", "Value"])
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
            params['opening'] = csv_bool_error(settings_df.loc["Noise Removal (Open Transform)", "Value"])
            params['closing'] = csv_bool_error(settings_df.loc["Hole Filling (Close Transform)", "Value"])
            params['eroding'] = csv_bool_error(settings_df.loc["Border Thinning (Erode Transform)", "Value"])
            params['dilating'] = csv_bool_error(settings_df.loc["Border Thickening (Dilate Transform)", "Value"])
            params['open_iter'] = int(settings_df.loc["Noise Removal Iterations", "Value"])
            params['close_iter'] = int(settings_df.loc["Hole Sealing Iterations", "Value"])
            params['erode_iter'] = int(settings_df.loc["Border Thinning Iterations", "Value"])
            params['dilate_iter'] = int(settings_df.loc["Border Thickening Iterations", "Value"])
            params['hole_size'] = int(float(settings_df.loc["Hole Size", "Value"]))


        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    # Create a form for manual user input of parameters
    with st.expander("Parameters", expanded=expanded_bool):

        with st.container():
            st.subheader("Cell Characteristics")
            columns = st.columns(2)
            with columns[0]:
                params['minimum_area'] = st.number_input("Minimum Area to Be Consider Cell (µm\u00b2)",
                                                         value=params['minimum_area'])
                params['average_cell_area'] = st.number_input("Average Size of Single Cell (µm\u00b2)",
                                                              value=params['average_cell_area'])

            with columns[1]:
                params['connected_cell_area'] = st.number_input(
                    "Max Size of Cell (µm\u00b2) (for cases where cells are stuck together)",
                    value=params['connected_cell_area'])
                params['lower_intensity'], params['upper_intensity'] = st.select_slider(
                    "Intensity Thresholds", options=list(range(101)),
                    value=(params['lower_intensity'], params['upper_intensity']))
                params['scaling'] = st.number_input("Scaling Between Pixels and Micrometers",
                                                    value=params['scaling'])

            st.divider()

        st.subheader("Image Processing")
        cl = st.columns(2)
        with cl[0]:
            params['image_method'] = st.selectbox("Processing Method (I recommend using Sobel, Canny, or Laplace)",
                                                  ("Sobel", "Canny", "Laplace", "Block Segmentation",
                                                   "Histogram (Not Working)"), index=params['image_method'])

            col = st.columns(3)
            with col[0]:
                params['morph_checkbox'] = st.checkbox("Apply Morphological Transformations?",
                                                       value=params['morph_checkbox'])
            with col[1]:
                params['fluorescence'] = st.checkbox("Use Fluorescence?", value=params['fluorescence'])
            with col[2]:
                hole_checkbox = st.checkbox("Ignore Holes?", value=False)
            if not hole_checkbox:
                params['hole_size'] = st.slider("Minimum hole size for subtraction (µm\u00b2)", min_value=1,
                                                max_value=8000, value=params['minimum_area'], step=2)
            else:
                params['hole_size'] = 1e9

            if params['image_method'] == "Block Segmentation":
                params['block_size'] = st.slider("Block Size", min_value=50, max_value=200,
                                                 value=params['block_size'], step=10)

        with cl[1]:
            if params['morph_checkbox']:
                params['kernel_size'] = st.slider("Kernel Size (must be odd number)", min_value=1, max_value=11,
                                                  value=params['kernel_size'], step=2)
                cl = st.columns(4)
                with cl[0]:
                    params['opening'] = st.checkbox("Remove Noise? (Open Transform)",
                                                    value=params['opening'])
                with cl[1]:
                    params['closing'] = st.checkbox("Fill In Holes? (Close Transform)",
                                                    value=params['closing'])
                with cl[2]:
                    params['eroding'] = st.checkbox("Thin Borders? (Erode Transform)",
                                                    value=params['eroding'])
                with cl[3]:
                    params['dilating'] = st.checkbox("Thicken Borders? (Dilate Transform)",
                                                     value=params['dilating'])

                if params['opening']:
                    params['open_iter'] = st.slider("Noise Removal Iterations", min_value=1, max_value=10,
                                                    value=params['open_iter'])
                if params['closing']:
                    params['close_iter'] = st.slider("Hole Sealing Iterations", min_value=1, max_value=10,
                                                     value=params['close_iter'])
                if params['eroding']:
                    params['erode_iter'] = st.slider("Border Thinning Iterations", min_value=1, max_value=10,
                                                     value=params['erode_iter'])
                if params['dilating']:
                    params['dilate_iter'] = st.slider("Border Thickening Iterations", min_value=1, max_value=10,
                                                      value=params['dilate_iter'])

    return params


def csv_bool_error(value):
    bool_value = value.lower() == "true"
    return bool_value