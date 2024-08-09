from edgeDetection import *
from contourDetection import *
from imageManipulation import *
from thresholding import *


def cell_detection(image, **kwargs):
    """
    Main function for cell detection using various image processing techniques.

    Args:
        image (numpy.ndarray): Input image.
        **kwargs: Additional parameters for processing.

    Returns:
        tuple: Processed images and detection results.
    """

    # Extract parameters from kwargs
    lower_intensity = kwargs['lower_intensity']
    upper_intensity = kwargs['upper_intensity']
    image_method = kwargs['image_method']
    block_size = kwargs['block_size']
    morph_filter = kwargs['morph_checkbox']
    scaling = kwargs['scaling']
    kernel_size = kwargs['kernel_size']
    noise = kwargs['noise']
    opening = kwargs['opening']
    closing = kwargs['closing']
    erosion = kwargs['eroding']
    dilation = kwargs['dilating']
    iter1 = kwargs['open_iter']
    iter2 = kwargs['close_iter']
    iter3 = kwargs['erode_iter']
    iter4 = kwargs['dilate_iter']

    # Copy the original image to prevent overwriting
    original = image.copy()

    # Convert image to grayscale if it is not already
    try:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = original

    # Selected edge detection method
    if image_method == "Sobel":
        processed = cv2.bitwise_not(sobel_filter(original))
    elif image_method == 'Canny':
        processed = cv2.bitwise_not(canny_filter(original))
    elif image_method == 'Laplace':
        processed = cv2.bitwise_not(laplace_filter(original))
    elif image_method == 'Prewitt':
        processed = cv2.bitwise_not(prewitt_filter(original))
    elif image_method == 'Roberts Cross':
        processed = cv2.bitwise_not(roberts_cross_filter(original))
    elif image_method == 'Scharr':
        processed = cv2.bitwise_not(scharr_filter(original))
    elif image_method == 'Frei Chen':
        processed = cv2.bitwise_not(frei_chen_filter(original))
    elif image_method == "Block Segmentation":
        processed = shadow_correction(original, block_size)
    elif image_method == "Histogram":
        processed = histogram_equalization(original)
    else:
        processed = cv2.bitwise_not(gray.copy())

    # Apply Otsu's or global thresholding to create a binary mask
    mask = global_threshold(processed, lower_intensity, upper_intensity)

    # Apply morphological operations if enabled
    if morph_filter:
        morphed = morphological_effects(mask.copy(), opening, closing, erosion, dilation, iter1, iter2, iter3, iter4, kernel_size)
    else:
        morphed = mask.copy()

    # Apply noise reduction if enabled
    if noise:
        morphed = median_filter(morphed)

    # Calculate the threshold area (number of non-zero pixels)
    threshold_area = cv2.countNonZero(morphed)


    # Detect contours and analyze cell areas and intensities
    overlay, cells, cell_areas, average_intensities = cv2_contours(original, gray, morphed, **kwargs)


    # Convert pixel area to real-world area using the scaling factor
    converted_area_total = int(sum(cell_areas) / scaling ** 2)
    converted_area_mean = round(np.mean(cell_areas) / scaling ** 2, 2) if cell_areas else 0
    converted_threshold_area = int(threshold_area / scaling ** 2)

    return processed, morphed, mask, overlay, cells, converted_area_total, converted_threshold_area, converted_area_mean, average_intensities
