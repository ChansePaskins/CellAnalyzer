import cv2
import numpy as np
import math
from image_processing import *


# Master function
def cell_detection(image, **kwargs):

    lower_intensity = kwargs['lower_intensity']
    upper_intensity = kwargs['upper_intensity']
    fluorescence = kwargs['fluorescence']
    image_method = kwargs['image_method']
    block_size = kwargs['block_size']
    hole_size = kwargs['hole_size']
    morph_filter = kwargs['morph_checkbox']
    minimum_area = kwargs['minimum_area']
    average_cell_area = kwargs['average_cell_area']
    connected_cell_area = kwargs['connected_cell_area']
    scaling = kwargs['scaling']
    kernel_size = kwargs['kernel_size']
    opening = kwargs['opening']
    closing = kwargs['closing']
    erosion = kwargs['eroding']
    dilation = kwargs['dilating']
    iter1 = kwargs['open_iter']
    iter2 = kwargs['close_iter']
    iter3 = kwargs['erode_iter']
    iter4 = kwargs['dilate_iter']

    # copies image to prevent from overwriting
    original = image.copy()
    # fluorescent images invert normal coloring. In brightfield, cells are dark. Under fluorescence, cells are bright
    if fluorescence:
        # converts to grayscale and inverts image
        original = cv2.bitwise_not(xyz_channel(original))
    # Uses Sobel edge detection algorithm to better define cell edges
    if image_method == "Sobel":
        processed = cv2.bitwise_not(apply_sobel_filter(original))

    # Uses Canny edge detection algorithm to better define cell edges.
    # Very similar to Sobel, but includes Non-maxima suppression and more intricate edge tracking
    elif image_method == 'Canny':
        processed = cv2.bitwise_not(apply_canny_filter(original))

    # Similar to both Sobel and Canny
    elif image_method == 'Laplace':
        processed = cv2.bitwise_not(apply_laplace_filter(original))

    # For cases of extreme shadowing. This breaks the image into several blocks and processed brightness. Old method.
    elif image_method == "Block Segmentation":
        # fixes shadowing and applies histogram
        processed = shadow_correction(original, block_size)

    # Skips block shadow correction and just applies histogram. Old method
    elif image_method == "Histogram":
        # perform histogram eq and corner brightness adjustment (not great)
        processed = histogram_equalization(original)

    # Skips all processing methods
    else:
        processed = original.copy()

    # This creates a new image based on the lower/upper thresholds defined by the user
    # This is pretty much just defining how bright/dark the cells are
    mask = cv2.inRange(processed.copy(), lower_intensity, upper_intensity)

    # Applies morphological functions based on user input
    if morph_filter:
        morphed = morphological_effects(
            mask.copy(), opening, closing, erosion, dilation, iter1, iter2, iter3, iter4, kernel_size
        )
    else:
        morphed = mask.copy()

    threshold_area = cv2.countNonZero(morphed)

    # Finds contours as well as parent/child associations
    # The goal is to determine the difference between clumps of cells and the holes within them.
    # Think of it like a donut. We want to subtract the area of a donut hole from the parent donut.
    # In practice, this is more trivial. Holes appear in red when found.
    cnts, hierarchy = cv2.findContours(morphed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Initializes variables for overlaying contours and counting cells/cell areas
    cells = 0
    cell_areas = []
    overlay = image.copy()
    color = (36, 255, 12) if not fluorescence else (0, 100, 255)

    # Process contours
    for i, c in enumerate(cnts):
        # Only consider parent contours (outer contours)
        if hierarchy[0][i][3] == -1:  # No parent, it's an outer contour
            area = cv2.contourArea(c)

            # Find the area of the holes and subtract from the outer contour's area
            holes_area = 0
            holes = []
            k = hierarchy[0][i][2]
            while k != -1:
                hole_area = cv2.contourArea(cnts[k])
                if hole_area > hole_size:
                    holes_area += hole_area
                    holes.append(cnts[k])
                k = hierarchy[0][k][0]

            # Subtract the total hole area from the outer contour area
            area = area - holes_area

            # Only counts the contour if it is bigger than the minimum area described
            if area > minimum_area:
                cv2.drawContours(overlay, [c], -1, color, 2)  # Draw outer contour
                # holes aren't working
                for hole in holes:
                    cv2.drawContours(overlay, [hole], -1, (255, 0, 0), 2)  # Draw holes

                # If area is determined to be larger than a single cell, calculate the number of cells
                if area > connected_cell_area:
                    num_cells = math.ceil(area / average_cell_area)
                    cells += num_cells
                    for _ in range(num_cells):
                        cell_areas.append(average_cell_area)
                else:
                    cells += 1
                    cell_areas.append(area)

    # Converts the pixel count into a real-world number
    converted_area_total = int(sum(cell_areas) / scaling ** 2)
    converted_area_mean = round(np.mean(cell_areas) / scaling ** 2, 2) if cell_areas else 0
    converted_threshold_area = int(threshold_area / scaling ** 2)

    return processed, morphed, mask, overlay, cells, converted_area_total, converted_threshold_area, converted_area_mean

