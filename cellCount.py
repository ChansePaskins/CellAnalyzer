import cv2
import numpy as np
import math
from image_normalization import *


def cell_detection(image, lower_intensity, upper_intensity, shadow_toggle,
                   block_size, morph_filter, minimum_area, average_cell_area,
                   connected_cell_area, scaling, kernel_size, opening, closing,
                   erosion, dilation, iter1, iter2, iter3, iter4):

    original = image.copy()
    if shadow_toggle == "Block Segmentation":
        # shadowing block correction, histogram eq (recommended)
        normalized = shadow_correction(original, block_size)
    elif shadow_toggle == "Histogram":
        # perform histogram eq and corner brightness adjustment (not great)
        normalized = histogram_equalization(original)
    elif shadow_toggle == "Sobel":
        normalized = cv2.bitwise_not(apply_sobel_filter(original))
    elif shadow_toggle == 'Canny':
        normalized = cv2.bitwise_not(apply_canny_filter(original))
    elif shadow_toggle == 'Canny Channel':
        normalized = cv2.bitwise_not(apply_canny_filter_area(original))
    else:
        normalized = original.copy()

    mask = cv2.inRange(normalized.copy(), lower_intensity, upper_intensity)

    if morph_filter:
        morphed = morphological_effects(
            mask.copy(), opening, closing, erosion, dilation, iter1, iter2, iter3, iter4, kernel_size
        )
    else:
        morphed = mask.copy()

    # Find contours with hierarchy
    cnts, hierarchy = cv2.findContours(morphed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    cells = 0
    cell_areas = []

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
                if hole_area > connected_cell_area:
                    holes_area += cv2.contourArea(cnts[k])
                    holes.append(cnts[k])
                k = hierarchy[0][k][0]

                area = area - holes_area

            if area > minimum_area:
                cv2.drawContours(original, [c], -1, (36, 255, 12), 2)  # Draw outer contour in green
                for hole in holes:
                    cv2.drawContours(original, [hole], -1, (255, 0, 0), 2)  # Draw holes in red
                if area > connected_cell_area:
                    cells += math.ceil(area / average_cell_area)
                    for _ in range(cells):
                        cell_areas.append(average_cell_area)
                else:
                    cells += 1 if area > 0 else None
                    cell_areas.append(area)

    converted_area_total = int(sum(cell_areas) / scaling ** 2)
    converted_area_mean = round(np.mean(cell_areas) / scaling ** 2, 2) if cell_areas else 0

    return normalized, morphed, mask, original, cells, converted_area_total, converted_area_mean


if __name__ == "__main__":
    directory_in = r"C:\Users\chans\PycharmProjects\CMV\cellCounter\imageDump\Raw\Test0\b3.tif"
    image = cv2.imread(directory_in)

    minimum_area = 200
    average_cell_area = 400
    connected_cell_area = 1000

    # Define thresholds for intensity
    lower_intensity = 0
    upper_intensity = 60

    shadow_toggle = 2

    block_size = 100

    cell_counter(image, lower_intensity, upper_intensity, shadow_toggle,
        block_size, minimum_area, average_cell_area, connected_cell_area
    )
