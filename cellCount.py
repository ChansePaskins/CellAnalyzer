import cv2
import numpy as np
import math
from image_normalization import *


def cell_counter(image, lower_intensity, upper_intensity, shadow_toggle,
        block_size, morph_filter, minimum_area, average_cell_area, connected_cell_area, scaling):
    original = image.copy()

    # check for color image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV image into its channels (splits into color channels for processing)
    h, s, v = cv2.split(hsv)

    # Display each channel separately (for testing)
    """cv2.namedWindow('Hue', cv2.WINDOW_NORMAL)
    cv2.imshow('Hue', h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.namedWindow('Saturation', cv2.WINDOW_NORMAL)
    cv2.imshow('Saturation', s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.namedWindow('Value', cv2.WINDOW_NORMAL)
    cv2.imshow('Value', v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    # checks saturation level. If there is a distinct amount of saturation (from a color image or GFP), it will use HSV
    if np.mean(s) >= 255:
        print("Colored Picture Detected: Splitting into HSV format\n Notice, this method hasn't been implemented yet")
        color_picture(image, original)

    # for grayscale images
    else:
        normalized, morphed, overlayed, cells, total_area, avg_area = grayscale_picture(original, lower_intensity, upper_intensity, shadow_toggle,
        block_size, morph_filter, minimum_area, average_cell_area, connected_cell_area, scaling)
        return normalized, morphed, overlayed, cells, total_area, avg_area
def color_picture(hsv, original):

    hsv_lower = np.array([156, 60, 0])
    hsv_upper = np.array([179, 115, 255])
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    minimum_area = 200
    average_cell_area = 650
    connected_cell_area = 1000
    cells = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > minimum_area:
            cv2.drawContours(original, [c], -1, (36,255,12), 2)
            if area > connected_cell_area:
                cells += math.ceil(area / average_cell_area)
            else:
                cells += 1
    print('Cells: {}'.format(cells))

    cv2.imshow('close', close)
    cv2.imshow('original', original)
    cv2.waitKey()


def grayscale_picture(original, lower_intensity, upper_intensity, shadow_toggle,
        block_size, morph_filter, minimum_area, average_cell_area, connected_cell_area, scaling):

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

    mask = cv2.inRange(normalized, lower_intensity, upper_intensity)

    # Morphological operations
    if morph_filter:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        morphed = close.copy()
    else:
        morphed = close = normalized

    # Find contours with hierarchy
    cnts, hierarchy = cv2.findContours(close, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

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

    return normalized, morphed, original, cells, converted_area_total, converted_area_mean


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
