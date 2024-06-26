import numpy as np
import cv2


def morphological_effects(image, opening, closing, erosion, dilation, iter1, iter2, iter3, iter4, kernel_size):

    # Create kernel based on user input (normal people call them filters, graphic designers call them masks)
    # Instagram filters use the same base operations, this is no different.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # To prevent from overwriting image
    result = image.copy()

    # Apply erosion
    if erosion:
        result = cv2.erode(result, kernel, iterations=iter3)

    # Apply dilation
    if dilation:
        result = cv2.dilate(result, kernel, iterations=iter4)

    # Apply opening
    if opening:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iter1)

    # Apply closing
    if closing:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iter2)

    return result


def apply_sobel_filter(image):

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Apply Sobel filter in the x direction
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, dx=1, dy=0, ksize=1)

    # Apply Sobel filter in the y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=1)

    # Compute the gradient magnitude
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize the result to fit in the range [0, 255]
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 type for displaying
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    # Convert to Grayscale
    if len(image.shape) > 2:
        gradient_magnitude = cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gradient_magnitude)

    return equalized


def apply_canny_filter(image):
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 0, 60)

    # Apply dilation to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    return dilated_edges

def apply_laplace_filter(src):

    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)

    # Apply Laplace function
    dst = cv2.Laplacian(src, cv2.CV_16S, ksize=5)

    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)

    return abs_dst

# Adjusts edge brightness based on average values
def histogram_equalization(image):

    height, width, _ = image.shape
    center_x, center_y = width / 2, height / 2

    # Calculate distances of each pixel from the image center
    y, x = np.ogrid[:height, :width]
    distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Calculate the maximum distance from the center
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    # Calculate brightness decrease factor based on distance
    brightness_decrease = distance_to_center / max_distance

    # Calculate the brightness change
    brightness_change = (brightness_decrease * 2).astype(np.uint8)

    # Split the image into channels
    b, g, r = cv2.split(image)

    # Add brightness_change to each channel separately
    b_adjusted = cv2.add(b, brightness_change)
    g_adjusted = cv2.add(g, brightness_change)
    r_adjusted = cv2.add(r, brightness_change)

    # Merge the adjusted channels back into an image
    image_with_brightness = cv2.merge((b_adjusted, g_adjusted, r_adjusted))

    # Convert to Grayscale
    image = cv2.cvtColor(image_with_brightness, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(image)

    return equalized


# Automatic shadow correction
def shadow_correction(image, block_size):
    # Divide the image into sections and adjust brightness of each section
    def adjust_brightness(image):
        height, width = image.shape[:2]

        # Calculate number of blocks in each dimension
        num_blocks_x = (width + block_size - 1) // block_size
        num_blocks_y = (height + block_size - 1) // block_size

        # Iterate over each block
        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                # Define block boundaries
                y_start = i * block_size
                y_end = min((i + 1) * block_size, height)
                x_start = j * block_size
                x_end = min((j + 1) * block_size, width)

                # Calculate average brightness of the block
                block = image[y_start:y_end, x_start:x_end]
                avg_brightness = np.mean(block)

                # Adjust brightness of the block
                image[y_start:y_end, x_start:x_end] = np.clip(block + (128 - avg_brightness), 0, 255)

        return image


    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image


    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray_image)

    # Adjust brightness of image sections
    corrected_image = adjust_brightness(equalized)

    return corrected_image


def xyz_channel(image):
    # This isn't quite working yet. Having issues with different channels. For now, we use only grayscale
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    h, L, S = cv2.split(image)

    #cv2.imshow('H', h)
    #cv2.imshow('L', L)
    #cv2.imshow('S', S)
    #cv2.waitKey()

    return L
