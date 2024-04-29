import cv2
import argparse
import numpy as np

def bw_kernel(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_kernel(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def sobel_kernel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx)
    return magnitude, angle

def threshold_kernel(image, threshold_low, threshold_high):
    _, thresholded = cv2.threshold(image, threshold_low, threshold_high, cv2.THRESH_BINARY)
    return thresholded

def hysterisis_kernel(image):
    # Define high and low thresholds
    low_threshold = 50
    high_threshold = 150

    # Copy the input image
    output_image = np.copy(image)

    # Get the indices of strong edges
    strong_edges_i, strong_edges_j = np.where(image >= high_threshold)

    # Get the indices of weak edges
    weak_edges_i, weak_edges_j = np.where((image < high_threshold) & (image >= low_threshold))

    # Iterate over weak edges and check if they are connected to strong edges
    for i, j in zip(weak_edges_i, weak_edges_j):
        # Check 8-neighbors of each weak edge pixel
        if any(image[i+di, j+dj] >= high_threshold for di in [-1, 0, 1] for dj in [-1, 0, 1]):
            output_image[i, j] = 255
        else:
            output_image[i, j] = 0

    return output_image


def main():
    parser = argparse.ArgumentParser(description='Canny Edge Detection')
    parser.add_argument('inputImage', help='Input image file')
    parser.add_argument('outputImage', help='Output image file')
    parser.add_argument('--tb', type=int, help='Size of a thread block for all operations')
    parser.add_argument('--bw', action='store_true', help='Perform only the bw_kernel')
    parser.add_argument('--gauss', action='store_true', help='Perform the bw_kernel and the gauss_kernel')
    parser.add_argument('--sobel', action='store_true', help='Perform all kernels up to sobel_kernel and write to disk the magnitude of each pixel')
    parser.add_argument('--threshold', action='store_true', help='Perform all kernels up to threshold_kernel')

    args = parser.parse_args()

    image = cv2.imread(args.inputImage)

    if args.bw:
        result = bw_kernel(image)
    elif args.gauss:
        result = gaussian_kernel(bw_kernel(image))
    elif args.sobel:
        magnitude, _ = sobel_kernel(gaussian_kernel(bw_kernel(image)))
        cv2.imwrite(args.outputImage, magnitude)
        return
    elif args.threshold:
        magnitude, _ = sobel_kernel(gaussian_kernel(bw_kernel(image)))
        thresholded = threshold_kernel(magnitude, 50, 150)  # Adjust threshold values as needed
        result = hysterisis_kernel(thresholded)
    else:
        magnitude, _ = sobel_kernel(gaussian_kernel(bw_kernel(image)))
        thresholded = threshold_kernel(magnitude, 50, 150)  # Adjust threshold values as needed
        result = hysterisis_kernel(thresholded)

    cv2.imwrite(args.outputImage, result)

if __name__ == '__main__':
    main()
