
import argparse
import numpy as np
import math
from PIL import Image
from numba import cuda
kernel = np.array([[1, 4,6,4, 1], [4, 16, 24,16,4], [6, 24, 36,24,6],[4, 16, 24,16,4],[1, 4,6,4, 1]]) / 256
def compute_threads_and_blocks(imagetab,threadsperblock):
    width, height = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print("Thread blocks ", threadsperblock)
    print("Grid ", blockspergrid)
    return threadsperblock,blockspergrid



@cuda.jit
def RGBToBWKernel(source, destination):
    height = source.shape[1]
    width = source.shape[0]
    #offset =8 
    x,y = cuda.grid(2)
    if (x<width and y<height) :
        # ( (0.3 * R) + (0.59 * G) + (0.11 * B) )
        destination[x,y]=int(math.ceil(0.3*source[x,y,0]+0.59*source[x,y,1]+0.11*source[x,y,2]))

@cuda.jit
def gaussian_kernel(input_image, output_image):
    row, col = cuda.grid(2)
    #Process one pixel
    if row < input_image.shape[0] and col < input_image.shape[1]:
        # Apply Gaussian blur
        half_kernel = kernel.shape[0] // 2
        new_pixel_value = 0.0
        for i in range(-half_kernel, half_kernel + 1):
            for j in range(-half_kernel, half_kernel + 1):
                if row+i>input_image.shape[0] - 1 or row+i<0 :
                    y=row

                else:
                    y = row + i
                if col+j>input_image.shape[1] - 1 or col+j<0:
                    x=col
                else:
                    x =  col + j
                w = kernel[i + half_kernel, j + half_kernel]
                new_pixel_value += w * input_image[y, x,0]
        output_image[row, col,0] = new_pixel_value
        output_image[row, col,1] = new_pixel_value
        output_image[row, col,2] = new_pixel_value


@cuda.jit
def sobel_kernel(input_image, magnitude, angle):
    row, col = cuda.grid(2)
    
    if row < input_image.shape[0] and col < input_image.shape[1]:
        sobelx = input_image[max(0, row - 1), min(input_image.shape[1] - 1, col + 1),0] - \
                 input_image[max(0, row - 1), max(0, col - 1),0] + \
                 2 * input_image[row, min(input_image.shape[1] - 1, col + 1),0] - \
                 2 * input_image[row, max(0, col - 1),0] + \
                 input_image[min(input_image.shape[0] - 1, row + 1), min(input_image.shape[1] - 1, col + 1),0] - \
                 input_image[min(input_image.shape[0] - 1, row + 1), max(0, col - 1),0]
                 
        sobely = input_image[max(0, row - 1), min(input_image.shape[1] - 1, col + 1),0] + \
                 2 * input_image[min(input_image.shape[0] - 1, row + 1), min(input_image.shape[1] - 1, col + 1),0] + \
                 input_image[min(input_image.shape[0] - 1, row + 1), min(input_image.shape[1] - 1, col + 1),0] - \
                 (input_image[max(0, row - 1), max(0, col - 1),0] + \
                  2 * input_image[min(input_image.shape[0] - 1, row + 1), max(0, col - 1),0] + \
                  input_image[min(input_image.shape[0] - 1, row + 1), max(0, col - 1),0])

        magnitude[row, col,0] =math.sqrt(sobelx ** 2 + sobely ** 2)
        magnitude[row, col,1] = math.sqrt(sobelx ** 2 + sobely ** 2)
        magnitude[row, col,2] = math.sqrt(sobelx ** 2 + sobely ** 2)
        angle[row, col,0] =math.atan2(sobely, sobelx)
        angle[row, col,1] =math.atan2(sobely, sobelx)
        angle[row, col,2] =math.atan2(sobely, sobelx)


@cuda.jit
def threshold_kernel(magnitude, thresholded):
    threshold_low = 51  
    threshold_high = 102
    row, col = cuda.grid(2) 
    if row < magnitude.shape[0] and col < magnitude.shape[1]:
        if magnitude[row, col,0] > threshold_high:
            thresholded[row, col,0] = 255
            thresholded[row, col,1] = 255
            thresholded[row, col,2] = 255
        elif magnitude[row, col,0] < threshold_low:
            thresholded[row, col,0] = 0
            thresholded[row, col,1] = 0
            thresholded[row, col,2] = 0
        else:
            thresholded[row, col,0] = 128
            thresholded[row, col,1] = 128
            thresholded[row, col,2] = 128

@cuda.jit
def hysterisis_kernel(thresholded, output_image):

    row, col = cuda.grid(2)
    
    if row < thresholded.shape[0] and col < thresholded.shape[1]:
        if thresholded[row, col,0] == 255:
            output_image[row, col,0] = 255
            output_image[row, col,1] = 255
            output_image[row, col,2] = 255
        else:
            output_image[row, col,0] = 0
            output_image[row, col,1] = 0
            output_image[row, col,2] = 0




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
    threads_per_block = (16, 16)   
    image = Image.open(args.inputImage)
    imagetab = np.array(image)
    result=np.zeros_like(imagetab, dtype=np.uint8) 
    if args.tb:
        threads_per_block = (args.tb,args.tb)
    threads_per_block,blocks_per_grid=compute_threads_and_blocks(imagetab,threads_per_block)
    if args.bw:
        RGBToBWKernel[blocks_per_grid, threads_per_block](imagetab, result)
    elif args.gauss:
        output_bw = np.zeros_like(imagetab)
        RGBToBWKernel[blocks_per_grid, threads_per_block](imagetab, output_bw)
        gaussian_kernel[blocks_per_grid, threads_per_block](output_bw, result)
    elif args.sobel:
        output_bw = np.zeros_like(imagetab)
        RGBToBWKernel[blocks_per_grid, threads_per_block](imagetab, output_bw)
        output_gaussian = np.zeros_like(imagetab)
        gaussian_kernel[blocks_per_grid, threads_per_block](output_bw, output_gaussian)
        angle = np.zeros_like(imagetab, dtype=np.float32)
        sobel_kernel[blocks_per_grid, threads_per_block](output_gaussian, result, angle)
    elif args.threshold:
        output_bw = np.zeros_like(imagetab)
        RGBToBWKernel[blocks_per_grid, threads_per_block](imagetab, output_bw)
        output_gaussian = np.zeros_like(imagetab)
        gaussian_kernel[blocks_per_grid, threads_per_block](output_bw, output_gaussian)
        magnitude = np.zeros_like(imagetab, dtype=np.float32)
        angle = np.zeros_like(imagetab, dtype=np.float32)
        sobel_kernel[blocks_per_grid, threads_per_block](output_gaussian, magnitude, angle)
        threshold_kernel[blocks_per_grid, threads_per_block](magnitude, result)
    else:
        output_bw = np.zeros_like(imagetab)
        RGBToBWKernel[blocks_per_grid, threads_per_block](imagetab, output_bw)
        output_gaussian = np.zeros_like(imagetab)
        gaussian_kernel[blocks_per_grid, threads_per_block](output_bw, output_gaussian)
        magnitude = np.zeros_like(imagetab, dtype=np.float32)
        angle = np.zeros_like(imagetab, dtype=np.float32)
        sobel_kernel[blocks_per_grid, threads_per_block](output_gaussian, magnitude, angle)
        thresholded = np.zeros_like(imagetab)
        threshold_kernel[blocks_per_grid, threads_per_block](magnitude, thresholded)
        hysterisis_kernel[blocks_per_grid, threads_per_block](thresholded, result)
    output_image_pillow = Image.fromarray(result)
    output_image_pillow.save(args.outputImage)
    print("L'image a été enregistrée avec succès sous",args.outputImage)


if __name__ == '__main__':
    main()
