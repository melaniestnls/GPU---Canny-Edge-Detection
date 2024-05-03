import math
import numpy as np
from numba import cuda
from PIL import Image
@cuda.jit
def RGBToBWKernel(source, destination):
    height = source.shape[1]
    width = source.shape[0]
    #offset =8 
    x,y = cuda.grid(2)
    if (x<width and y<height) :
        # ( (0.3 * R) + (0.59 * G) + (0.11 * B) )
        destination[x,y]=np.uint8(0.3*source[x,y,0]+0.59*source[x,y,1]+0.11*source[x,y,2])


@cuda.jit
def bw_kernel(input_image, output_image):
    # Obtenez les coordonnées de thread
    row, col = cuda.grid(2)
    
    if row < input_image.shape[0] and col < input_image.shape[1]:
        # Convertir en niveaux de gris en moyennant les canaux RGB
        gray_value = (input_image[row, col, 0] + input_image[row, col, 1] + input_image[row, col, 2]) / 3
        output_image[row, col] = gray_value

@cuda.jit
def gaussian_kernel(input_image, output_image, kernel):
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
def threshold_kernel(magnitude, threshold_low, threshold_high, thresholded):

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


def compute_threads_and_blocks(imagetab,threadsperblock):
    width, height = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print("Thread blocks ", threadsperblock)
    print("Grid ", blockspergrid)
    return threadsperblock,blockspergrid


def main():
    image_path = "chat.jpeg"
    image = Image.open(image_path)
    imagetab = np.array(image)
    # Déterminez les tailles de grille et de bloc appropriées pour vos données
    threads_per_block = (16, 16)
    threads_per_block,blocks_per_grid=compute_threads_and_blocks(imagetab,threads_per_block)
    # Convertir en niveaux de gris
    output_bw = np.zeros_like(imagetab)
    RGBToBWKernel[blocks_per_grid, threads_per_block](imagetab, output_bw)

    # Appliquer un flou gaussien
    kernel = np.array([[1, 4,6,4, 1], [4, 16, 24,16,4], [6, 24, 36,24,6],[4, 16, 24,16,4],[1, 4,6,4, 1]]) / 256  # Kernel gaussien
    output_gaussian = np.zeros_like(imagetab)
    gaussian_kernel[blocks_per_grid, threads_per_block](output_bw, output_gaussian, kernel)
    output_gaussian_pillow=Image.fromarray(output_gaussian)
    output_gaussian_pillow.save("gaus.jpg")
    # Calculer la magnitude et l'angle de Sobel
    magnitude = np.zeros_like(imagetab, dtype=np.float32)
    angle = np.zeros_like(imagetab, dtype=np.float32)
    sobel_kernel[blocks_per_grid, threads_per_block](output_gaussian, magnitude, angle)

    # Appliquer le seuillage
    thresholded = np.zeros_like(imagetab)
    threshold_low = 51
    threshold_high = 102
    threshold_kernel[blocks_per_grid, threads_per_block](magnitude, threshold_low, threshold_high, thresholded)

    # Supprimer les bords faibles
    output_image = np.zeros_like(imagetab)
    hysterisis_kernel[blocks_per_grid, threads_per_block](thresholded, output_image)
    output_image_path = "sortie.jpg"  # Remplacez par votre chemin de destination
    output_image_pillow = Image.fromarray(output_image)


    output_image_pillow.save(output_image_path)
    print("L'image a été enregistrée avec succès sous", output_image_path)

if __name__ == "__main__":
    main()
