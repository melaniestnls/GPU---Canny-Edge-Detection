import argparse  
import numpy as np  
import math  
from PIL import Image 
from numba import cuda 

# Fonction pour calculer les dimensions de la grille de threads
def compute_threads_and_blocks(imagetab, threads_per_block):

    width, height = imagetab.shape[:2]
    blockspergrid_x = int(math.ceil(width / threads_per_block[0]))
    blockspergrid_y = int(math.ceil(height / threads_per_block[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print("Thread blocks ", threads_per_block)
    print("Grid ", blockspergrid)
    return threads_per_block, blockspergrid

# Noyau CUDA pour convertir l'image en niveaux de gris
@cuda.jit
def RGBToBWKernel(input_image, output_image):
    # Récupérer la hauteur et la largeur de l'image d'entrée
    height = input_image.shape[1]
    width = input_image.shape[0]
    x, y = cuda.grid(2)
    # Vérifie si le thread est dans les limites de l'image
    if (x < width and y < height):
        # Conversion en niveaux de gris: (0.3 * R) + (0.59 * G) + (0.11 * B)
        output_image[x, y] = int(math.ceil(0.3 * input_image[x, y, 0] + 0.59 * input_image[x, y, 1] + 0.11 * input_image[x, y, 2]))

# Noyau CUDA pour appliquer un flou gaussien
@cuda.jit
def gaussian_kernel(input_image, output_image,kernel):
    #Obtienir les indices de thread en deux dimensions
    row, col = cuda.grid(2)
    #Vérifie si le thread est dans les limites de l'image
    if row < input_image.shape[0] and col < input_image.shape[1]:
        #Appliquer le flou gaussien
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
        output_image[row, col,0] = new_pixel_value/ 256
        output_image[row, col,1] = new_pixel_value/ 256
        output_image[row, col,2] = new_pixel_value/ 256

# Noyau CUDA pour calculer le gradient de Sobel
@cuda.jit
def sobel_kernel(input_image, magnitude, angle):

    row, col = cuda.grid(2)
    if row < input_image.shape[0] and col < input_image.shape[1]:
        if row == 0 or row == input_image.shape[0] - 1 or col == 0 or col == input_image.shape[1] - 1:
            magnitude[row, col, 0] = 0
            angle[row, col, 0] = 0
        else:
            # Convolution
            Ix = (
                input_image[row - 1, col - 1, 0] * (-1) + input_image[row - 1, col, 0] * 0 + input_image[row - 1, col + 1, 0] * 1 +
                input_image[row, col - 1, 0] * (-2) + input_image[row, col, 0] * 0 + input_image[row, col + 1, 0] * 2 +
                input_image[row + 1, col - 1, 0] * (-1) + input_image[row + 1, col, 0] * 0 + input_image[row + 1, col + 1, 0] * 1
            )
            Iy = (
                input_image[row - 1, col - 1, 0] * 1 + input_image[row - 1, col, 0] * 2 + input_image[row - 1, col + 1, 0] * 1 +
                input_image[row, col - 1, 0] * 0 + input_image[row, col, 0] * 0 + input_image[row, col + 1, 0] * 0 +
                input_image[row + 1, col - 1, 0] * (-1) + input_image[row + 1, col, 0] * (-2) + input_image[row + 1, col + 1, 0] * (-1)
            )
            # Clamp the sobel x and y value to 175
            Ix = min(175,Ix)
            Iy = min(175,Iy)
            # Gradient magnitude
            magnitude[row, col, 0] = math.sqrt(Ix ** 2 + Iy ** 2)
            # Gradient angle
            angle[row, col, 0] = math.atan2(Iy, Ix)


# Noyau CUDA pour appliquer un seuillage
@cuda.jit
def threshold_kernel(magnitude, thresholded, highThreshold, lowThreshold):

    M, N = magnitude.shape[:2]
    row, col = cuda.grid(2)
    if row < M and col < N:
        # Comparaison de la magnitude avec les seuils
        if magnitude[row, col, 0] > highThreshold:
            thresholded[row, col, 0] = 255
            thresholded[row, col, 1] = 255
            thresholded[row, col, 2] = 255
        elif magnitude[row, col, 0] < lowThreshold:
            thresholded[row, col, 0] = 0
            thresholded[row, col, 1] = 0
            thresholded[row, col, 2] = 0
        else:
            thresholded[row, col, 0] = 127
            thresholded[row, col, 1] = 127
            thresholded[row, col, 2] = 127

# Noyau CUDA pour appliquer le traitement d'hystérésis
@cuda.jit
def hysterisis_kernel(thresholded, result):

    M, N = thresholded.shape[:2]
    weak = 127  # Valeur du pixel faible
    strong = 255  # Valeur du pixel fort
    row, col = cuda.grid(2)
    if row < M and col < N:
        if thresholded[row, col, 0] == weak:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= row + i < M and 0 <= col + j < N:
                        if thresholded[row + i, col + j, 0] == strong:
                            result[row, col, 0] = strong
                            result[row, col, 1] = strong
                            result[row, col, 2] = strong
                            return
            result[row, col, 0] = 0
            result[row, col, 1] = 0
            result[row, col, 2] = 0
        else:
            result[row, col, 0] = thresholded[row, col, 0]
            result[row, col, 1] = thresholded[row, col, 1]
            result[row, col, 2] = thresholded[row, col, 2]

# Fonction principale pour traiter l'image
def process_image(imagetab, threads_per_block, blocks_per_grid, d_kernel, args):

    output_bw = cuda.to_device(np.zeros_like(imagetab))  # Création d'un tableau vide sur le GPU
    RGBToBWKernel[blocks_per_grid, threads_per_block](imagetab, output_bw)  # Appel du noyau CUDA
    cuda.synchronize()  # Attente de la fin de l'exécution sur le GPU

    if args.bw:
        return output_bw.copy_to_host()  # Renvoie l'image en niveaux de gris vers le CPU

    output_gaussian = cuda.to_device(np.zeros_like(imagetab))  
    gaussian_kernel[blocks_per_grid, threads_per_block](output_bw, output_gaussian, d_kernel)  
    cuda.synchronize()  

    if args.gauss:
        return output_gaussian.copy_to_host()  # Renvoie l'image après flou gaussien vers le CPU

    magnitude = cuda.to_device(np.zeros_like(imagetab, dtype=np.float32))  
    angle = cuda.to_device(np.zeros_like(imagetab, dtype=np.float32))  
    sobel_kernel[blocks_per_grid, threads_per_block](output_gaussian, magnitude, angle) 
    cuda.synchronize() 

    if args.sobel:
        return magnitude.copy_to_host(), angle.copy_to_host()  # Renvoie la magnitude et l'angle vers le CPU

    thresholded = cuda.to_device(np.zeros_like(imagetab))  
    highThreshold = 102  # Seuil supérieur pour le seuillage
    lowThreshold = 51  # Seuil inférieur pour le seuillage
    threshold_kernel[blocks_per_grid, threads_per_block](magnitude, thresholded, highThreshold, lowThreshold)  
    cuda.synchronize() 

    if args.threshold:
        return thresholded.copy_to_host()  # Renvoie l'image seuillée vers le CPU

    result = cuda.to_device(np.zeros_like(imagetab, dtype=np.uint8))  
    hysterisis_kernel[blocks_per_grid, threads_per_block](thresholded, result)  
    cuda.synchronize() 

    return result.copy_to_host()  # Renvoie l'image résultante vers le CPU

# Fonction principale pour exécuter le programme
def main():

    # Analyse des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Canny Edge Detection')
    parser.add_argument('inputImage', help='Input image file')
    parser.add_argument('outputImage', help='Output image file')
    parser.add_argument('--tb', type=int, help='Size of a thread block for all operations')
    parser.add_argument('--bw', action='store_true', help='Perform only the bw_kernel')
    parser.add_argument('--gauss', action='store_true', help='Perform the bw_kernel and the gauss_kernel')
    parser.add_argument('--sobel', action='store_true', help='Perform all kernels up to sobel_kernel and write to disk the magnitude of each pixel')
    parser.add_argument('--threshold', action='store_true', help='Perform all kernels up to threshold_kernel')
    kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])  # Noyau gaussien
    d_kernel = cuda.to_device(kernel)  # Copie du noyau gaussien vers le GPU
    args = parser.parse_args()  # Analyse des arguments en ligne de commande

    # Chargement de l'image et conversion en tableau numpy
    image = Image.open(args.inputImage)
    imagetab = np.array(image)
    d_imagetab = cuda.to_device(imagetab)  # Copie de l'image vers le GPU

    # Calcul des dimensions de la grille de threads
    threads_per_block = (16, 16)
    if args.tb:
        threads_per_block = (args.tb, args.tb)
    threads_per_block, blocks_per_grid = compute_threads_and_blocks(imagetab, threads_per_block)

    # Traitement de l'image en fonction des arguments spécifiés
    if args.sobel:
        magnitude, _ = process_image(d_imagetab, threads_per_block, blocks_per_grid, d_kernel, args)  # Appel de la fonction de traitement
        output_image_pillow = Image.fromarray(magnitude[:, :, 0].astype(np.uint8), mode='L')  # Conversion en mode 'L' (niveaux de gris)
        output_image_pillow.save(args.outputImage)  # Enregistrement de l'image 
        print("L'image de magnitude a été enregistrée avec succès sous", args.outputImage)
        return
    else:
        result = process_image(d_imagetab, threads_per_block, blocks_per_grid, d_kernel, args)  # Appel de la fonction de traitement

        # Sauvegarde de l'image 
        output_image_pillow = Image.fromarray(result)
        output_image_pillow.save(args.outputImage)
        print("L'image a été enregistrée avec succès sous", args.outputImage)

if __name__ == '__main__':
    main()  # Exécution de la fonction principale
