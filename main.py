import cv2
import matplotlib.pyplot as plt
import numpy as np

image=cv2.imread("image.jpg") #insert your "grayscale" image here
#img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(image)
plt.show()

image.shape

def add_noise(image, noise_factor=500):
    noisy_image = image+noise_factor*np.random.randn(*image.shape)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

noisy_image = add_noise(image)
plt.imshow(noisy_image)
plt.show()

def arithmetic_mean_filter(image, kernel_size=3):
    height, width,color = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    border = kernel_size//2
    
    for y in range(border, height - border):
        for x in range(border, width - border):
            window = image[y - border : y + border + 1, x - border : x + border + 1]
            avg = np.sum(window)
            filtered_image[y, x] = (avg/(kernel_size**2)).astype(image.dtype)
    return filtered_image

am = arithmetic_mean_filter(noisy_image)

def geometric_mean_filter(image, kernel_size=3):
    height, width,color = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    border = kernel_size//2
    
    for y in range(border, height - border):
        for x in range(border, width - border):
            window = image[y - border : y + border + 1, x - border : x + border + 1]
            geometric_mean = np.prod(window)
            filtered_image[y, x] = (geometric_mean**(1/(kernel_size**2))).astype(image.dtype)
    return filtered_image

gm = geometric_mean_filter(noisy_image)
plt.imshow(gm)
plt.show()

def harmonic_mean_filter(image, kernel_size=3):
    height, width,color = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    border = kernel_size//2
    
    for y in range(border, height - border):
        for x in range(border, width - border):
            window = image[y - border : y + border + 1, x - border : x + border + 1]
            harmonic_mean = kernel_size**2 / np.sum(1.0 / (window + 1e-6))
            filtered_image[y, x] = (harmonic_mean).astype(image.dtype)
    return filtered_image

hm = harmonic_mean_filter(noisy_image)
plt.imshow(hm)
plt.show()

def counter_harmonic_mean_filter(image, kernel_size=3, Q=1.5):
    height, width,color = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    border = kernel_size//2
    
    for y in range(border, height - border):
        for x in range(border, width - border):
            window = image[y - border : y + border + 1, x - border : x + border + 1]
            numerator = np.sum(np.power(window, Q+1))
            denominator = np.sum(np.power(window, Q))
            
            if denominator != 0:
                counter_harmonic_mean = numerator / denominator
            else:
                counter_harmonic_mean = 0
                
            filtered_image[y, x] = (counter_harmonic_mean).astype(image.dtype)
    return filtered_image

chm = counter_harmonic_mean_filter(noisy_image)
plt.imshow(chm)
plt.show()

