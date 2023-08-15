import cv2
import matplotlib.pyplot as plt
import numpy as np

image=cv2.imread("image.jpg") #insert your image here
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
plt.imshow(am)
plt.show()


