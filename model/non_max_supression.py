import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def nms(image) -> np.ndarray:
    assert isinstance(image, np.ndarray)

    image = np.squeeze(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype(np.float32)
    image = cv.GaussianBlur(image, (11, 11), sigmaX = 0)
    

    Gy = cv.Sobel(image, cv.CV_64F, dx = 0,  dy = 1)
    Gx = cv.Sobel(image, cv.CV_64F, dx = 1, dy = 0)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    angles = np.arctan2(Gy, Gx)
    angles = np.degrees(angles)
    angles[angles < 0] += 180  


    H, W = magnitude.shape
    non_max_supression = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            direction = angles[i, j]
            m = magnitude[i, j]

            # Define os pixels vizinhos de acordo com a direção do gradiente
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                before = magnitude[i, j - 1]
                after = magnitude[i, j + 1]
            elif 22.5 <= direction < 67.5:
                before = magnitude[i - 1, j + 1]
                after = magnitude[i + 1, j - 1]
            elif 67.5 <= direction < 112.5:
                before = magnitude[i - 1, j]
                after = magnitude[i + 1, j]
            elif 112.5 <= direction < 157.5:
                before = magnitude[i - 1, j - 1]
                after = magnitude[i + 1, j + 1]

            if m >= before and m >= after:
                non_max_supression[i, j] = m
            else:
                non_max_supression[i, j] = 0

    # Normaliza resultado para 0-1
    non_max_supression = non_max_supression / (non_max_supression.max() + 1e-8)
    return non_max_supression

