import cv2 as cv

im1 = cv.imread("model/checkpoints/model_14/epoch#0/epoch_0.png")
im2 = cv.imread("model/checkpoints/model_14/epoch#final/epoch_final.png")

cv.imwrite("/model/dif.png", im2)