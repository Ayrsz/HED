import cv2 as cv

im = cv.imread("/home/softex/Documents/mas11/states/model_04/epoch#final/final_result_epoch_#final.jpg", 0)

cv.imwrite("result.jpg", ~im)
