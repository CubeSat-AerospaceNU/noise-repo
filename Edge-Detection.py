import cv2
import numpy as np

img = cv2.imread("n7.jpg", 0)
bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)

edges = cv2.Canny(img, 100, 200)
bilateral_edges = cv2.Canny(bilateral_blur, 100, 200)


cv2.imshow("Original", img)
cv2.imshow("Original Edges", edges)
cv2.imshow("Bilateral Blur", bilateral_blur)
cv2.imshow("Bilateral Edges", bilateral_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()