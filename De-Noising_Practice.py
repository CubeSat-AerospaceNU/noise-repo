import cv2
import numpy as np

img = cv2.imread("n2.jpg", 1)
kernel = np.ones((3,3), np.float32)/9

filt_2D = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (3,3))
gaussian_blur = cv2.GaussianBlur(img, (3,3), 0)
median_blur = cv2.medianBlur(img, 3)
bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)


cv2.imshow("Original", img)
cv2.imshow("2D custom filter", filt_2D)
cv2.imshow("Blur", blur)
cv2.imshow("Gaussian Blur", gaussian_blur)
cv2.imshow("Median Filter", median_blur)
cv2.imshow("Bilateral Blur", bilateral_blur)


cv2.waitKey(0)
cv2.destroyAllWindows()