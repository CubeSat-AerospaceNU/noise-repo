import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("n2.jpg", 1)
kernel = np.ones((3,3), np.float32)/9

filt_2D = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (3,3))
gaussian_blur = cv2.GaussianBlur(img, (3,3), 0)
median_blur = cv2.medianBlur(img, 3)
bilateral_blur = cv2.bilateralFilter(img, 75, 50, 500)
gaussian_blur2 = cv2.GaussianBlur(bilateral_blur, (5,5), 0)
filt_2D2 = cv2.filter2D(gaussian_blur2, -1, kernel)
dst = cv2.fastNlMeansDenoisingColored(gaussian_blur2, None, 7, 10, 10, 15)
imgAltered = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
finalImage = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)



#cv2.imshow("Original", img)
#cv2.imshow("2D custom filter", filt_2D)
#cv2.imshow("Blur", blur)
cv2.imshow("Gaussian Blur", gaussian_blur)
#cv2.imshow("Median Filter", median_blur)
#cv2.imshow("Median Bilateral Blur", bilateral_blur)
#cv2.imshow("Median Bilateral Gaussian Blur", gaussian_blur2)
#cv2.imshow("Filterd Median Bilateral Gaussian Blur", filt_2D2)
#cv2.imshow("Denoising", dst)
plt.subplot(121),plt.imshow(imgAltered)
plt.subplot(122),plt.imshow(finalImage)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()