import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('n7.jpg')


#dst = cv.fastNlMeansDenoisingColored(img,None,20,20,7,21)
dst = cv.fastNlMeansDenoising(img,None,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()


*cv.imwrite('n5.jpg', dst)