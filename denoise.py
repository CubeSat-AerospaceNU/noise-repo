#@Jack
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import os
import PIL

#params: https://shimat.github.io/opencvsharp_2410/html/635f2450-96f2-cee1-9d4f-7b2c191c6d1d.htm

for image in os.listdir('original'):
    if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.png') or image.endswith('.webp'):
        print(image)
        #img = os.path.join('original',img)
        img = cv2.imread(os.path.join('original',image))

        denoised = cv2.fastNlMeansDenoisingColored(img,None,15,20,7,30) 
        #enoised = cv2.fastNlMeansDenoisingMulti(img,None,15,21,7,31) 
        output_name = os.path.join("denoised",(image.split(".")[0]+'.png'))
        cv2.imwrite(output_name, denoised) 
        #fastNlMeansDenoisingMulti