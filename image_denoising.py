import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

extensions = ["jpg", "png", "webp", "jpeg"]
out_path = "out/"

greyscale_factor = 10
color_factor = 5
tws = 7 # template window size
sws = 21 # search window size


# Returns if f matches one of the extensions. Should use regex instead.
def correctFileType(f):
    parts = f.split('.')
    return (parts[-1] in extensions) and (len(parts)>1)


# returns a list of all the images in the dir
def getImagesInDir(dir):
    return [f for f in os.listdir(dir) if (os.path.isfile(f) and correctFileType(f))]


# denoises image at image_path using the specified parameters,
# saving it at out_path
def denoiseImage(img_path, out_path, greyscale_factor, color_factor, tws, sws):
    parts = img_path.split('.')
    path = out_path + parts[0] + "_denoised_" + str(greyscale_factor) + "_" + str(color_factor) + "_" + str(tws) + "_" + str(sws) + "." + parts[-1]
    cv2.imwrite(path, cv2.fastNlMeansDenoisingColored(cv2.imread(img_path), None, greyscale_factor, color_factor, tws, sws))


def main():
    images = getImagesInDir('.')

    for img_path in images: 
        denoiseImage(img_path, 'out/', greyscale_factor, color_factor, tws, sws)


if __name__ == '__main__':
    main()