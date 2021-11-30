from tensorflow import keras
from PIL import Image, ImageEnhance

import os, math
import numpy as np

# Real-world application 
ALGODIM_W = 32
ALGODIM_H = 32
MODELPATH = 'autoencoderfile-15epochs'
model = keras.models.load_model(MODELPATH) # loads the model into the program

def main():
    '''
    Runs all the helper functions, then saves the scaled images from 
    prepimagescale() and the predicted images from runalgo() to file.
    '''
    dir = os.getcwd()
    rawimgs = prepimagedict(dir)
    scaledimgs = prepimagescale(rawimgs)
    imgoutput = runalgo(scaledimgs)

    for filekeyscaled, filekeyout in zip(scaledimgs, imgoutput):
        imgout = imgoutput.get(filekeyout)
        imgout.save(f"{dir}\denoised32-32algo\output\{filekeyout}")

        imgscaled = scaledimgs.get(filekeyscaled)
        imgscaled.save(f"{dir}\denoised32-32algo\downscaled\{filekeyscaled}")
    print("All images saved.")
        

def prepimagedict(directory: str) -> dict:
    '''
    Gets the height and width of the image and scales both to a multiple 
    of 32 to prepare image for splitting into 32*32 chunks that will 
    then be fed to the algorithm. Returns a dictionary of images that
    were present in the given directory. 
    '''
    rawimgs = {}
    print(f"Image directory: {directory}")

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(filename).convert('RGB')
            if img.size[0] > img.size[1]:
                imgsize = (math.ceil(img.size[0]/ALGODIM_W)*ALGODIM_W, 
                    math.ceil(img.size[0]/ALGODIM_H)*ALGODIM_H)
            else: 
                imgsize = (math.ceil(img.size[1]/ALGODIM_W)*ALGODIM_W, 
                    math.ceil(img.size[1]/ALGODIM_H)*ALGODIM_H)
            # gets h and w of img and scales to mult of 32
            
            img = ImageEnhance.Contrast(img).enhance(1.7) # enhances contrast of image for downscaling
            
            baseimg = Image.new("RGB", imgsize, "GRAY")
            # creates a new image that is a mult of 32 to offset scaling issues
            baseimg.paste(img, (0, 0)) 
            rawimgs.update({filename : baseimg})
            # baseimg.show()
            
    return rawimgs


def prepimagescale(rawimgs: dict) -> dict:
    '''
    Scales down an image to 32*32 pixels for the algorithm. 
    Returns a dictionary of ndarrays.
    '''
    imgkeys = rawimgs.keys() # imgkeys is type dict_keys, we must typecast to list first in order to index
    rescaledimgs = {}
    
    for imgname in list(imgkeys):
    
        img = rawimgs.get(imgname) # gets the image for a corresponding key
        
        #img.show()
        imgby32 = img.resize((32, 32))
        rescaledimgs.update({imgname : imgby32})
        #imgby32.show()

    return rescaledimgs


def runalgo(scaledimgs: dict) -> dict:
    '''
    Preps the scaled images, then runs them thru the autoencoder. 
    Returns a dictionary of the predicted images. 
    '''
    tupleimglist = list(scaledimgs.items())
    imglist = []
    algoimgs = {}

    for img in tupleimglist:
        imglist.append(np.asarray(img[1]))
    
    imgarr = np.array(imglist).astype('float16')/255.
    #print(imgarr)
    predimgs = model.predict(imgarr)

    for image, key in zip(predimgs, list(scaledimgs.keys())):
        algoimgs.update({key : Image.fromarray(np.uint8(image*255))})
    
    return algoimgs


if __name__ == '__main__':
    main()