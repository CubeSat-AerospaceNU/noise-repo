import os
from PIL import Image
from PIL import ImageFilter

directory = os.getcwd()
print(f"Image directory: {directory}")

filters = [ImageFilter.GaussianBlur(4), ImageFilter.SMOOTH, ImageFilter.EDGE_ENHANCE(), 
           ImageFilter.UnsharpMask(radius=3, percent=180, threshold=2), ImageFilter.MaxFilter(1)]

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(filename)
        img = img.convert('RGB')
        for filter in filters:
            img = img.filter(filter)
        img.save(f"{directory}\denoisedpil\{filename}")
        print(f"{filename} - saved to \\denoisedpil")