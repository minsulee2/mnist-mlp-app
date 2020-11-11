from PIL import Image, ImageOps

def preprocess(src, width, height):
    gray_image = src.convert('L')
    resized_image = gray_image.resize((width, height))
    dst = ImageOps.invert(resized_image)
    return dst
