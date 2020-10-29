import cv2

def resize(src, width, height):
    dst = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
    return dst
