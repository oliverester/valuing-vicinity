from collections import defaultdict
from PIL import Image
import time 

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (max(im1.width,im2.width), im1.height + im2.height), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height,im2.height)), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst