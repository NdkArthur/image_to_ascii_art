# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:08:07 2021

@author: andok
"""
import numpy as np
import cv2
import string
from matplotlib import pyplot as plt
import fire


def get_idx_closest(v, values):
      idx = np.argmin(np.abs(values -v))
      return idx
  
def normalize(a, bot_quant=0, top_quant=1, gamma=1):
    
    a = a**(1/gamma)
    if bot_quant>0 and top_quant<1:
        top = np.quantile(a, top_quant)
        bot = np.quantile(a, bot_quant)
        a = np.clip(a, bot, top)
    a = a.astype(np.float)
    a_norm = (a-np.min(a))/(np.max(a)-np.min(a))
    return 255*(a_norm**gamma)



def compute_lums(outfile):
    size = 28
    font                   = cv2.FONT_HERSHEY_SIMPLEX()
    bottomLeftCornerOfText = (1, int(1.6*size),)
    fontScale              = 1
    fontColor              = (0, 0, 0)
    lineType               = 2
    
    NUM_PRINTABLE = 95
      

    for i, char in enumerate(string.printable[:NUM_PRINTABLE]):
        img = 255*np.ones((2*size, size,3), np.uint8)
        cv2.putText(img, char, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.imshow("letter", cv2.resize(img, (100, 200)))
        cv2.waitKey(1000)
        lums[i]= np.mean(img)

    lums_norm = normalize(lums)
    np.savetxt("outfile", lums_norm)


def image_to_ascii(src_image, outfile, line_size=120, contrast_strength=0.01, gamma=1):
    """
    Convert an image (formats supported by openCV) to a ascii art text file

    Parameters
    ----------
    src_image : TYPE
        source image.
    outfile : TYPE
        path to output file.
    line_size : TYPE, optional
        Line length in the output text file (controls the resolution). The default is 120.
    contrast_strength : TYPE, optional
        Increases contrast, from 0 (no change) to 0.5. The default is 0.01.
    gamma : TYPE, optional
        gamma correction exponent. The default is 1.

    Returns
    -------
    None.

    """
    lums_norm = np.loadtxt("character_luminosity.data")
    top_quantile = 1 - contrast_strength
    bottom_quantile = contrast_strength
    
    im_gray = cv2.imread(src_image,  flags=cv2.IMREAD_GRAYSCALE)
    height = im_gray.shape[0]
    width = im_gray.shape[1]
    ratio = float(height)/width
    print("Source image: height: {}, width: {}, ratio {:.2f}".format(height, width, ratio))
    
    n_rows = int(0.7*line_size*ratio)
    print("Output image: n rows: {}, line length: {}, ratio {:.2f}".format(
        n_rows, line_size, n_rows/line_size))


    im_resized = cv2.resize(im_gray, (line_size, n_rows))
    im_norm_clip = normalize(cv2.resize(im_gray/255, (line_size, n_rows)), 
                             bottom_quantile, top_quantile, gamma)
    
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(im_resized , cmap="gray")
    # ax[0].set_title("resized")
    # ax[1].imshow(im_norm_clip, cmap="gray")
    # ax[1].set_title("normalized")
    # fig.show()
    lines = []
    for i in range(n_rows):
        l = ""
        for j in range(line_size):
            idx = get_idx_closest(im_norm_clip[i,j], lums_norm)
            l += string.printable[idx]
        lines.append(l)
        
    text = "\n".join(lines)
    with open(outfile, "w") as f:
        f.write(text)
        
# image_to_ascii("joconde.jpg", "joconde.txt",
#                line_size=240, 
#                contrast_strength=0.05, 
#                gamma=0.6)

if __name__=="__main__":
    fire.Fire(image_to_ascii)


