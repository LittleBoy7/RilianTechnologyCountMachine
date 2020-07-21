import os
import math
import numpy as np
from cv2 import cv2

def Crop(img, num, file, outpath,padding):
    sum_rows = img.shape[0]  # 高度
    sum_cols = img.shape[1]  # 宽度

    cols = int(sum_cols / num)
    rows = int(sum_rows / num)

    outpath = outpath + os.path.splitext(os.path.basename(file))[0]
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    print(str(num) + " from crop")
    #print(file)

    #padding = 192
    for i in range(num):
        for j in range(num):
            file = os.path.basename(file)
            outname = outpath + '/' + os.path.splitext(file)[0] + '_' + str(j) + '_' + str(i) + os.path.splitext(file)[1]
            top = j * rows
            bottom = (j + 1) * rows
            left = i * cols
            right = (i + 1) * cols
            top -= padding if j >0 else 0
            bottom += padding if j < num - 1 else 0
            left -= padding if i > 0 else 0
            right += padding if i < num - 1 else 0
            #print((right - left - cols) / padding)
            out = img[ top : bottom, left : right]
            cv2.imwrite(outname, out)
    return outpath, rows
