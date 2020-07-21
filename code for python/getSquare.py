import math

def getNum(height):
    num = round(height / 1000)
    num = num if num != 0 else num + 2
    num = num if num % 2 == 0 else num + 1
    return num

def reduceLength(top, bottom, value):
    top += math.floor(value / 2)
    bottom -= math.ceil(value / 2)
    return top, bottom

def increaseLength(top, bottom, value):
    top -= math.floor(value / 2)
    bottom += math.ceil(value / 2)
    return top, bottom

def GetSquare(img, height, width, center, radius):

    center = center[0]
    radius = radius [0] #if radius[0] % (4 * num) == 0 else int(radius[0] / (4 * num)) * (4 * num)

    left = center[1] - radius if center[1] - radius > 0 else 0
    right = center[1] + radius if center[1] + radius < width else width
    top = center[0] - radius if center[0] - radius > 0 else 0
    bottom = center[0] + radius if center[0] + radius < height else height
    
    if right-left < bottom -top:
        diff = ((bottom - top) - (right - left))
        top, bottom = reduceLength(top, bottom, diff)
    elif right - left > bottom - top:
        diff = ((right - left) - (bottom - top))
        left, right = reduceLength(left ,right, diff)
    
    num = getNum(right - left)
    diff = int((right - left) / (8 * num)) * (8 * num) - (right - left)
    left, right = increaseLength(left, right, diff)
    diff = int((bottom - top) / (8 * num)) * (8 * num) - (bottom - top)
    top, bottom = increaseLength(top, bottom, diff)

    img = img[left: right, top : bottom]
    return img, num