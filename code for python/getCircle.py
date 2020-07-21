from cv2 import  cv2
from clearBlackEdge import ClearBlackEdge #去黑边函数，有的图有黑边阴影，会影响外围的检测
import time
def GetCircle(img):
    #ClearBlackEdge - GaussianBlur - Binary - Erode -FindCountour
    time0_start = time.time()
    img = ClearBlackEdge(img)  #去黑边
    time0_end = time.time()
    t0 = time0_end - time0_start
    #print('black:',t0)
    time2_start = time.time()
    blur = cv2.GaussianBlur(img,(9,9),0)  #高斯平滑
    ret3, binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  #双峰二值化，不过现在有一种情况有点问题，后续鲤鱼进一步改进
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (300, 300))  #定义一直贼大的矩形核
    grad = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) #用刚才那个贼大的核，做个闭操作
    out = cv2.erode(grad, None, iterations=10) #膨胀10次
    time2_end = time.time()
    t2 = time2_end-time2_start
    #print(t2)
    #以上骚操作，都是为了把圆盘变成个实心圆

    #边缘检测得到一些外围的候选
    time1_start = time.time()
    contours, hier = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    time1_end = time.time()
    t1 = time1_end-time1_start
    #print('edge:',t1)
    height, width = img.shape
    allCenter = []
    allRadius = []
    #这个循环是为了过滤掉一些不太外围候选，得到外围
    
    for c in contours :
        (x, y), radius = cv2.minEnclosingCircle(c)
        if width/3 < x < width/3*2 and 0.1*width < radius < 0.8*width:
            allCenter.append((int(x), int(y)))
            allRadius.append(int(radius)+100)
        else:
            pass

    

        # for i in range(len(allCenter)):
        #     img = cv2.circle(img, allCenter[i], allRadius[i], (0, 0, 255), 10)
    return img, height, width, allCenter, allRadius
#返回值是去了黑边的大图及其长宽，外围中心与半径