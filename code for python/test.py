import os
import numpy as np
from cv2 import cv2
import time

from getCircle import GetCircle  #得到外围的函数
from getSquare import GetSquare  #得到外围外接正方形的函数
from crop import Crop            #切成小图的函数
from merge import merge          #把小图密度图矩阵合并的函数
from MyNet import MyNet          #网络


import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import torch.nn as nn
import torch.nn.functional as F


#inpath = '/home/whd/new_image/all/'
inpath = './test_img/'
filename = '08-57-24.jpg'

outpath = './out/'
crop_outpath = './crop_out/'
den_outpath = './den_out/'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.benchmark = True
model_path = './model.pth'


mean_std = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std) ])

if __name__ == '__main__':
    
    print('start loading module')
    
    time0_start = time.time()
    net = MyNet()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()


    name = os.path.splitext(filename)[0]
    infile = os.path.join(inpath,filename)
    outfile = os.path.join(outpath,filename)
    den_outfile = os.path.join(den_outpath,filename)
   
    img = cv2.imread(infile,0)

    print('start getting circle')

    img, height, width, allCenter, allRadius = GetCircle(img)


    print('start cutting img into small pieces')

    all_img,num = GetSquare(img, height, width, allCenter, allRadius)

    cv2.imwrite(outfile,all_img)
    padding = 16

    net_inpath,size = Crop(all_img,num, infile, crop_outpath, padding)
    
    print('piece len:',size)


    
    print('start referencing the small pieces')

    out = []

    for filename in os.listdir(net_inpath) :

        imgname = os.path.join(net_inpath,filename)
  
        img = cv2.imread(imgname , 1)
 
        img = img_transform(img)
        with torch.no_grad():
            inputs = Variable(img[None,:,:,:]).cuda()
            outputs = net(inputs)
            pred_map = outputs.cpu().data.numpy()[0,0,:,:]
            out.append(pred_map)
    
  
    row_len = int((out[0].shape)[0] - padding/2)

    
    print('start merging the small pieces')


    all_out = merge(out,num,padding)


    number = int(round(all_out.sum()))


  
    print('start drawing circles')

    points = all_out * 100
    res, binary = cv2.threshold(points, 3.5, 255, cv2.THRESH_BINARY)
    if number < 300:
        out = cv2.dilate(binary, None, iterations=2)
    else :
        out = cv2.dilate(binary, None, iterations=1)
        
  
    

    out = np.array(out, dtype='uint8')

 
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(out, connectivity=4)
 
    centroids = centroids[1:, ]

    nLabels = nLabels - 1
    center_x = centroids[:, 0]
    center_y = centroids[:, 1]

    
    
    all_img = cv2.imread(outfile,1)

    for i in range(nLabels):
        cv2.circle(all_img,(int(round(center_x[i])), int(round(center_y[i]))),5,(0,0,255),-1)

    cv2.putText(all_img, 'Number: ' + str(nLabels) , (int(allRadius[0]-300), int(allRadius[0])) , cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0),8)
    
    cv2.imwrite(outfile, all_img)



    



 




