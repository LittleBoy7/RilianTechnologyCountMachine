# RilianTechnologyCountMachine

## 项目背景：
项目开发可用于对日联科技原有基于线阵的点料机的算法系统进行升级，也可用于新设计的基于FPD取像的新设备，算法预研究阶段采用的是原有线阵取像方式的设备所存储的图像，后期需要通过一定的算法标注，数据集扩充等方式对基于FPD取信的设备图像进行兼容.  
## 项目目的：
1）对各种尺寸料号的兼容性，解决传统算法需要根据料号进行参数设置的问题；  
2）解决由于射线照射角度导致部分料的图像有黏连的问题；  
3）解决由于电压电流兼容性问题导致的部分较大料图像亮度对比度较低时的识别问题；  
4）算法不仅能给出较准确的零件数目，也要给出相应的零件中心。  
## 算法流程
1. [总体流程](#总体流程)
2. [预处理](#预处理)
3. [模型推理](#模型推理)
4. [后处理](#后处理)
5. [结果展示](#结果展示)
### 总体流程  
首先对原始数据进行预处理，包括裁掉图片周围黑边，检测轮盘外圆并得到外接正方形，根据外接正方形切割成小图。对分割后的小图进行模型推理，得到小图对应的密度图矩阵。最后经过后处理，将小图的密度图矩阵合并成原始大图的密度图矩阵，根据密度图矩阵得到计数矩阵以及元件位置信息。  
<img src="https://github.com/LittleBoy7/RilianTechnologyCountMachine/blob/master/images/36.png" alt="算法流程" width="500" height="400" align="center" />
### 预处理  
预处理主要包括裁剪周围黑边，检测轮盘外圆并得到外接正方形，以及小图的裁剪。最终得到的效果图如下所示： 
<img src="https://github.com/LittleBoy7/RilianTechnologyCountMachine/blob/master/images/37.png" alt="小图裁剪" width="909" height="235" align="center" />
### 模型推理  
将一张大图裁剪得到的所有小图送入设计的网络模型进行模型推理，网络得到与输入小图大小一致的密度图矩阵，用于后续的计数与位置回归。输入小图与密度矩阵可视化热图如图所示:
<img src="https://github.com/LittleBoy7/RilianTechnologyCountMachine/blob/master/images/38.png" alt="模型推理" width="799" height="398" align="center" />
### 后处理  
由于裁剪小图过程中，边界区域被切割成两半的元件，会造成较大漏检与误检的情况，在小图周围覆盖过渡带，缓解这一问题:  
<img src="https://github.com/LittleBoy7/RilianTechnologyCountMachine/blob/master/images/40.png" alt="后处理" width="302" height="278" align="center" />
### 结果展示  
