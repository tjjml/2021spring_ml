import cv2
import matplotlib.pyplot as plt
MAX_WIDTH = 500
#加载图像
img = cv2.imread("./test/1.jpg")
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pic_hight, pic_width = img.shape[:2]
pic_hight,pic_width

#调整图像大小
resize_rate = MAX_WIDTH / pic_width
img = cv2.resize(img, (MAX_WIDTH, int(pic_hight*resize_rate)), interpolation=cv2.INTER_AREA)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#高斯去噪
img = cv2.GaussianBlur(img,(3,3),0)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#保存旧的图像，便于以后操作
oldimg = img
#转化灰度图
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np

#去掉图像中不会是车牌的区域
kernel = np.ones((20, 20), np.uint8)
#形态学滤波（开运算，先腐蚀后膨胀）
img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow("img_opening",img_opening)
cv2.waitKey(0)
cv2.destroyAllWindows()


#图像加权
img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
cv2.imshow("img_open",img_opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

#二值化
ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("img_thresh",img_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 边缘检测(找到图像边缘)
img_edge_canny = cv2.Canny(img_thresh, 100, 200)
cv2.imshow("img_edge_canny",img_edge_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


image_x=cv2.Sobel(img_opening,cv2.CV_64F,1,0,ksize=3)  #X方向Sobel

absX=cv2.convertScaleAbs(image_x)   # 转回uint8
cv2.imshow("absX",absX)

image_y=cv2.Sobel(img_opening,cv2.CV_64F,0,1,ksize=3)  #Y方向Sobel
absY=cv2.convertScaleAbs(image_y)
cv2.imshow('absY',absY)

#进行权重融合
dst=cv2.addWeighted(absX,0.5,absY,0.5,0)
ret, img_thresh_sobel = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('dst',dst)
cv2.imshow("img_thresh_soble",img_thresh_sobel)
cv2.waitKey()

cv2.imshow("canny",img_edge_canny)
cv2.imshow("sobel",img_thresh_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()



kernel = np.ones((4, 19))
# 闭运算，先进行膨胀然后进行腐蚀操作。通常是被用来填充前景物体中的小洞，或者抹去前景物体上的小黑点。因为可以想象，其就是先将白色部分变大，把小的黑色部分挤掉，然后再将一些大的黑色的部分还原回来，整体得到的效果就是：抹去前景物体上的小黑点了。
img_edge1_canny = cv2.morphologyEx(img_edge_canny, cv2.MORPH_CLOSE, kernel)

# 开运算（通过先进行腐蚀操作，再进行膨胀操作得到。移除小的对象时候很有用(假设物品是亮色，前景色是黑色)，被用来去除噪声。）
img_edge2_canny = cv2.morphologyEx(img_edge1_canny, cv2.MORPH_OPEN, kernel)
cv2.imshow("img_edge2_canny",img_edge2_canny)
cv2.waitKey(0)