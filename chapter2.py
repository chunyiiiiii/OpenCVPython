import cv2
import numpy as np

# 创建一个全是1的矩阵，大小为7*7，类型为无符号整型
kernel = np.ones((7, 7), np.uint8)
img = cv2.imread("Resources/lena.png")

#灰度图像
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#模糊图像，添加高斯滤波
img_Blur = cv2.GaussianBlur(img_Gray, (7, 7), 0)
#边缘检测
#函数有两个阈值参数，第一个阈值参数为低阈值，用于确定哪些梯度变化被认为是潜在的边缘。所有梯度值高于低阈值的像素点都被认为是潜在的边缘点。
# 第二个阈值参数为高阈值，用于确定哪些潜在的边缘点是真正的边缘。所有梯度值高于高阈值的像素点都被认为是真正的边缘点。
# 同时，所有梯度值低于低阈值的像素点都被认为不是边缘点。
# 在实际应用中，合适的阈值参数需要根据具体图像和任务进行调整，以获得最佳效果。通常，可以通过试验不同的参数值来确定最佳的阈值参数。
img_Canny = cv2.Canny(img, 100, 150)

#膨胀操作，对图像边缘进行扩张
img_Dilation = cv2.dilate(img_Canny, kernel=kernel, iterations=1)

#腐蚀操作，这个操作会把前景物体的边界腐蚀掉
img_Eroded = cv2.erode(img_Dilation, kernel=kernel, iterations=1)

cv2.imshow("Gary Image", img_Gray)
cv2.imshow("Blur Image", img_Blur)
cv2.imshow("Canny Image", img_Canny)
cv2.imshow("Dilation Image", img_Dilation)
cv2.imshow("Erode Image", img_Eroded)
cv2.waitKey(0)