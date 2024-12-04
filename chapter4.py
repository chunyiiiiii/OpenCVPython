import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
# img[:] = 255, 0, 0

# 在图像上添加线条
# shape函数的输出为宽、高、通道数，宽、高对应列、行
cv2.line(img, (0, 0),(img.shape[1], img.shape[0]), (0, 255, 0), 3)

# 在图像上添加矩形
# 参数为要处理的图像、矩形左上角、矩形右下角、颜色、宽度（为-1时可以填充矩形）
cv2.rectangle(img, (50, 20), (400, 450), (255, 0, 255), 3)

# 在图像上添加圆
# 参数为图像、圆心、半径、颜色、宽度
cv2.circle(img, (300, 80), 30, (0, 255, 255), 3)

# 在图像上添加文字
# 参数为图像、文本、原点、字体、大小、颜色、宽度
cv2.putText(img, "Chunyiiiiii", (300,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 3)

cv2.imshow("Image", img)
cv2.waitKey(0)