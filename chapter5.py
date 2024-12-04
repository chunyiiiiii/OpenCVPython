# 导入必要的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数组和矩阵操作

# 设置目标图像的宽度和高度（即透视变换后的图像尺寸）
width, height = 250, 350

# 读取输入图像
# "Resources/cards.jpg" 是输入图像路径，请确保路径正确
img = cv2.imread("Resources/cards.jpg")

# 定义原始图像上的四个点的坐标（这四个点围成了一个四边形区域）
# 这些点是从原始图像中选取的感兴趣区域（例如卡片的四个角点）
pts1 = np.float32([
    [111, 219],  # 左上角点
    [154, 482],  # 左下角点
    [287, 188],  # 右上角点
    [352, 440]   # 右下角点
])

# 定义目标图像上的四个点的坐标（这些点围成了一个矩形区域）
# 这些点是透视变换后图像的角点，定义为矩形的四个顶点
pts2 = np.float32([
    [0, 0],               # 目标图像的左上角
    [0, height],          # 目标图像的左下角
    [width, 0],           # 目标图像的右上角
    [width, height]       # 目标图像的右下角
])

# 使用 OpenCV 的 getPerspectiveTransform 方法计算透视变换矩阵
# 输入点：pts1（原始图像上的四个点）和 pts2（目标图像上的四个点）
# 输出：matrix 是一个 3x3 的变换矩阵，用于将原图像的四边形区域变换为矩形区域
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# 使用 warpPerspective 方法对图像进行透视变换
# 输入图像：img（原始图像）
# 变换矩阵：matrix（透视变换矩阵）
# 输出图像的大小：宽度和高度 (width, height)
# 结果：imgOutput 是透视变换后的图像
imgOutput = cv2.warpPerspective(img, matrix, (width, height))

# 显示原始图像
# 窗口名为 "Image"，内容为原始图像 img
cv2.imshow("Image", img)

# 显示透视变换后的图像
# 窗口名为 "WarpPerspective"，内容为变换后的图像 imgOutput
cv2.imshow("WarpPerspective", imgOutput)

# 使用 waitKey 方法等待用户按下任意键
# 参数 0 表示等待无限长时间，直到用户按键
cv2.waitKey(0)