# 导入必要的库
import cv2  # OpenCV库，用于图像处理和计算机视觉任务
import numpy as np  # NumPy库，用于数值计算和数组操作
from stcakImages import stackImages  # 自定义模块或函数，用于将多张图像堆叠在一起显示

# 设置摄像头捕获图像的宽度和高度
widthImg = 640  # 图像宽度
heightImg = 480  # 图像高度

# 打开摄像头（设备ID为0，表示默认摄像头）
cap = cv2.VideoCapture(0)

# 设置摄像头的属性
cap.set(3, widthImg)  # 属性ID 3：设置视频帧的宽度
cap.set(4, heightImg)  # 属性ID 4：设置视频帧的高度
cap.set(10, 150)  # 属性ID 10：设置摄像头的亮度


# 图像预处理函数
def preProcessing(img):
    """
    对输入图像进行灰度化、模糊、边缘检测和膨胀腐蚀操作，
    提高后续轮廓检测的准确性。
    """
    # 将图像从BGR颜色空间转换为灰度图像
    img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊，减少噪声（核大小为5x5，标准差为1）
    img_Blur = cv2.GaussianBlur(img_Gray, (5, 5), 1)

    # 使用Canny边缘检测提取图像的边缘
    img_Canny = cv2.Canny(img_Blur, 200, 200)

    # 使用膨胀操作增强边缘
    kernel = np.ones((5, 5))  # 定义5x5的矩阵作为膨胀核
    img_Dila = cv2.dilate(img_Canny, kernel, iterations=2)

    # 使用腐蚀操作去除一些噪声点
    img_Thres = cv2.erode(img_Dila, kernel, iterations=1)

    # 返回处理后的二值化图像
    return img_Thres


# 获取图像轮廓并提取最大的四边形函数
def getContours(img):
    """
    在二值化图像中检测轮廓，并找到面积最大的四边形轮廓。
    """
    biggest = np.array([])  # 初始化变量，用于存储最大的四边形
    maxArea = 0  # 初始化最大面积为0

    # 使用findContours函数提取轮廓
    # 参数说明：
    # - img: 输入的二值化图像
    # - cv2.RETR_EXTERNAL: 提取外部轮廓
    # - cv2.CHAIN_APPROX_NONE: 保存轮廓上的每个点，不进行压缩
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 遍历每一个轮廓
    for cnt in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(cnt)
        if area > 500:  # 过滤掉面积小于500的噪声轮廓
            # 计算轮廓的周长
            peri = cv2.arcLength(cnt, True)
            # 使用多边形近似简化轮廓
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # 如果轮廓是四边形且面积大于当前最大面积，则更新最大值
            if len(approx) == 4 and area >= maxArea:
                biggest = approx  # 更新最大的四边形轮廓
                maxArea = area  # 更新最大面积

    # 绘制检测到的最大四边形轮廓
    cv2.drawContours(img_Contour, biggest, -1, (255, 0, 0), 20)

    # 返回最大的四边形轮廓
    return biggest


# 图像透视变换函数
def getWarp(img, biggest):
    """
    对检测到的最大四边形进行透视变换，将其转换为俯视图。
    """
    # 对四边形的顶点进行重新排序，使其顺序为 [左上, 右上, 左下, 右下]
    biggest = reorder(biggest)

    # 定义透视变换的源点（四边形的顶点）
    pts1 = np.float32([biggest])

    # 定义透视变换的目标点（标准矩形的四个角）
    pts2 = np.float32([
        [0, 0],  # 目标图像的左上角
        [widthImg, 0],  # 目标图像的右上角
        [0, heightImg],  # 目标图像的左下角
        [widthImg, heightImg]  # 目标图像的右下角
    ])

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 应用透视变换，生成俯视图
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # 对透视变换后的图像进行裁剪并调整大小
    img_Cropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    img_Cropped = cv2.resize(img_Cropped, (widthImg, heightImg))

    # 返回处理后的图像
    return img_Cropped


# 顶点重新排序函数
def reorder(myPoints):
    """
    按照 [左上, 右上, 左下, 右下] 的顺序重新排列四边形的四个顶点。
    """
    myPoints = myPoints.reshape((4, 2))  # 将顶点数组从 (4,1,2) 转换为 (4,2)
    myPointsNew = np.zeros((4, 1, 2), np.int32)  # 初始化新的顶点数组

    # 按顶点坐标的 x+y 之和找到左上角和右下角
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # 左上角
    myPointsNew[3] = myPoints[np.argmax(add)]  # 右下角

    # 按顶点坐标的 x-y 的差找到右上角和左下角
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # 右上角
    myPointsNew[2] = myPoints[np.argmax(diff)]  # 左下角

    # 返回重新排序的顶点
    return myPointsNew


# 主循环：实时捕获图像并处理
while True:
    # 从摄像头读取一帧图像
    success, img = cap.read()

    # 调整图像大小到指定的宽度和高度
    img = cv2.resize(img, (widthImg, heightImg))

    # 创建一个副本，用于绘制轮廓
    img_Contour = img.copy()

    # 调用预处理函数对图像进行处理
    img_Thres = preProcessing(img)

    # 调用轮廓检测函数，找到最大的四边形
    biggest = getContours(img_Thres)

    # 如果检测到有效的四边形，则进行透视变换
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)  # 透视变换
        imageArray = ([img_Contour, imgWarped])  # 显示原图和透视变换结果
        cv2.imshow("ImageWarped", imgWarped)  # 显示透视变换后的图像
    else:
        imageArray = ([img_Contour, img])  # 如果没有检测到四边形，则显示原图像

    # 将图像堆叠在一起显示
    stackedImages = stackImages(0.6, imageArray)
    cv2.imshow("WorkFlow", stackedImages)  # 显示堆叠的图像

    # 按下 "q" 键退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break