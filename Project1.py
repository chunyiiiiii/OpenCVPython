# 导入必要的库
import cv2  # OpenCV库，用于处理视频流、图像处理等计算机视觉任务
import numpy as np  # NumPy库，用于数值计算和数组操作

# 设置摄像头捕获视频帧的宽度和高度
frameWidth = 640  # 视频帧宽度
frameHeight = 480  # 视频帧高度

# 打开摄像头（设备ID为0，表示默认摄像头）
cap = cv2.VideoCapture(0)  # 通过摄像头捕获视频流

# 设置摄像头的属性
cap.set(3, frameWidth)  # 属性ID 3：设置视频帧的宽度
cap.set(4, frameHeight)  # 属性ID 4：设置视频帧的高度
cap.set(10, 150)  # 属性ID 10：设置摄像头的亮度

# 定义需要检测的颜色范围（HSV颜色空间）
# 每个颜色的范围是 [Hue_min, Sat_min, Val_min, Hue_max, Sat_max, Val_max]
myColors = [[67, 18, 115, 86, 41, 160],  # 青绿色
            [148, 25, 72, 176, 45, 130],  # 粉紫色
            [0, 50, 120, 3, 75, 200]]  # 红色

# 定义每个颜色对应的BGR值，用于绘制标记
# BGR值表示蓝、绿、红通道的颜色值
myColorValues = [[255, 255, 0],  # 黄色 (BGR)
                 [255, 102, 178],  # 粉红色 (BGR)
                 [0, 0, 153]]  # 深蓝色 (BGR)

# 用于存储检测到的点 (x, y, colorId)
myPoints = []


# 函数：检测图像中的给定颜色
def findColor(img, myColors, myColorValues):
    # 将图像从BGR颜色空间转换为HSV颜色空间
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    count = 0  # 用于跟踪当前处理的颜色索引
    newPoints = []  # 用于临时存储新检测到的点

    # 遍历每种颜色范围
    for color in myColors:
        # 定义颜色范围的下限和上限
        lower = np.array(color[0:3])  # 下限 [Hue_min, Sat_min, Val_min]
        upper = np.array(color[3:6])  # 上限 [Hue_max, Sat_max, Val_max]

        # 使用颜色范围创建掩码，过滤掉图像中不在范围内的颜色
        mask = cv2.inRange(img_HSV, lower, upper)

        # 调用getContours函数，找到掩码中的目标并获取其中心点
        x, y = getContours(mask)

        # 如果检测到目标（x, y 不为0），绘制目标点并将其添加到newPoints列表
        cv2.circle(img_Result, (x, y), 10, myColorValues[count], -1)  # 绘制标记点
        if x != 0 and y != 0:
            newPoints.append([x, y, count])  # 将点的坐标和颜色索引存储
        count += 1  # 更新颜色索引

    # 返回检测到的新点
    return newPoints


# 函数：找到目标的轮廓，并提取其中心点
def getContours(img):
    # 使用findContours函数找到轮廓
    # 参数说明：
    # - img: 输入的二值化图像
    # - cv2.RETR_EXTERNAL: 提取外部轮廓
    # - cv2.CHAIN_APPROX_NONE: 保存轮廓上的每个点，不进行压缩
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 初始化变量，用于存储目标的位置和大小
    x, y, w, h = 0, 0, 0, 0

    # 遍历每一个轮廓
    for cnt in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(cnt)
        if area > 500:  # 过滤掉面积小于500的噪声轮廓
            # 计算轮廓的周长
            peri = cv2.arcLength(cnt, True)
            # 使用多边形近似简化轮廓
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # 获取多边形的外接矩形
            x, y, w, h = cv2.boundingRect(approx)

    # 返回目标的中心点坐标
    return x + w // 2, y


# 函数：在画布上绘制所有存储的点
def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        # 根据点的颜色索引绘制标记点
        cv2.circle(img_Result, (point[0], point[1]), 10, myColorValues[point[2]], -1)


# 主循环：捕获视频帧并处理
while 1:
    # 从摄像头读取一帧图像
    success, img = cap.read()
    # 创建一个副本，用于绘制结果
    img_Result = img.copy()

    # 调用findColor函数，检测图像中的目标颜色
    newPoints = findColor(img, myColors, myColorValues)

    # 如果检测到新的点，将其添加到myPoints列表
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)

    # 调用drawOnCanvas函数，在画布上绘制所有点
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)

    # 显示处理后的图像
    cv2.imshow("Result", img_Result)

    # 按下 "q" 键退出程序
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break