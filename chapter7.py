# 导入必要的库
import cv2  # OpenCV 用于图像处理的库
import numpy as np  # Numpy 用于处理数组
from chapter6 import stackImages

# 定义一个空函数，用作 TrackBar 的回调函数
# TrackBar 的回调函数需要一个参数，但在这里我们不需要处理任何事情
def empty(a):
    pass

# 创建一个名为 "TrackBars" 的窗口，用于显示滑动条
cv2.namedWindow("TrackBars")

# 调整 "TrackBars" 窗口的大小为 640x240 像素
cv2.resizeWindow("TrackBars", 640, 240)

# 在 "TrackBars" 窗口中添加滑动条，用于调整 HSV 的上下阈值
# 每个滑动条包括滑动范围、初始值和最大值
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)  # 色调最小值滑动条
cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)  # 色调最大值滑动条
cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)  # 饱和度最小值滑动条
cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)  # 饱和度最大值滑动条
cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)  # 亮度最小值滑动条
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)  # 亮度最大值滑动条

# 进入一个无限循环，实时处理图像
while 1:
    # 读取图像文件，路径为 "Resources/lambo.PNG"
    img = cv2.imread("Resources/lambo.PNG")  # 原始图像
    if img is None:  # 如果图像未加载成功（路径错误或文件丢失），则跳过
        print("图像文件未找到，请检查路径是否正确")
        break

    # 将图像从 BGR 颜色空间转换为 HSV 颜色空间
    # HSV 更适合用于颜色过滤，因为它将颜色信息与亮度信息分离
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 从滑动条获取 HSV 的上下阈值
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")  # 获取色调最小值
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")  # 获取色调最大值
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")  # 获取饱和度最小值
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")  # 获取饱和度最大值
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")  # 获取亮度最小值
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")  # 获取亮度最大值

    # 打印当前滑动条的值（调试用）
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    # 定义下阈值和上阈值，用于颜色过滤
    # 下阈值：HSV 的最小值数组
    lower = np.array([h_min, s_min, v_min])
    # 上阈值：HSV 的最大值数组
    upper = np.array([h_max, s_max, v_max])

    # 根据 HSV 阈值创建一个掩码图像
    # 符合范围的像素值为 255（白色），不符合的为 0（黑色）
    mask = cv2.inRange(img_HSV, lower, upper)

    # 使用掩码对原始图像进行按位与运算，保留符合条件的部分
    # 不符合条件的部分将被设置为黑色
    img_Result = cv2.bitwise_and(img, img, mask=mask)

    # # 显示原始图像
    # cv2.imshow("Original", img)
    #
    # # 显示转换后的 HSV 图像（用于调试）
    # cv2.imshow("HSV", img_HSV)
    #
    # # 显示掩码图像（黑白图像，白色为符合 HSV 范围的区域）
    # cv2.imshow("Mask", mask)
    #
    # # 显示最终的结果图像（只保留符合 HSV 范围的颜色部分）
    # cv2.imshow("Result", img_Result)

    img_stack = stackImages(0.7, ([img, img_HSV], [mask, img_Result]))
    cv2.imshow("StackImages", img_stack)

    # 等待 1 毫秒以刷新窗口
    # 如果用户按下 "q" 键，可以退出程序（如果需要退出功能，可以扩展代码）
    cv2.waitKey(1)

# RGB 更适合处理图像的基本显示和操作，而 HSV 更适合颜色相关的高级任务，比如颜色分割、目标跟踪等。