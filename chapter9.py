# 导入 OpenCV 库，用于图像处理和计算机视觉
import cv2

# 设置视频帧的宽度和高度
frameWidth = 640  # 视频帧宽度设置为 640 像素
frameHeight = 480  # 视频帧高度设置为 480 像素

# 初始化摄像头，参数 0 表示使用默认摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头的属性
cap.set(3, frameWidth)  # 设置视频帧的宽度，属性 ID 为 3
cap.set(4, frameHeight)  # 设置视频帧的高度，属性 ID 为 4
cap.set(10, 150)  # 设置摄像头的亮度，属性 ID 为 10，值为 150

# 使用一个无限循环处理摄像头捕获的视频流
while 1:
    # 从摄像头读取当前帧，返回两个值：
    # success 是布尔值，表示帧捕获是否成功
    # img 是捕获到的当前帧图像
    success, img = cap.read()

    # 加载预训练的 Haar 特征级联分类器，用于人脸检测
    # "Resources/haarcascade_frontalface_default.xml" 是 XML 文件路径
    # 文件中包含人脸检测的特征数据
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

    # 将捕获的彩色帧转换为灰度图像
    # 人脸检测在灰度图像上的计算效率更高
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 Haar 特征分类器检测灰度图像中的人脸
    # detectMultiScale 返回检测到的人脸矩形坐标的列表
    # 参数解释：
    # 1.1 是缩放系数，用于缩放检测窗口的大小
    # 4 是最小邻居数，每个候选矩形至少需要 4 个邻居才能被保留
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

    # 遍历检测到的人脸矩形列表
    for (x, y, w, h) in faces:
        # 在原始彩色图像上绘制矩形框，标记出人脸区域
        # 参数解释：
        # (x, y) 是矩形框左上角的坐标
        # (x + w, y + h) 是矩形框右下角的坐标
        # (255, 0, 0) 是矩形框的颜色，蓝色 (BGR)
        # 2 是矩形框的厚度
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示带有矩形框的人脸检测结果的实时视频帧
    # "Webcam" 是显示窗口的标题
    cv2.imshow("Webcam", img)

    # 检测键盘输入，等待 1 毫秒，返回按键的 ASCII 值
    # 如果用户按下 'q' 键（ASCII 值为 ord('q')），则退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放摄像头资源，停止视频捕获
cap.release()

# 关闭所有 OpenCV 窗口，释放显示资源
cv2.destroyAllWindows()