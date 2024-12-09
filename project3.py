# 导入OpenCV库，用于图像处理和车牌检测
import cv2

# 配置参数
frameWidth = 640  # 设置视频帧宽度
frameHeight = 480  # 设置视频帧高度

# 加载车牌检测的Haar特征分类器模型
nPlateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")

minArea = 200  # 车牌检测的最小面积阈值，用于过滤较小的噪声
color = (255, 0, 255)  # 车牌框的颜色 (粉红色)

# 打开视频文件
cap = cv2.VideoCapture("Resources/video12.mp4")

# 设置视频帧的宽度、高度和亮度
cap.set(3, frameWidth)  # 属性ID 3：设置帧宽度
cap.set(4, frameHeight)  # 属性ID 4：设置帧高度
cap.set(10, 150)  # 属性ID 10：设置亮度

count = 0  # 初始化计数器，用于保存车牌图像时的命名

# 循环读取视频帧
while True:
    success, img = cap.read()  # 从视频读取一帧图像
    if not success:  # 如果没有成功读取到帧（视频结束）
        break

    # 将图像转换为灰度图像（Haar分类器需要灰度图像作为输入）
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用Haar分类器检测车牌
    # detectMultiScale参数说明：
    # - imgGray: 输入的灰度图像
    # - 1.1: 每次图像尺寸缩小的比例（越小检测越精确，但计算量更大）
    # - 10: 每个候选矩形需要的最少邻近矩形数量
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)

    # 遍历检测到的车牌区域
    for (x, y, w, h) in numberPlates:
        # 计算车牌区域的面积
        area = w * h
        if area > minArea:  # 只处理面积大于最小阈值的区域
            # 在原始图像上绘制车牌的矩形边框
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # 在车牌框上方添加文字 "Number Plate"
            cv2.putText(img, "Number Plate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

            # 提取车牌区域的图像（感兴趣区域，ROI）
            imgRoi = img[y:y + h, x:x + w]

            # 显示提取的车牌区域
            cv2.imshow("ROI", imgRoi)

    # 显示处理后的图像（包括标注车牌的矩形框和文字）
    cv2.imshow("Result", img)

    # 检测按键操作
    if cv2.waitKey(1) & 0xFF == ord('s'):  # 如果按下 's' 键
        # 保存ROI图像到指定路径，文件名格式为 "NoPlate_<计数器>.jpg"
        cv2.imwrite("Resources/Scanned/NoPlate_" + str(count) + ".jpg", imgRoi)

        # 在原始图像上绘制绿色填充矩形，表示已保存车牌
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)

        # 在图像上显示 "Scan Saved" 的提示信息
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
                    2, (0, 0, 255), 2)

        # 显示保存提示的图像
        cv2.imshow("Result", img)

        # 等待500毫秒（0.5秒），展示保存提示信息
        cv2.waitKey(500)

        # 增加计数器，用于保存下一个车牌图像
        count += 1