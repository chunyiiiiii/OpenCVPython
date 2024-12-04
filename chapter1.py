import cv2

# 1.读取图片并展示
# 使用imread来读取
img = cv2.imread("Resources/lena.png")
# 使用imshow来展示 第一个参数为窗口名称
cv2.imshow("lena", img)
# 使用waitKey来延迟窗口的关闭
cv2.waitKey(0)

# 2.读取视频并展示
frameWidth = 640
frameHeight = 480
# 使用VideoCapture获取视频
cap = cv2.VideoCapture("Resources/test_video.mp4")
# 视频是很多帧图片在一起,通过循环进行展示
while 1:
    #success是图片的状态，是否获取成功
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    cv2.imshow("test_video", img)
    # ord（“q”）是q的ASCII码值，设置q键退出视频播放，
    # cv2.waitKey(1)和0xFF进行与运算，避免操作系统不同导致前者值不同
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 3.调用摄像头
# 笔记本就一个摄像头，序号0
cap = cv2.VideoCapture(0)
# 设置宽、高、亮度
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
while 1:
    success, img = cap.read()
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
