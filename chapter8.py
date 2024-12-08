# 导入必要的库
import cv2  # OpenCV库，用于图像处理和计算机视觉任务
import numpy as np  # NumPy库，用于数值计算和数组操作
from stcakImages import stackImages  # 自定义模块或函数，用于将多张图像堆叠在一起显示


# 定义一个函数，用于从二值化图像中提取轮廓，并对轮廓进行形状分析和分类
def getContours(img):
    # 使用OpenCV的findContours函数提取图像中的轮廓信息
    # 参数说明：
    # - img: 输入的是二值化图像（黑白图像）
    # - cv2.RETR_EXTERNAL: 只检测外部的轮廓
    # - cv2.CHAIN_APPROX_NONE: 保存轮廓上的所有点，不进行压缩
    # 返回值：
    # - contours: 检测到的轮廓列表，每个轮廓是一个点集
    # - hierarchy: 轮廓的层级结构（在这段代码中未使用）
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 遍历所有找到的轮廓
    for cnt in contours:
        # 调用cv2.contourArea函数计算轮廓的面积
        # 面积是轮廓内部的像素点总数，用于过滤掉噪声和小的区域
        area = cv2.contourArea(cnt)
        print(f"轮廓面积: {area}")  # 打印每个轮廓的面积

        # 设置一个面积阈值，忽略小于500的轮廓，避免处理噪声
        if area > 500:
            # 绘制轮廓到图像上，用蓝色 (255, 0, 0) 表示，线宽为3
            # 参数说明：
            # - img_Contour: 绘制目标图像
            # - cnt: 当前的轮廓点集
            # - -1: 绘制所有轮廓
            # - (255, 0, 0): 颜色（BGR格式）
            # - 3: 线宽
            cv2.drawContours(img_Contour, cnt, -1, (255, 0, 0), 3)

            # 调用cv2.arcLength函数计算轮廓的周长
            # 参数 True 表示轮廓是封闭的
            peri = cv2.arcLength(cnt, True)
            print(f"轮廓周长: {peri}")  # 打印每个轮廓的周长

            # 使用cv2.approxPolyDP函数对轮廓进行多边形近似
            # 参数说明：
            # - cnt: 当前轮廓点集
            # - 0.02 * peri: 近似精度（值越小，近似越接近原始轮廓）
            # - True: 表示轮廓是封闭的
            # 返回值：
            # - approx: 近似得到的多边形的顶点集合
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # 计算多边形的顶点数
            objCor = len(approx)
            print(f"顶点数: {objCor}")  # 打印每个轮廓的顶点数

            # 使用cv2.boundingRect函数获取多边形的最小边界框
            # 返回值：
            # - (x, y): 边界框左上角的坐标
            # - (w, h): 边界框的宽度和高度
            x, y, w, h = cv2.boundingRect(approx)

            # 根据顶点数对形状进行分类
            if objCor == 3:  # 如果顶点数为3，判断为三角形
                objectType = "Triangle"
            elif objCor == 4:  # 如果顶点数为4，可能是正方形或矩形
                # 计算宽高比（aspect ratio）
                aspRatio = w / float(h)
                # 如果宽高比接近1，认为是正方形
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                else:  # 否则认为是矩形
                    objectType = "Rectangle"
            elif objCor > 4:  # 如果顶点数大于4，判断为圆形
                objectType = "Circle"
            else:  # 如果不满足以上条件，则认为无法分类
                objectType = "None"

            # 在图像上绘制边界框
            # 参数说明：
            # - img_Classify: 绘制目标图像
            # - (x, y): 边界框左上角坐标
            # - (x + w, y + h): 边界框右下角坐标
            # - (0, 127, 127): 深黄色边框
            # - 2: 边框线宽
            cv2.rectangle(img_Classify, (x, y), (x + w, y + h), (0, 127, 127), 2)

            # 在图像上标注形状名称
            # 参数说明：
            # - img_Classify: 绘制目标图像
            # - objectType: 要标注的文本内容（形状名称）
            # - (x + w // 2 - 40, y + h // 2): 文本的位置，居中显示
            # - cv2.FONT_HERSHEY_COMPLEX: 字体类型
            # - 0.7: 字体大小
            # - (0, 0, 0): 文本颜色（黑色）
            # - 2: 文本线宽
            cv2.putText(img_Classify, objectType, (x + w // 2 - 40, y + h // 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)


# 主程序部分，加载图像并进行处理
# 读取彩色图像，路径为 "Resources/shapes.png"，请确保路径有效
img = cv2.imread("Resources/shapes.png")

# 将彩色图像转换为灰度图像
# 灰度图像是单通道的（每个像素表示亮度），便于后续处理
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图像进行高斯模糊，降低噪声
# 参数说明：
# - (7, 7): 模糊核大小（越大越模糊）
# - 1: 高斯核的标准差
img_Blur = cv2.GaussianBlur(img_Gray, (7, 7), 1)

# 使用Canny边缘检测提取图像中的边缘
# 参数说明：
# - 50, 50: 阈值1和阈值2，用于边缘检测
img_Canny = cv2.Canny(img_Blur, 50, 50)

# 创建两个图像副本，用于绘制轮廓和分类结果
img_Contour = img.copy()
img_Classify = img.copy()

# 调用getContours函数，传入边缘检测后的图像
# 在img_Contour上绘制轮廓，在img_Classify上标注分类
getContours(img_Canny)

# 使用自定义的stackImages函数将多张图片堆叠在一起
# 参数说明：
# - 0.7: 缩放比例，将图像大小缩小到原来的70%
# - ([img, img_Gray, img_Blur], [img_Canny, img_Contour, img_Classify]): 图像的排列方式
img_stack = stackImages(0.7, ([img, img_Gray, img_Blur],
                              [img_Canny, img_Contour, img_Classify]))

# 显示堆叠后的图像，窗口名称为 "stack"
cv2.imshow("stack", img_stack)

# 等待用户按键，按键后关闭窗口
cv2.waitKey(0)