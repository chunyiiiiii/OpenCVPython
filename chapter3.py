import cv2

img = cv2.imread("Resources/lambo.PNG")
print(img.shape)

# 矩阵输入是行、列，对应处理后图像的高、宽
img_resize = cv2.resize(img, (300, 200))
# 输出是宽、高、通道数
print(img_resize.shape)

# 裁剪图像，对应的行、列
img_cropped = img[0:200, 200:500]

cv2.imshow("Lambo", img)
cv2.imshow("Lambo_Resize", img_resize)
cv2.imshow("Lambo_Cropped", img_cropped)
cv2.waitKey(0)