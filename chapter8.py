import cv2
import numpy as np
from stcakImages import stackImages

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(img_Contour, cnt, -1, (255, 0, 0), 3)

            peri = cv2.arcLength(cnt, True)
            print(peri)

            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            print(objCor)

            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 3 :
                objectType = "Triangle"
            elif objCor == 4 :
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05 :
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4 :
                objectType = "Circle"
            else:
                objectType = "None"

            cv2.rectangle(img_Classify, (x, y), (x + w, y + h), (0, 127, 127),2)
            cv2.putText(img_Classify, objectType, (x + w // 2 - 40, y + h // 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)



img = cv2.imread("Resources/shapes.png")
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_Blur = cv2.GaussianBlur(img_Gray, (7, 7), 1)
img_Canny = cv2.Canny(img_Blur, 50, 50)

img_Contour = img.copy()
img_Classify = img.copy()

# img_Black = np.zeros_like(img)

getContours(img_Canny)

img_stack = stackImages(0.7, ([img, img_Gray, img_Blur],
                              [img_Canny, img_Contour, img_Classify]))

cv2.imshow("stack", img_stack)


cv2.waitKey(0)
