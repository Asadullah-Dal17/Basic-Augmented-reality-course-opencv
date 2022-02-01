import cv2 as cv
import numpy as np

img = cv.imread("images/images_1.jpg")
print(img.shape)

cv.putText(
    img=img,
    text=f"opencv version: {cv.__version__}",
    org=(30, 40),
    fontFace=cv.FONT_HERSHEY_PLAIN,
    fontScale=1.5,
    color=(0, 255, 0),
    thickness=2,
    lineType=cv.LINE_AA,
)
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
