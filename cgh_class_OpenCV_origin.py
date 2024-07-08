import cv2

src = cv2.imread('c:/images/Nature99(Small)/picture02_gray.jpg')
#src = cv2.imread('c:/images/Nature99(Small)/picture04.jpg')
#_, binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)

print(src.ndim, src.shape, src.dtype)

cv2.imshow("src",src)
cv2.imwrite('c:/images/Nature99(Small)/picture02_gray.jpg',src)
#cv2.imshow("binary",binary)
cv2.waitKey(0)
cv2.destroyWindow("src")
#cv2.destroyWindow("binary")
