import cv2
from networktables import NetworkTables
import numpy as np
import math

def nothing(x):
    pass

Tm = 0.205
fov = 27.7665349671
tan_frame = math.tan(math.radians(fov))

cap = cv2.VideoCapture(1)
slider_img = np.zeros((300,512,3), np.uint8)

winname = 'slider'
cv2.namedWindow(winname)

cv2.createTrackbar('h_min', winname, 0, 255, nothing)
cv2.createTrackbar('s_min', winname, 0, 255, nothing)
cv2.createTrackbar('v_min', winname, 0, 255, nothing)
cv2.createTrackbar('h_max', winname, 255, 255, nothing)
cv2.createTrackbar('s_max', winname, 255, 255, nothing)
cv2.createTrackbar('v_max', winname, 255, 255, nothing)

# NetworkTables.setIPAdress('10.45.86.2')
# NetworkTables.setClientMode()
# NetworkTables.initialize()
# table = NetworkTables.getTable('imgProc')
# table.putValue('distance', 0)

while(True):
    # _, img = cap.read()
    img = cv2.imread("2019VisionImages\CargoSideStraightDark36in.jpg")
    ABpx,_,_= img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_lower = np.array(
    # [cv2.getTrackbarPos('h_min', winname),
    # cv2.getTrackbarPos('s_min', winname),
    # cv2.getTrackbarPos('v_min', winname)])
    # hsv_upper = np.array(
    # [cv2.getTrackbarPos('h_max', winname),
    # cv2.getTrackbarPos('s_max', winname),
    # cv2.getTrackbarPos('v_max', winname)])
    # mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    
    mask = cv2.inRange(hsv, np.array([62, 142, 84]),
                             np.array([255, 255, 255]))
    
    filltered = []
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        m = 700
        if(area > m ):
            print(area)
            filltered.append(cnt)
    new_cnt = filltered[0]
    for cnt in filltered:
        new_cnt += cnt
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cnt)/(w*h)
        Tpx = w
    
    cv2.drawContours(img,filltered, -1, (0,255,255), 2)

    cv2.imshow('frame', img)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(1)
    if(key == 27):
        break

cv2.destroyAllWindows()