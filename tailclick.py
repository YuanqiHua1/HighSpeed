import cv2
import numpy as np


def handleclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = x
        param[1] = y


def pickpoint(frame):
    cv2.namedWindow('first')
    cv2.imshow("first", frame)
    cv2.moveWindow('first', 0, 0)
    cv2.waitKey(10)
    point = np.array([-1, -1])
    cv2.setMouseCallback("first", handleclick, point)
    print("Click on start of the fish's tail")
    cv2.waitKey(10)
    while (point == np.array([-1, -1])).all():
        cv2.waitKey(10)
    cv2.destroyWindow('first')
    return point


def picktwopoints(frame):
    cv2.namedWindow('first')
    cv2.imshow("first", frame)
    cv2.moveWindow('first', 0, 0)
    cv2.waitKey(10)
    point = np.array([-1, -1])
    cv2.setMouseCallback("first", handleclick, point)
    print("Click on start of the fish's tail")
    cv2.waitKey(10)
    while (point == np.array([-1, -1])).all():
        cv2.waitKey(10)
    startpoint = point
    # cv2.circle(frame, tuple(startpoint), 5, (0,0,0))
    cv2.drawMarker(frame, tuple(startpoint), (255, 0, 0), markerSize=30, thickness=3)
    cv2.imshow("first", frame)

    point = np.array([-1, -1])
    print("Click on end of the fish's tail")
    cv2.setMouseCallback("first", handleclick, point)
    while (point == np.array([-1, -1])).all():
        cv2.waitKey(10)

    cv2.drawMarker(frame, tuple(point), (0, 0, 255), markerSize=30, thickness=3)
    cv2.imshow("first", frame)
    cv2.waitKey(20)

    cv2.destroyWindow('first')
    return [startpoint, point]


