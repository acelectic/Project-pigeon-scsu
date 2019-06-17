import cv2
from until.yolo_model import yolo_model
import time



class yolo_detect:
    # index = 0
    # _, img = cv2.VideoCapture('pigeon-12.mp4').read()
    def __init__(self):
        # self.cap = cv2.VideoCapture(0)
        self.yolo = yolo_model()

    def run(self):
        testvdo = '/home/minibear3e/Desktop/Pigeon Detect/Yolo_example/video/YouTube.mp4'
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(testvdo)
        res, img = cap.read()

        while res:
            self.yolo.detect(img)
        cap.release()

    def runtest(self):
        testvdo = '/home/minibear3e/Desktop/Pigeon Detect/Yolo_example/video/YouTube.mp4'
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(testvdo)


        while res:
            res, img = cap.read()
            self.yolo.detect(img)
        cap.release()


# yolo_detect().runtest()