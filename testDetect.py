import cv2
from until.elas_api import elas_api
import retinanet.retinanet_model

if __name__ == '__main__':
    es = elas_api(ip = '192.168.1.29')
    detect_model = Model(es=es, es_mode=True)

    img = cv2.VideoCapture(r"video/YouTube4.mp4")
    img = cv2.VideoCapture(0)
    while 1:
        _, frame = img.read()

        if _:
            detect_model.detect(frame)
            turretData = detect_model.getDataTurret()
            print(turretData)
            img_ = detect_model._getlastFrame()
            img_ = cv2.drawMarker(img_, turretData['centroid'], color=(255, 125, 128), markerSize=4)
            cv2.imshow('turret', img_)

            # cv2.imshow('sss', frame)
            cv2.waitKey()