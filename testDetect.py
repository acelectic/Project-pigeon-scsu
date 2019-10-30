import cv2
import time
from until.elas_api import elas_api
es_ip = '192.168.1.29'
es_port = 9200
es = elas_api(ip=es_ip)
es_status = None
es_status = es.checkStatus()[0]

from retinanet import testModel


if __name__ == '__main__':
    es = elas_api(ip = '192.168.1.29')
    detect_model = testModel.Model(es=es, es_mode=True)

    img = cv2.VideoCapture(r"video/YouTube4.mp4")
    while 1:
        _, frame = img.read()
        if _:
            detect_model.detect(frame)

