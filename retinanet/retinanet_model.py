import time
from datetime import datetime
from uuid import uuid4

import numpy as np
import radar
# set tf backend to allow memory to grow, instead of claiming everything
from keras_retinanet.models import load_model
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption


# thai_timezone = pytz.timezone('Asia/Bangkok')

class Model:
    def __init__(self, confidence=0.5, es=None, es_mode=False):

        # Size image for train on retinenet
        self.min_side4train = 600
        self.max_side4train = 800

        # Size image for Save 2 elasticsearch
        self.min_side4elas = 600
        self.max_side4elas = 800

        # labels_to_names = {0: 'Pigeon'}
        # print('model confidence:', self.confThreshold)

        # self.time2store = self.gen_datetime()
        self.time2store = datetime.now()
        self.__data4turret = None
        self.__lastFrame = np.zeros((600, 800, 1))

        # # es = Elasticsearch()
        # self.es = Elasticsearch([{'host': '192.168.1.29', 'port': 9200}])
        # # es = Elasticsearch([{'host': '172.27.228.44', 'port': 9200}])
        #
        if es == None or es.checkStatus() == False:
            # raise ValueError("Connection failed")
            self.es_status = False
        else:
            self.es_status = True

        self.es_mode = es_mode

        self.confThreshold = float(confidence)

        self.model = load_model(
            r'C:\Users\Kuy Loan\Desktop\Project-pigeon-scsu\retinanet\model\resnet50_coco_best_v2.1.0.h5',
            backbone_name='resnet50')

        self.labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                6: 'train',
                                7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
                                18: 'sheep',
                                19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                                31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                                35: 'baseball glove',
                                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                                41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                                48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                                54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                                60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
                                71: 'sink',
                                72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                                77: 'teddy bear',
                                78: 'hair drier', 79: 'toothbrush'}

    def gen_datetime(self, from_date, to_date):

        from_date = from_date.split('-')

        return radar.random_date(
            start=datetime(year=from_date.year, month=from_date.month, day=from_date.day),
            stop=datetime(year=to_date.year, month=to_date.month, day=to_date.day))

    def gen_datetime(self):
        return radar.random_date(
            start=datetime(year=2019, month=6, day=1),
            stop=datetime(year=2019, month=6, day=17))

    def detect(self, image):
        # copy to draw on
        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=self.min_side4train, max_side=self.max_side4train)

        time_ = self.time2store
        # time_ = datetime.now()

        eventid = time_.strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        # print('imgae_id', eventid)
        # print('time', time_)
        # print('local', self.localtimezone(time_))
        # print('utc',  self.utctimezone(time_))

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        processing_time = time.time() - start
        print("processing time: ", processing_time)

        img4elas, scale4elas = resize_image(draw, min_side=self.min_side4elas, max_side=self.max_side4elas)

        # correct for image scale
        boxes /= scale4elas
        found_ = {}

        main_body = {'eventid': eventid, 'time_': time_}
        self.__data4turret = {
            "box": boxes[0][0],
            "score": scores[0][0],
            "label": labels[0][0],
            "scale": scale,
            "raw_image-shape": draw.shape}
        # visualize detections
        index = 1
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            # print(index, box, score, self.labels_to_names[label])
            if score < self.confThreshold:
                break
            color = label_color(label)

            b = box.astype(int)

            draw_box(img4elas, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            # print(caption)
            draw_caption(img4elas, b, caption)
            box = [np.ushort(x).item() for x in box]

            try:
                found_[self.labels_to_names[label]] += 1
            except:
                found_[self.labels_to_names[label]] = 1

            if self.es_mode and self.es_status:
                self.es.elas_record(label=label, score=np.float32(score).item(), box=box, **main_body)
            index += 1

        self.__updatelastFrame(img4elas)
        print()
        if self.es_mode and self.es_status:
            self.es.elas_image(image=img4elas, scale=scale, found_=found_, processing_time=processing_time, **main_body)
        return 'Now: {}\nelas_id: {}\tbirds: {}\nProcess Time: {}\n{}'.format(datetime.now(),
                                                                              eventid, len(boxes), processing_time,
                                                                              '\n{:#>20} {} {:#<20}'.format('',
                                                                                                            'END 1 FRAME',
                                                                                                            ''))

    def getCentroid(self, box):
        return (int((box[0] + box[2])/2), int((box[1] + box[3])/2))

    def __updatelastFrame(self, img):
        self.__lastFrame = img

    def _getlastFrame(self):
        return self.__lastFrame

    def getDataTurret(self):
        tmp = self.__data4turret
        data = tmp
        data['centroid']= self.getCentroid(tmp['box'])

        return data

    def _sentdata2turret(self):
        pass

    def setConfidence(self, confidence):
        print("confident: {} ---> {}".format(self.confThreshold, confidence))
        self.confThreshold = float(confidence)


import cv2

if __name__ == '__main__':
    detect_model = Model()
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
            cv2.waitKey()
