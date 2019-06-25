

from elasticsearch import Elasticsearch
from datetime import datetime
import pytz, radar, base64
from uuid import uuid4



from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2

import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
from keras_retinanet.models import load_model
import pytz
from tzlocal import get_localzone


thai_timezone = pytz.timezone('Asia/Bangkok')

class Model:
    def __init__(self, confidence=0.5):

        # self.time2store = self.gen_datetime()
        self.time2store = datetime.now()

        # es = Elasticsearch()
        self.es = Elasticsearch([{'host': '192.168.1.29', 'port': 9200}])
        # es = Elasticsearch([{'host': '172.27.228.44', 'port': 9200}])

        if not self.es.ping():
            # raise ValueError("Connection failed")
            self.es_status = False
        else:
            print(self.es.info)
            self.es_status = True
            self.es_index = 'pigeon-test'
            self.es_image = 'pigeon-image-test2'

            # self.es_index = 'pigeon-recoed-test3'
            # self.es_image = 'pigeon-image-test3'

            # self.es_index = 'pre-data'
            # self.es_image = 'pre-image'

        self.confThreshold = float(confidence)

        self.weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

        self.hour_text = ["00.00-01.00", "01.00-02.00", "02.00-03.00", "03.00-04.00", "04.00-05.00", "05.00-06.00",
                     "06.00-07.00", "07.00-08.00", "08.00-09.00", "09.00-10.00", "10.00-11.00", "11.00-12.00",
                     "12.00-13.00", "13.00-14.00", "14.00-15.00", "15.00-16.00", "16.00-17.00", "17.00-18.00",
                     "18.00-19.00", "19.00-20.00", "20.00-21.00", "21.00-22.00", "22.00-23.00", "23.00-24.00"]

        self.months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']

        self.model = load_model('retinanet/model/resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')

        self.labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                           7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                           12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                           19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                           25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                           31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                           36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                           41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                           48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                           54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                           60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                           66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                           72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                           78: 'hair drier', 79: 'toothbrush'}

        # Size image for train on retinenet
        self.min_side4train = 600
        self.max_side4train = 800

        # Size image for Save 2 elasticsearch
        self.min_side4elas = 600
        self.max_side4elas = 800

        # labels_to_names = {0: 'Pigeon'}
        # print('model confidence:', self.confThreshold)

    def month_int(self, time):
        return time.month

    def month_text(self, time):
        return self.months[time.month - 1]

    def dayofweek_int(self, time):
        return time.weekday()

    def dayofweek_text(self, time):
        return self.weekDays[time.weekday()]

    def localtimezone(self, time):
        local_tz = get_localzone()
        timezone = pytz.timezone(str(local_tz))
        return timezone.localize(time)

    def utctimezone(self, time):
        # local_tz = get_localzone()
        # return time.astimezone(local_tz)
        try:
            utc_tz = pytz.timezone('UTC')
            return time.astimezone(utc_tz)
        except:
            utc_tz = pytz.timezone('UTC')
            tmp_ = self.localtimezone(time)
            return tmp_.astimezone(utc_tz)

    def img2string(self, img):
        retval, buffer = cv2.imencode('.jpg', img)
        # Convert to base64 encoding and show start of data
        return str(base64.b64encode(buffer))

    def gen_datetime(self, from_date, to_date):

        from_date = from_date.split('-')

        return radar.random_date(
        start = datetime(year=from_date.year, month=from_date.month, day=from_date.day),
        stop = datetime(year=to_date.year, month=to_date.month, day=to_date.day))

    def gen_datetime(self):
        return radar.random_date(
        start = datetime(year=2019, month=6, day=1),
        stop = datetime(year=2019, month=6, day=17))


    def elas_image(self, eventid, image, scale, time_, found_, processing_time):
        if self.es_status:
            body = {}
            body['original_image'] = self.img2string(image)
            body['scale'] = scale

            body['timestamp'] = self.localtimezone(time_)
            # body['timestamp_utc'] = self.utctimezone(time_)

            body['Hour_int'] = int(time_.strftime('%-H'))
            body['Hour_text'] = self.hour_text[body['Hour_int']]

            body['dayofweek_int'] = self.dayofweek_int(time_)
            body['dayofweek_text'] = self.dayofweek_text(time_)

            body['Mouth_int'] = self.month_int(time_)
            body['Mouth_text'] = self.month_text(time_)

            body['found'] = found_

            body['processing_time'] = processing_time
            # print(body)
            self.es.index(index=self.es_image, id=eventid, body=body)

    def elas_record(self, eventid, time_, label, score, box):
        if self.es_status:
            body = {}
            body['timestamp'] = self.localtimezone(time_)
            # body['timestamp_utc'] = self.utctimezone(time_)

            body['Hour_int'] = int(time_.strftime('%-H'))
            body['Hour_text'] = self.hour_text[body['Hour_int']]

            body['dayofweek_int'] = self.dayofweek_int(time_)
            body['dayofweek_text'] = self.dayofweek_text(time_)

            body['Mouth_int'] = self.month_int(time_)
            body['Mouth_text'] = self.month_text(time_)

            body['image_id'] = eventid
            body['found'] = {self.labels_to_names[label]: 1}
            body['confidence'] = score
            body['box'] = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}

            # print(body)
            self.es.index(index=self.es_index, body=body)

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

        main_body = {'eventid':eventid,  'time_':time_}

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < self.confThreshold:
                break
            color = label_color(label)

            b = box.astype(int)

            draw_box(img4elas, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            print(caption)
            draw_caption(img4elas, b, caption)
            box = [np.ushort(x).item() for x in box]

            try:
                found_[self.labels_to_names[label]] += 1
            except:
                found_[self.labels_to_names[label]] = 1

            self.elas_record(label=label, score=np.float32(score).item(), box=box, **main_body)

        print()

        self.elas_image(image=img4elas, scale=scale, found_=found_, processing_time=processing_time, **main_body)


    def setConfidence(self, confidence):
        self.confThreshold = float(confidence)
