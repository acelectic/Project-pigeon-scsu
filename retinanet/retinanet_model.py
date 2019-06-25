

from elasticsearch import Elasticsearch
from datetime import datetime
import pytz, radar, base64
from uuid import uuid4

thai_timezone = pytz.timezone('Asia/Bangkok')

# es = Elasticsearch()
es = Elasticsearch([{'host': '192.168.1.29', 'port': 9200}])
# es = Elasticsearch([{'host': '172.27.228.44', 'port': 9200}])

es_index = 'pigeon-test'
es_image = 'pigeon-image-test2'

# es_index = 'pigeon-recoed-test3'
# es_image = 'pigeon-image-test3'

# es_index = 'pre-data'
# es_image = 'pre-image'

# import keras
import keras

# import keras_retinanet
# from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from keras_retinanet.models import load_model
import pytz
from tzlocal import get_localzone

class Model:
    def __init__(self, confidence=0.5):

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
        self.min_side4train = 720
        self.max_side4train = 900

        # Size image for Save 2 elasticsearch
        self.min_side4store = 600
        self.max_side4store = 800

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
        body = {}
        body['orginal_image'] = self.img2string(image)
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
        es.index(index=es_image, id=eventid, body=body)

    def elas_record(self, eventid, time_, label, score, box):
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
        body['box'] = {'x1':box[0], 'y1':box[1], 'x2':box[2], 'y2':box[3]}


        # print(body)
        es.index(index=es_index, body=body)

    def test_video(self):
        cap = cv2.VideoCapture('video/pigeon-12.mp4')
        # cap = cv2.VideoCapture(0)

        currentFrame = 0
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # frame, scale = resize_image(frame, max_side=800)

            if ret:
                # print(frame.shape)
                # image = read_image_bgr(frame)

                # copy to draw on
                draw = frame.copy()
                # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

                # preprocess image for network
                image = preprocess_image(frame)
                image, scale = resize_image(image, min_side=self.min_side4train, max_side=self.max_side4train)

                time_ = self.gen_datetime()
                eventid = time_.strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
                print('imgae_id', eventid)

                # elas_image(image=image, eventid=eventid, scale=scale, time_=time_)

                # process image
                start = time.time()
                boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
                processing_time = time.time() - start
                print("processing time: ", processing_time)

                # correct for image scale
                boxes /= scale
                found_ = {}
                # visualize detections
                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    # scores are sorted so we can break
                    if score < self.confThreshold:
                        break
                    color = label_color(label)

                    b = box.astype(int)

                    draw_box(draw, b, color=color)

                    caption = "{} {:.3f}".format(self.labels_to_names[label], score)
                    print(b, caption)
                    draw_caption(draw, b, caption)
                    box = [np.ushort(x).item() for x in box]

                    try:
                        found_[self.labels_to_names[label]] += 1
                    except:
                        found_[self.labels_to_names[label]] = 1


                    self.elas_record(eventid=eventid, time_=time_, label=label, score=np.float32(score).item(), box=box, processing_time=processing_time)


                d2 = draw.copy()
                d2, s2 = resize_image(d2, min_side=self.min_side4store, max_side=self.max_side4store)
                print(d2.shape, s2)

                self.elas_image(image=d2, eventid=eventid, scale=scale, time_=time_, found_= found_)
                print(draw.shape)

                cv2.imshow('frame', draw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def test_image(self, image_path= None):
        if image_path is None:
            image_path = 'image/ca.jpg'
        image = read_image_bgr(image_path)

        # copy to draw on
        draw = image.copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        print(draw.shape)
        # preprocess image for network
        image = preprocess_image(draw)
        image, scale = resize_image(image, min_side=720, max_side=1280)
        print(image.shape)
        # time_ = gen_datetime()
        time_ = datetime.now()

        eventid = time_.strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        print('imgae_id', eventid)
        # self.elas_image(image=draw, eventid=eventid, scale=scale, time_=time_)

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        processing_time = time.time() - start
        print("processing time: ", processing_time)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < self.confThreshold:
                break

            color = label_color(label)

            b = box.astype(int)

            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            # print(b, caption)
            draw_caption(draw, b, caption)
            box = [np.float32(x).item() for x in box]
            self.elas_record(eventid=eventid, time_=time_, label=label, score=np.float32(score).item(), box=box,
                        processing_time=processing_time)

        cv2.imshow('result', draw)
        cv2.waitKey()

    def detect(self, image):
        # print(frame.shape)
        # image = read_image_bgr(frame)

        # copy to draw on
        draw = image.copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=self.min_side4train, max_side=self.max_side4train)

        time_ = self.gen_datetime()
        # time_ = datetime.now()

        eventid = time_.strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        # print('imgae_id', eventid)
        # print('time', time_)
        # print('local', self.localtimezone(time_))
        # print('utc',  self.utctimezone(time_))
        # elas_image(image=image, eventid=eventid, scale=scale, time_=time_)

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        processing_time = time.time() - start
        # print("processing time: ", processing_time)

        # correct for image scale
        boxes /= scale
        found_ = {}

        main_body = {'eventid':eventid,  'time_':time_}

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < self.confThreshold:
                break
            color = label_color(label)

            b = box.astype(int)

            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            # print(b, caption)
            draw_caption(draw, b, caption)
            box = [np.ushort(x).item() for x in box]

            try:
                found_[self.labels_to_names[label]] += 1
            except:
                found_[self.labels_to_names[label]] = 1

            self.elas_record(label=label, score=np.float32(score).item(), box=box, **main_body)

        d2 = draw.copy()
        d2, s2 = resize_image(d2, min_side=self.min_side4store, max_side=self.max_side4store)
        # print(d2.shape, s2)

        self.elas_image(image=d2, scale=scale, found_=found_, processing_time=processing_time, **main_body)
        # print(draw.shape)

        return d2

    def setConfidence(self, confidence):
        self.confThreshold = float(confidence)
