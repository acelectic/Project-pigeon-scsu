import glob
import os
import time
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np
import radar
# set tf backend to allow memory to grow, instead of claiming everything
from keras_retinanet.models import load_model
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption

try:
    import set_model2environ
except:
    import retinanet.set_model2environ
# thai_timezone = pytz.timezone('Asia/Bangkok')

try:
    import silen
except:
    from retinanet import silen


class Model:
    def __init__(self, confidence=0.5, es=None, es_mode=False, cam_api=None, model_is='resnet50'):

        self.silen_ = silen.Silen_control()

        self._cam_api = cam_api

        # print('model confidence:', self.confThreshold)

        # # es = Elasticsearch()
        # self.es = Elasticsearch([{'host': '192.168.1.29', 'port': 9200}])
        # # es = Elasticsearch([{'host': '172.27.228.44', 'port': 9200}])
        #
        if es == None or es.checkStatus() == False:
            # raise ValueError("Connection failed")
            self.es_status = False
        else:
            self.es_status = True
            self.es = es

        self.es_mode = es_mode

        self.confThreshold = float(confidence)

        resnet50_dir = os.environ['MODEL_RESNET50']
        resnet101_dir = os.environ['MODEL_RESNET101']
        c_resnet50_dir = os.environ['MODEL_cRESNET50']
        c_resnet101_dir = os.environ['MODEL_cRESNET101']
        if model_is == 'resnet50':

            # Size image for train on retinenet
            if self.es != None:
                self.es.setElasIndex(model_is)

            self.min_side4train = 700
            self.max_side4train = 700

            self.min_side4elas = 700
            self.max_side4elas = 700

            self.model = load_model(
                resnet50_dir, backbone_name='resnet50')
        elif model_is == 'c_resnet50':

            # Size image for train on retinenet
            if self.es != None:
                self.es.setElasIndex(model_is)

            self.min_side4train = 700
            self.max_side4train = 700

            self.min_side4elas = 700
            self.max_side4elas = 700

            self.model = load_model(
                c_resnet50_dir, backbone_name='resnet50')

        elif model_is == 'resnet101':
            if self.es != None:
                self.es.setElasIndex(model_is)
            # Size image for train on retinenet
            self.min_side4train = 400
            self.max_side4train = 400

            self.min_side4elas = 400
            self.max_side4elas = 400

            self.model = load_model(
                resnet101_dir, backbone_name='resnet101')
        elif model_is == 'c_resnet101':
            if self.es != None:
                self.es.setElasIndex(model_is)
            # Size image for train on retinenet
            self.min_side4train = 400
            self.max_side4train = 400

            self.min_side4elas = 400
            self.max_side4elas = 400
            self.model = load_model(
                c_resnet101_dir, backbone_name='resnet101')

        # self.labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        #                         6: 'train',
        #                         7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
        #                         12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
        #                         18: 'sheep',
        #                         19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        #                         25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
        #                         31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        #                         35: 'baseball glove',
        #                         36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
        #                         41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
        #                         48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        #                         54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        #                         60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        #                         66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
        #                         71: 'sink',
        #                         72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
        #                         77: 'teddy bear',
        #                         78: 'hair drier', 79: 'toothbrush'}

        self.labels_to_names = {0: 'pigeon'}

    def gen_datetime(self, from_date, to_date):

        from_date = from_date.split('-')

        return radar.random_date(
            start=datetime(year=from_date.year,
                           month=from_date.month, day=from_date.day),
            stop=datetime(year=to_date.year, month=to_date.month, day=to_date.day))

    def gen_datetime(self):
        return radar.random_date(
            start=datetime(year=2019, month=1, day=1),
            stop=datetime(year=2019, month=7, day=23))

    def detect(self, image):
        # print('indetect')
        # cv2.imshow('s',image)
        # cv2.waitKey()
        # self.time2store = self.gen_datetime()
        self.time2store = datetime.now()

        self.cen_x = image.shape[1]//2
        self.cen_y = image.shape[0]//2

        # copy to draw on
        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)
        # cv2.imshow('ss22', image)
        image, scale = resize_image(
            image, min_side=self.min_side4train, max_side=self.max_side4train)

        time_ = self.time2store
        # time_ = datetime.now()

        image_id = time_.strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(
            np.expand_dims(image, axis=0))
        processing_time = time.time() - start

        print("processing time: ", processing_time)

        img4elas, scale4elas = resize_image(
            draw, min_side=self.min_side4elas, max_side=self.max_side4elas)

        # correct for image scale
        # boxes /= scale
        box4turret = boxes/scale
        found_ = {}

        main_body = {'image_id': image_id, 'time_': time_}

        self.__data4turret = {
            "box": box4turret[0][0],
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
            print(time_, '\t', caption)
            draw_caption(img4elas, b, caption)

            box = [np.ushort(x).item() for x in box]

            if self.labels_to_names[label] == 'pigeon':
                if self.es_mode and self.es_status:
                    self.es.elas_record(label=self.labels_to_names[label], score=np.float32(
                        score).item(), box=box, **main_body)
            index += 1

            try:
                found_[self.labels_to_names[label]] += 1
            except:
                found_[self.labels_to_names[label]] = 1

        print('#'*30)

        try:
            if found_['pigeon'] > 0:
                print("{e}".format(e=found_['pigeon']))
                save_imname = "pigeon_{}.png".format(image_id)
                cv2.imwrite("evalresult/test_alert/{}".format(save_imname), img4elas)
                self.silen_.run_alert()
                if self.es_mode and self.es_status:
                    self.es.elas_image(image=img4elas, scale=scale, found_=found_,
                                       processing_time=processing_time, **main_body)
                    # self.es.elas_date(**main_body)
        except Exception as e:
            print("{e}".format(e=e))
        print('#'*30)
        return 'Hello'

    def getCentroid(self, box):
        return (int((box[0] + box[2])/2), int((box[1] + box[3])/2))

    def setConfidence(self, confidence):
        print("confident: {} ---> {}".format(self.confThreshold, confidence))
        self.confThreshold = float(confidence)


