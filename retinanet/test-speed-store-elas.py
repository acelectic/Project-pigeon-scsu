

import glob
import os
import sys
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

# thai_timezone = pytz.timezone('Asia/Bangkok')

import set_model2environ


class Model:
    def __init__(self, confidence=0.5, es=None, es_mode=False, model_is='resnet50'):

        # Size image for Save 2 elasticsearch
        self.min_side4elas = 600
        self.max_side4elas = 600

        # print('model confidence:', self.confThreshold)

        # # es = Elasticsearch()
        # self.es = Elasticsearch([{'host': '192.168.1.29', 'port': 9200}])
        # # es = Elasticsearch([{'host': '172.27.228.44', 'port': 9200}])
        #
        if es == None or es.checkStatus() == False:
            # raise ValueError("Connection failed")
            self.es = None
            self.es_status = False
            print("can't connect to es")
            self.es_mode = False
        else:
            self.es_status = True
            self.es = es
            print('connect es')
            self.es_mode = True

        self.confThreshold = float(confidence)

        # try:model_
        #     model_path = os.getcwd() + '/retinanet/model/resnet50_coco_best_v2.1.0.h5'
        #     self.model = load_model(
        #         model_path,
        #         backbone_name='resnet50')
        # except:
        #     model_path = os.getcwd() + '/fldel/resnet50_coco_best_v2.1.0.h5'
        #     self.model = load_model(
        #         model_path,
        #         backbone_name='resnet50')fl
        # model_path = os.getcwd() + '/model/pigeon_resnet50_midway.h5'
        resnet50_dir = os.environ['MODEL_RESNET50']
        resnet101_dir = os.environ['MODEL_RESNET101']
        if model_is == 'resnet50':

            # Size image for train on retinenet
            if self.es != None:
                self.es.setElasIndex(model_is)

            self.min_side4train = 700
            self.max_side4train = 700
            self.model = load_model(
                resnet50_dir, backbone_name='resnet50')
        elif model_is == 'resnet101':
            if self.es != None:
                self.es.setElasIndex(model_is)
            # Size image for train on retinenet
            self.min_side4train = 400
            self.max_side4train = 400
            self.model = load_model(
                resnet101_dir, backbone_name='resnet101')

        self.labels_to_names = {0: 'pigeon'}

    def gen_datetime(self, from_date, to_date):

        from_date = from_date.split('-')

        return radar.random_date(
            start=datetime(year=from_date.year,
                           month=from_date.month, day=from_date.day),
            stop=datetime(year=to_date.year, month=to_date.month, day=to_date.day))

    def gen_datetime(self):
        return radar.random_date(
            start=datetime(year=2018, month=1, day=1),
            stop=datetime(year=2019, month=10, day=26))

    def detect(self, image):

        self.time2store = self.gen_datetime()
        # self.time2store = datetime.now()

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
        # print("processing time: ", processing_time)

        img4elas, scale4elas = resize_image(
            draw, min_side=self.min_side4elas, max_side=self.max_side4elas)

        # correct for image scale
        # boxes /= scale
        box4turret = boxes/scale
        found_ = {}

        main_body = {'image_id': image_id, 'time_': time_}
        # visualize detections
        print('es_mode: {}\nstatus: {}'.format(self.es_mode, self.es_status))
        temp_data = []
        index = 1
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            # print(index, box, score, self.labels_to_names[label])
            if score < self.confThreshold:
                break

            color = label_color(label)

            b = box.astype(int)

            draw_box(img4elas, b, color=color)

            caption = "{} {:.3f}".format(
                self.labels_to_names[label], score)
            # print(caption)
            draw_caption(img4elas, b, caption)
            temp_data.append([self.labels_to_names[label],
                              score, b, processing_time])

            box = [np.ushort(x).item() for x in box]

            #  if self.es_mode and self.es_status:
            #     self.es.elas_record(label=label, score=np.float32(score).item(), box=box, image_id=image_id, time_=time_)
            # index += 1

            if self.es_mode and self.es_status and self.labels_to_names[label] == 'pigeon':
                # print('{tag}\n\n{data}\n\n{tag}'.format(
                #     tag='#'*20,
                #     data='label: {label}\nscore: {score}'.format(
                #         label=self.labels_to_names[label], score=score)
                # ))
                self.es.elas_record(label=label, score=np.float32(
                    score).item(), box=box, **main_body)
            index += 1

            try:
                found_[self.labels_to_names[label]] += 1
            except:
                found_[self.labels_to_names[label]] = 1
        try:

            if self.es_mode and self.es_status and found_['pigeon'] > 0:
                print('{tag}\n\n{data}\n\n{tag}'.format(
                    tag='#'*50,
                    data='id: {id}\nbird count: {found}'.format(
                        id=image_id, found=found_)))
                self.es.elas_image(image=img4elas, scale=scale, found_=found_,
                                   processing_time=processing_time, **main_body)
                # self.es.elas_date(**main_body)
        except Exception as e:
            print(e)

        return temp_data

    def setConfidence(self, confidence):
        print("confident: {} ---> {}".format(self.confThreshold, confidence))
        self.confThreshold = float(confidence)


def testResnet50(base_dir, es):
    print("{}\n\n{}\n\n{}".format('#'*30, 'Test Speed Resnet 50', '#'*30))
    base_dir = base_dir
    model = Model(model_is='resnet50', es=es)
    result_detect = {}
    avg_process_time = 0
    imgs_dir = glob.glob(base_dir+'/*.png')[:]

    for img_path in imgs_dir:
        img_name = img_path.split(',')[0].split('/')[-1]
        # print(img_name)

        img = cv2.VideoCapture(img_path)

        _, frame = img.read()

        if _:
            result = model.detect(frame)


def testResnet101(base_dir, es):
    print("{}\n\n{}\n\n{}".format('#'*30, 'Test Speed Resnet 101', '#'*30))
    base_dir = base_dir
    model = Model(model_is='resnet101', es=es)
    result_detect = {}
    avg_process_time = 0
    imgs_dir = glob.glob(base_dir+'/*.png')[:]

    for img_path in imgs_dir:
        img_name = img_path.split(',')[0].split('/')[-1]
        # print(img_name)

        img = cv2.VideoCapture(img_path)

        _, frame = img.read()

        if _:
            result = model.detect(frame)


if __name__ == '__main__':
    from elas_api4test import Elas_api

    es_ip = '192.168.1.29'
    es_port = 9200
    es = Elas_api(ip=es_ip)

    args = sys.argv[1:][0]
    base_dir = 'data4eval/test_merge/'
    print(args)

    if args == '1':
        testResnet50(base_dir=base_dir, es=es)
    elif args == '2':
        testResnet101(base_dir=base_dir, es=es)
