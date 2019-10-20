import time
from datetime import datetime
from uuid import uuid4

import glob
import os

import numpy as np
import radar
# set tf backend to allow memory to grow, instead of claiming everything
from keras_retinanet.models import load_model
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption

import cv2

# thai_timezone = pytz.timezone('Asia/Bangkok')


class Model:
    def __init__(self, confidence=0.5, es=None, es_mode=False):

        # Size image for train on retinenet
        self.min_side4train = 600
        self.max_side4train = 600

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
            self.es_status = False
            print("can't connect to es")
        else:
            self.es_status = True
            self.es = es
            print('connect es')

        self.es_mode = es_mode

        self.confThreshold = float(confidence)

        # try:
        #     model_path = os.getcwd() + '/retinanet/model/resnet50_coco_best_v2.1.0.h5'
        #     self.model = load_model(
        #         model_path,
        #         backbone_name='resnet50')
        # except:
        #     model_path = os.getcwd() + '/model/resnet50_coco_best_v2.1.0.h5'
        #     self.model = load_model(
        #         model_path,
        #         backbone_name='resnet50')
        # model_path = os.getcwd() + '/model/pigeon_resnet50_midway.h5'

        self.model = load_model(
            '/home/minibear-l/Desktop/pre-data_script/evalresult/model-infer.h5', backbone_name='resnet50')

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

        # time_ = self.time2store
        time_ = datetime.now()

        eventid = time_.strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

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

        main_body = {'eventid': eventid, 'time_': time_}
        self.__data4turret = {
            "box": box4turret[0][0],
            "score": scores[0][0],
            "label": labels[0][0],
            "scale": scale,
            "raw_image-shape": draw.shape}

        # visualize detections

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

            caption = "{} {:.3f} {}".format(
                self.labels_to_names[label], score, box)
            print(caption)
            draw_caption(img4elas, b, caption)
            temp_data.append([self.labels_to_names[label], score, b, processing_time])
        return temp_data

    def setConfidence(self, confidence):
        print("confident: {} ---> {}".format(self.confThreshold, confidence))
        self.confThreshold = float(confidence)


if __name__ == '__main__':

    model = Model()
    result_detect = {}
    for img_path in glob.glob('data4eval/test/*.png')[:]:
        img_name = img_path.split(',')[0].split('/')[-1]
        print(img_name)

        img = cv2.VideoCapture(img_path)
       
        _, frame = img.read()

        if _:
            result = model.detect(frame)
            print(result)
            for label, score, box, processing_time in result:
                try:
                    result_detect[img_name] += [{
                        'label': label,
                        'score' : score,
                        'processing_time': processing_time,
                        'box': (box[0], box[1], box[2], box[3])
                    }]
                except:
                    result_detect[img_name] = [{
                        'label': label,
                        'score': score,
                        'processing_time': processing_time,
                        'box': (box[0], box[1], box[2], box[3])
                    }]
    detect_dir = 'data4eval/test/detections'
    os.makedirs(detect_dir, exist_ok=True)

    for key, data in result_detect.items():
        # print(key, data)
        with open(detect_dir+ '/' + key.replace('.png', '.txt'), 'w') as f:
            for data_2 in data:
                # print(data_2)
                f.write(data_2['label'] + ' ' +  '{:.6f} '.format(data_2['score']) +' '.join(map(str, data_2['box'])) + '\n')
    
