<<<<<<< HEAD
import time
from datetime import datetime
from uuid import uuid4

import glob, os

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
    def __init__(self, confidence=0.5, es=None, es_mode=False, cam_api=None):
        self._cam_api = cam_api

        self.cen_x = 0
        self.cen_y = 0

        # Size image for train on retinenet
        self.min_side4train = 600
        self.max_side4train = 600

        # Size image for Save 2 elasticsearch
        self.min_side4elas = 600
        self.max_side4elas = 600

        # labels_to_names = {0: 'Pigeon'}
        # print('model confidence:', self.confThreshold)

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
            self.es = es

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
        # self.model = load_model(
        #     r'/home/minibear-l/Desktop/pre-data_script/store_models/test-neg/resnet101/default/snapshots/resnet101_20_loss-0.1521_val-loss-19.6176_mAP-0.0490.h5',
        #     backbone_name='resnet50')


        
        self.model = load_model(
            'models/model-infer-neg50-epoch-20-loss_0.1431.h5.keras.h5', backbone_name='resnet50')

        # self.model = load_model(
        #     '/home/minibear-l/Desktop/pre-data_script/evalresult/model-infer-neg101-epoch-20-loss_0.1521.h5', backbone_name='resnet101')


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

    def create_Tracker(self , bbox, frame):
        trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        def createTrackerByName(trackerType):
            # Create a tracker based on tracker name
            if trackerType == trackerTypes[0]:
                tracker = cv2.TrackerBoosting_create()
            elif trackerType == trackerTypes[1]:
                tracker = cv2.TrackerMIL_create()
            elif trackerType == trackerTypes[2]:
                tracker = cv2.TrackerKCF_create()
            elif trackerType == trackerTypes[3]:
                tracker = cv2.TrackerTLD_create()
            elif trackerType == trackerTypes[4]:
                tracker = cv2.TrackerMedianFlow_create()
            elif trackerType == trackerTypes[5]:
                tracker = cv2.TrackerGOTURN_create()
            elif trackerType == trackerTypes[6]:
                tracker = cv2.TrackerMOSSE_create()
            elif trackerType == trackerTypes[7]:
                tracker = cv2.TrackerCSRT_create()
            else:
                tracker = None
                print('Incorrect tracker name')
                print('Available trackers are:')
                for t in trackerTypes:
                    print(t)

            return tracker

        self.__old_cenTroid = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
        print('start: ', bbox, '\tcentroid:', self.__old_cenTroid)
        bboxes = [bbox]
        # Specify the tracker type
        trackerType = "CSRT"

        # Create MultiTracker object
        self.__multiTracker = cv2.MultiTracker_create()
        # Initialize MultiTracker
        for bbox in bboxes:
            self.__multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    def _updateTracker(self, frame):
        success, boxes = self.__multiTracker.update(frame)
        print('tracker{}'.format(boxes))
        # draw tracked objects
        for i, newbox in enumerate(boxes):

            # p1 = (int(newbox[0]), int(newbox[1]))
            # p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            self.__new_cenTroid = (int(newbox[0]) + int(newbox[2]/2), int(newbox[1])+ int(newbox[3]/2))
            print(self.__new_cenTroid)
            self.moveCamera(self.calMove(old_cenTroid=self.__old_cenTroid, new_cenTroid=self.__new_cenTroid))
        return self.stopmove(self.__new_cenTroid)
    def stopmove(self, new_cenTroid):
        stop_distance = 20
        new_x, new_y = new_cenTroid

        dis_x = abs(self.cen_x - new_x)
        dis_y = abs(self.cen_y - new_y)

        return 'Stop' if dis_x <= stop_distance and dis_y <= stop_distance else 'Continue'


    def moveCamera(self, command):
        cam_api = self._cam_api
        v, h = command
        if v == 'Left':
            cam_api.rotateLeft()
        elif v == 'Right':
            cam_api.rotateRight()

        if h == 'Up':
            cam_api.rotateUp()
        elif h == 'Down':
            cam_api.rotateDown()

    def calMove(self, old_cenTroid, new_cenTroid):
        old_x, old_y = old_cenTroid
        new_x, new_y = new_cenTroid

        return 'Left' if new_x > old_x else 'Right', 'Up' if new_y < old_y else 'Down'

    def gen_datetime(self, from_date, to_date):

        from_date = from_date.split('-')

        return radar.random_date(
            start=datetime(year=from_date.year, month=from_date.month, day=from_date.day),
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
        image, scale = resize_image(image, min_side=self.min_side4train, max_side=self.max_side4train)

        time_ = self.time2store
        # time_ = datetime.now()

        image_id = time_.strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        processing_time = time.time() - start
        print("processing time: ", processing_time)

        img4elas, scale4elas = resize_image(draw, min_side=self.min_side4elas, max_side=self.max_side4elas)

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
            # print(caption)
            draw_caption(img4elas, b, caption)
            
            box = [np.ushort(x).item() for x in box]

            #  if self.es_mode and self.es_status:
            #     self.es.elas_record(label=label, score=np.float32(score).item(), box=box, image_id=image_id, time_=time_)
            # index += 1

            if self.es_mode and self.es_status:
                self.es.elas_record(label=label, score=np.float32(score).item(), box=box, **main_body)
            index += 1

            try:
                found_[self.labels_to_names[label]] += 1
            except:
                found_[self.labels_to_names[label]] = 1


        

        if self.es_mode and self.es_status and found_ > 0:
            self.es.elas_image(image=img4elas, scale=scale, found_=found_, processing_time=processing_time, **main_body)
            # self.es.elas_date(**main_body)
        self.__updatelastFrame(img4elas)
        print('Head Shot')
        # return 'Now: {}\nDate: {}\nelas_id: {}\tbirds: {}\nProcess Time: {}\n{}'.format(datetime.now(), self.time2store,
        #                                                                       image_id, len(boxes), processing_time,
        #                                                                       '\n{:#>20} {} {:#<20}'.format('',
        #                                                                                                     'END 1 FRAME',
        #
        #                                                                                              ''))
        print("box[0][0]:{}".format(box4turret[0][0]))
        return self.box2tupple(box4turret[0][0])

    def box2tupple(self, box):
        return (box[0], box[1], abs(box[2]-box[0]), abs(box[3]-box[1]))

    def __shot(self, target_box):
        pass


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

    def testDetect(self):
        return 'num {}'.format(datetime.now())


import cv2
# from ..until.elas_api import elas_api


import uuid
if __name__ == '__main__':
    # es = elas_api(ip = '192.168.1.29')
    detect_model = Model(es_mode=False)

    # # img = cv2.VideoCapture(r"C:\Users\Kuy Loan\Desktop\Project-pigeon-scsu\vdo\video_25620705_061211.mp4")
    # img = cv2.VideoCapture(0)
    # while 1:
    #     _, frame = img.read()

    #     if _:
    #         detect_model.detect(frame)
    #         turretData = detect_model.getDataTurret()
    #         print(turretData)
    #         img_ = detect_model._getlastFrame()
    #         img_ = cv2.drawMarker(img_, turretData['centroid'], color=(255, 125, 128), markerSize=4)
    #         cv2.imshow('turret', img_)

    #         cv2.imwrite(str(uuid4())+'.png', img_) 
    #         # cv2.imshow('sss', frame)
    #         cv2.waitKey()
    for i in glob.glob('data4eval/test/*.png'):
        img = cv2.VideoCapture(i)

        _, frame = img.read()

        if _:
            detect_model.detect(frame)
            turretData = detect_model.getDataTurret()
            print(turretData)
            img_ = detect_model._getlastFrame()
            img_ = cv2.drawMarker(img_, turretData['centroid'], color=(255, 125, 128), markerSize=4)
            cv2.imshow('turret', img_)

            cv2.imwrite(str(uuid4())+'.png', img_) 
            # cv2.imshow('sss', frame)
=======
import time
from datetime import datetime
from uuid import uuid4

import glob, os

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
    def __init__(self, confidence=0.5, es=None, es_mode=False, cam_api=None):
        self._cam_api = cam_api

        self.cen_x = 0
        self.cen_y = 0

        # Size image for train on retinenet
        self.min_side4train = 600
        self.max_side4train = 600

        # Size image for Save 2 elasticsearch
        self.min_side4elas = 600
        self.max_side4elas = 600

        # labels_to_names = {0: 'Pigeon'}
        # print('model confidence:', self.confThreshold)

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
            self.es = es

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
        # self.model = load_model(
        #     r'/home/minibear-l/Desktop/pre-data_script/store_models/test-neg/resnet101/default/snapshots/resnet101_20_loss-0.1521_val-loss-19.6176_mAP-0.0490.h5',
        #     backbone_name='resnet50')


        
        # self.model = load_model(
        #     '/home/minibear-l/Desktop/pre-data_script/evalresult/model-infer-neg50-epoch-20-loss_0.1431.h5', backbone_name='resnet50')

        self.model = load_model(
            '/home/minibear-l/Desktop/pre-data_script/evalresult/model-infer-neg101-epoch-20-loss_0.1521.h5', backbone_name='resnet101')


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

    def create_Tracker(self , bbox, frame):
        trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        def createTrackerByName(trackerType):
            # Create a tracker based on tracker name
            if trackerType == trackerTypes[0]:
                tracker = cv2.TrackerBoosting_create()
            elif trackerType == trackerTypes[1]:
                tracker = cv2.TrackerMIL_create()
            elif trackerType == trackerTypes[2]:
                tracker = cv2.TrackerKCF_create()
            elif trackerType == trackerTypes[3]:
                tracker = cv2.TrackerTLD_create()
            elif trackerType == trackerTypes[4]:
                tracker = cv2.TrackerMedianFlow_create()
            elif trackerType == trackerTypes[5]:
                tracker = cv2.TrackerGOTURN_create()
            elif trackerType == trackerTypes[6]:
                tracker = cv2.TrackerMOSSE_create()
            elif trackerType == trackerTypes[7]:
                tracker = cv2.TrackerCSRT_create()
            else:
                tracker = None
                print('Incorrect tracker name')
                print('Available trackers are:')
                for t in trackerTypes:
                    print(t)

            return tracker

        self.__old_cenTroid = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
        print('start: ', bbox, '\tcentroid:', self.__old_cenTroid)
        bboxes = [bbox]
        # Specify the tracker type
        trackerType = "CSRT"

        # Create MultiTracker object
        self.__multiTracker = cv2.MultiTracker_create()
        # Initialize MultiTracker
        for bbox in bboxes:
            self.__multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    def _updateTracker(self, frame):
        success, boxes = self.__multiTracker.update(frame)
        print('tracker{}'.format(boxes))
        # draw tracked objects
        for i, newbox in enumerate(boxes):

            # p1 = (int(newbox[0]), int(newbox[1]))
            # p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            self.__new_cenTroid = (int(newbox[0]) + int(newbox[2]/2), int(newbox[1])+ int(newbox[3]/2))
            print(self.__new_cenTroid)
            self.moveCamera(self.calMove(old_cenTroid=self.__old_cenTroid, new_cenTroid=self.__new_cenTroid))
        return self.stopmove(self.__new_cenTroid)
    def stopmove(self, new_cenTroid):
        stop_distance = 20
        new_x, new_y = new_cenTroid

        dis_x = abs(self.cen_x - new_x)
        dis_y = abs(self.cen_y - new_y)

        return 'Stop' if dis_x <= stop_distance and dis_y <= stop_distance else 'Continue'


    def moveCamera(self, command):
        cam_api = self._cam_api
        v, h = command
        if v == 'Left':
            cam_api.rotateLeft()
        elif v == 'Right':
            cam_api.rotateRight()

        if h == 'Up':
            cam_api.rotateUp()
        elif h == 'Down':
            cam_api.rotateDown()

    def calMove(self, old_cenTroid, new_cenTroid):
        old_x, old_y = old_cenTroid
        new_x, new_y = new_cenTroid

        return 'Left' if new_x > old_x else 'Right', 'Up' if new_y < old_y else 'Down'

    def gen_datetime(self, from_date, to_date):

        from_date = from_date.split('-')

        return radar.random_date(
            start=datetime(year=from_date.year, month=from_date.month, day=from_date.day),
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
        image, scale = resize_image(image, min_side=self.min_side4train, max_side=self.max_side4train)

        time_ = self.time2store
        # time_ = datetime.now()

        image_id = time_.strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        processing_time = time.time() - start
        print("processing time: ", processing_time)

        img4elas, scale4elas = resize_image(draw, min_side=self.min_side4elas, max_side=self.max_side4elas)

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
            # print(caption)
            draw_caption(img4elas, b, caption)
            
            box = [np.ushort(x).item() for x in box]

            #  if self.es_mode and self.es_status:
            #     self.es.elas_record(label=label, score=np.float32(score).item(), box=box, image_id=image_id, time_=time_)
            # index += 1

            if self.es_mode and self.es_status:
                self.es.elas_record(label=label, score=np.float32(score).item(), box=box, **main_body)
            index += 1

            try:
                found_[self.labels_to_names[label]] += 1
            except:
                found_[self.labels_to_names[label]] = 1


        

        if self.es_mode and self.es_status and found_ > 0:
            self.es.elas_image(image=img4elas, scale=scale, found_=found_, processing_time=processing_time, **main_body)
            # self.es.elas_date(**main_body)
        self.__updatelastFrame(img4elas)
        print('Head Shot')
        # return 'Now: {}\nDate: {}\nelas_id: {}\tbirds: {}\nProcess Time: {}\n{}'.format(datetime.now(), self.time2store,
        #                                                                       image_id, len(boxes), processing_time,
        #                                                                       '\n{:#>20} {} {:#<20}'.format('',
        #                                                                                                     'END 1 FRAME',
        #
        #                                                                                              ''))
        print("box[0][0]:{}".format(box4turret[0][0]))
        return self.box2tupple(box4turret[0][0])

    def box2tupple(self, box):
        return (box[0], box[1], abs(box[2]-box[0]), abs(box[3]-box[1]))

    def __shot(self, target_box):
        pass


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

    def testDetect(self):
        return 'num {}'.format(datetime.now())


import cv2
# from ..until.elas_api import elas_api


import uuid
if __name__ == '__main__':
    # es = elas_api(ip = '192.168.1.29')
    detect_model = Model(es_mode=False)

    # # img = cv2.VideoCapture(r"C:\Users\Kuy Loan\Desktop\Project-pigeon-scsu\vdo\video_25620705_061211.mp4")
    # img = cv2.VideoCapture(0)
    # while 1:
    #     _, frame = img.read()

    #     if _:
    #         detect_model.detect(frame)
    #         turretData = detect_model.getDataTurret()
    #         print(turretData)
    #         img_ = detect_model._getlastFrame()
    #         img_ = cv2.drawMarker(img_, turretData['centroid'], color=(255, 125, 128), markerSize=4)
    #         cv2.imshow('turret', img_)

    #         cv2.imwrite(str(uuid4())+'.png', img_) 
    #         # cv2.imshow('sss', frame)
    #         cv2.waitKey()
    for i in glob.glob('data4eval/test/*.png'):
        img = cv2.VideoCapture(i)

        _, frame = img.read()

        if _:
            detect_model.detect(frame)
            turretData = detect_model.getDataTurret()
            print(turretData)
            img_ = detect_model._getlastFrame()
            img_ = cv2.drawMarker(img_, turretData['centroid'], color=(255, 125, 128), markerSize=4)
            cv2.imshow('turret', img_)

            cv2.imwrite(str(uuid4())+'.png', img_) 
            # cv2.imshow('sss', frame)
>>>>>>> final
            cv2.waitKey()