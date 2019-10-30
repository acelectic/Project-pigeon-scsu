import base64

import cv2
import pytz
from elasticsearch import Elasticsearch
from tzlocal import get_localzone


class Elas_api:

    def __init__(self, ip=None, port=9200):

        self.ip = ip
        self.port = port

        self.preSomething()

        self._connectElas(ip=self.ip, port=self.port)
        # print(self.checkStatus())



    def preSomething(self):

        self.es_index = 'pigeon_data'
        self.es_image = 'pigeon_image'
        self.es_date = 'pigeon_date'

        self.es_index = 'jetson_data'
        self.es_image = 'jetson_image'
        self.es_date = 'jetson_date'

        self.weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

        self.hour_text = ["00.00-01.00", "01.00-02.00", "02.00-03.00", "03.00-04.00", "04.00-05.00", "05.00-06.00",
                          "06.00-07.00", "07.00-08.00", "08.00-09.00", "09.00-10.00", "10.00-11.00", "11.00-12.00",
                          "12.00-13.00", "13.00-14.00", "14.00-15.00", "15.00-16.00", "16.00-17.00", "17.00-18.00",
                          "18.00-19.00", "19.00-20.00", "20.00-21.00", "21.00-22.00", "22.00-23.00", "23.00-24.00"]

        self.months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                       'August', 'September', 'October', 'November', 'December']

        self.config = {
            'retry_on_timeout': False,
            'timeout': 10,
            'max_retries': 0,

        }

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

    def __connectElasNikko(self):
        config = self.config
        try:
            self.es = Elasticsearch([{'host': '192.168.1.29', 'port': 9200}], **config)
        # self.es = Elasticsearch([{'host': '172.27.228.44', 'port': 9200}])
        except:
            print("Can't conection nikko")

    def _connectElas(self, ip=None, port=9200):
        config = self.config
        if ip:
            assert ip is not str, 'Please Fill ip with String!!!'
            try:
                self.es = Elasticsearch([{'host': ip, 'port': port}], **config)
                print('Connect to Elasticsearch, {}:{}'.format(ip, port))
            except Exception as e:
                print(e)
        else:
            try:
                self.es = Elasticsearch(**config)
                print('Connect to local Elasticsearch')
            except Exception as e:
                print(e)

    def checkStatus(self):
        if not self.es.ping():
            # raise ValueError("Connection failed")
            self.es_status = False
        else:
            self.es_status = True

        return self.es_status, self.es.info
    
    def setElasIndex(self, backbone):
        self.es_index = 'jetson_{}_data'.format(backbone)
        self.es_image = 'jetson_{}_image'.format(backbone)

    def elas_image(self, image_id, image, scale, time_, found_, processing_time):
        if self.es_status:
            body = {}
            body['original_img'] = self.img2string(image)
            body['scale'] = scale

            body['timestamp'] = self.localtimezone(time_)
            # body['timestamp_utc'] = self.utctimezone(time_)

            body['Hour_int'] = self.hour_int(time_)
            body['Hour_text'] = self.hour_text[body['Hour_int']]

            body['dayofweek_int'] = self.dayofweek_int(time_)
            body['dayofweek_text'] = self.dayofweek_text(time_)

            body['Month_int'] = self.month_int(time_)
            body['Mouth_text'] = self.month_text(time_)

            body['birds_count'] = found_

            body['processing_time'] = processing_time
            print('insert image {}\n'.format(time_))
            self.es.index(index=self.es_image, doc_type="_doc", id=image_id, body=body)

    def elas_record(self, image_id, time_, label, score, box):
        if self.es_status:
            body = {}
            body['timestamp'] = self.localtimezone(time_)
            # body['timestamp_utc'] = self.utctimezone(time_)

            body['Hour_int'] = self.hour_int(time_)
            body['Hour_text'] = self.hour_text[body['Hour_int']]

            body['dayofweek_int'] = self.dayofweek_int(time_)
            body['dayofweek_text'] = self.dayofweek_text(time_)

            body['Month_int'] = self.month_int(time_)
            body['Mounh_text'] = self.month_text(time_)

            body['image_id'] = image_id
            # body['found'] = {self.labels_to_names[label]: 1}
            body['confidence'] = score
            body['box'] = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}


            print('insert index {}\n{}\n'.format(time_, body))
            self.es.index(index=self.es_index, doc_type="_doc", body=body)


    def hour_int(self, time):
        return time.hour

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
