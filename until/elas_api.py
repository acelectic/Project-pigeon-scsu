import base64

import cv2
import pytz
from elasticsearch import Elasticsearch
from tzlocal import get_localzone


class elas_api:

    def __init__(self, ip=None, port=9200):

        self.ip = ip
        self.port = port

        self._connectElas(ip=self.ip, port=self.port)
        print(self.checkStatus())

        self.es_index = 'pigeon-test'
        self.es_image = 'pigeon-image-test2'

        # self.es_index = 'pigeon-recoed-test3'
        # self.es_image = 'pigeon-image-test3'

        # self.es_index = 'pre-data'
        # self.es_image = 'pre-

        self.weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

        self.hour_text = ["00.00-01.00", "01.00-02.00", "02.00-03.00", "03.00-04.00", "04.00-05.00", "05.00-06.00",
                          "06.00-07.00", "07.00-08.00", "08.00-09.00", "09.00-10.00", "10.00-11.00", "11.00-12.00",
                          "12.00-13.00", "13.00-14.00", "14.00-15.00", "15.00-16.00", "16.00-17.00", "17.00-18.00",
                          "18.00-19.00", "19.00-20.00", "20.00-21.00", "21.00-22.00", "22.00-23.00", "23.00-24.00"]

        self.months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                       'August', 'September', 'October', 'November', 'December']

    def __connectElasNikko(self):
        self.es = Elasticsearch([{'host': '192.168.1.29', 'port': 9200}], retry_on_timeout=True, timeout=10)
        # self.es = Elasticsearch([{'host': '172.27.228.44', 'port': 9200}])

    def _connectElas(self, ip=None, port=9200):
        if ip:
            assert ip is not str, 'Please Fill ip with String!!!'
            self.es = Elasticsearch([{'host': ip, 'port': port}])
            print('Connect to Elasticsearch, {}:{}'.format(ip, port))
        else:
            self.es = Elasticsearch()
            print('Connect to local Elasticsearch')

    def checkStatus(self):
        if not self.es.ping():
            # raise ValueError("Connection failed")
            self.es_status = False
        else:
            self.es_status = True

        return self.es_status, self.es.info

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
