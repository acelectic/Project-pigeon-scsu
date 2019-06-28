# from tzlocal import get_localzone
#
# local_tz = get_localzone()
#
# import pytz
#
# print(local_tz)
# import radar
# from datetime import datetime
#
#
# def gen_datetime():
#     return radar.random_date(
#         start=datetime(year=2019, month=6, day=1),
#         stop=datetime(year=2019, month=6, day=11))
#
#
# def localtimezone(time):
#     local_tz = get_localzone()
#     timezone = pytz.timezone(str(local_tz))
#     return timezone.localize(time)
#
#
# def utctimezone(time):
#     # local_tz = get_localzone()
#     # return time.astimezone(local_tz)
#     try:
#         utc_tz = pytz.timezone('UTC')
#         return time.astimezone(utc_tz)
#     except:
#         utc_tz = pytz.timezone('UTC')
#         tmp_ = localtimezone(time)
#         return tmp_.astimezone(utc_tz)
#
#
# now = datetime.now()
# # now = gen_datetime()
#
# # print('now', now)
# # print(now.tzname())
# # tutc = datetime.now(tzutc())
# # print('utc',tutc)
# #
# # # time = datetime.now(tzlocal())
# # # print(time)
# # print(tutc.astimezone(local_tz))
#
#
# # print(now)
# print(now)
# sss = localtimezone(now)
# print(sss)
# # ssssc = now.astimezone()
# print(utctimezone(now))
#
# # class model_config:
# #     def __init__(self, confidence, detect_every_frame):
# #         self.__confidence = confidence
# #         self.__detect_every_frame = detect_every_frame
# #
# #     def setConfidence(self, confident):
# #         self.__confidence = confident
# #
# #     def getConfidence(self):
# #         return self.__confidence
# #
# #     def setDetect_every_frame(self, detect_every_frame):
# #         self.__detect_every_frame = detect_every_frame
# #
# #     def getDetect_every_frame(self):
# #         return self.__detect_every_frame
# #
# #     def __str__(self):
# #         return "con:{}\tfps{}".format(self.__confidence, self.__detect_every_frame)
# #
# # md = model_config(0.5, 5)
# #
# # print(md)
# #
# # md.setConfidence(0.3)
# # print(md)

from urllib.parse import quote, urlencode, quote_plus
data = {'tr':4522, 'opppoi':[45, 'asdas']}
a = quote("/go/", safe='', encoding='utf-16')
print(urlencode(data, quote_via=quote_plus))