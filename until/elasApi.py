import cv2
from elasticsearch import Elasticsearch


class elas_api:
    index_bbox = 'project-bbox'
    index_image = 'project-image'

    doc_type_bbox = 'box'
    doc_type_image = 'image'

    def __init__(self, ip=None, port=9200):
        # print('create elas api')
        # if ip:
        #     self.es = Elasticsearch([{'host': ip, 'port': port}])
        #     print('connect:', ip)
        # else:
        #     self.es = Elasticsearch()
        #     print('connect local')

        try:
            if ip:
                self.es = Elasticsearch([{'host': ip, 'port': port}])
                print('connect:', ip)
            else:
                self.es = Elasticsearch()
                print('connect local')

        except Exception as e:
            print(e)
            return None


    def putData(self, index, data, id=None):
        res = self.es.index(index=index, doc_type=self.doc_type_image, body=data)


    # def getAllimage_id(self):
    #     doc = {
    #         # 'size': 10000,
    #         # 'query': {
    #         #     'match_all': {}
    #         # },
    #         "stored_fields": []
    #     }
    #     out = []
    #     res = self.es.search(index=self.index_image, doc_type=self.doc_type_image, body=doc, scroll='1m')
    #     scroll = res['_scroll_id']
    #     for i in res:
    #         print(i)
    #     for i in res['hits']['hits']:
    #         print(i['_id'])
    #         out += i['_source']['orginal_image']
    #     return out

# el = elas_api()

