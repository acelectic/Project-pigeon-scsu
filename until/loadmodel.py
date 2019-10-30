import os
from keras.utils import get_file

def load_data4eval():
    
    model_store = os.getcwd()
    resnet_filename = 'data4eval.zip'
    resnet_resource = 'https://github.com/acelectic/Project-pigeon-scsu/releases/download/v1.1/data4eval.zip'

    get_file(
        resnet_filename,
        resnet_resource,
        cache_subdir=model_store,
    )


def download_imagenet():
    """ Downloads ImageNet weights and returns path to weights file.
    """

    model_store = os.getcwd() + '/models/resnet50'
    resnet_filename = 'infer-resnet50.h5'
    resnet_resource = 'https://github.com/acelectic/Project-pigeon-scsu/releases/download/v1.1/model-infer-merge-resnet50.h5'

    get_file(
        resnet_filename,
        resnet_resource,
        cache_subdir=model_store,
    )

    model_store = os.getcwd() + '/models/resnet101'
    resnet_filename = 'infer-resnet101.h5'
    resnet_resource = 'https://github.com/acelectic/Project-pigeon-scsu/releases/download/v1.1/model-infer-merge-resnet101.h5'

    get_file(
        resnet_filename,
        resnet_resource,
        cache_subdir=model_store,
    )

    return


download_imagenet()
load_data4eval()