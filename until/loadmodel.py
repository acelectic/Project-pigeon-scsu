import os
from keras.utils import get_file


def download_imagenet():
    """ Downloads ImageNet weights and returns path to weights file.
    """

    model_store = os.getcwd() + '/models'
    resnet_filename = 'model-infer-neg50-epoch-20-loss_0.1431.h5.keras.h5'
    resnet_resource = 'https://github.com/acelectic/Project-pigeon-scsu/releases/download/1.0/model-infer-neg50-epoch-20-loss_0.1431.h5'

    return get_file(
        resnet_filename,
        resnet_resource,
        cache_subdir=model_store,
    )


download_imagenet()
