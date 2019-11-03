import os

os.environ['MODEL_RESNET50'] = 'models/resnet50/infer-resnet50.h5'
print(os.environ['MODEL_RESNET50'])

os.environ['MODEL_RESNET101'] = 'models/resnet101/infer-resnet101.h5'
print(os.environ['MODEL_RESNET101'])

os.environ['MODEL_cRESNET50'] = 'models/resnet50/infer-resnet50-cAnchor.h5'
print(os.environ['MODEL_cRESNET50'])

os.environ['MODEL_cRESNET101'] = 'models/resnet101/infer-resnet101-cAnchor.h5'
print(os.environ['MODEL_cRESNET101'])
