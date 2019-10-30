import os 
cmd_resnet50 = "retinanet-evaluate --backbone 'resnet50' --iou-threshold 0.5 --score-threshold 0.5 --save-path 'evalresult/img-merge-resnet50/' --image-min-side 700 --image-max-side 700 csv data4eval/test_merge.csv data4eval/classes.txt models/resnet50/infer-resnet50.h5"

cmd_resnet101 = "retinanet-evaluate --backbone 'resnet101' --iou-threshold 0.5 --score-threshold 0.5 --save-path 'evalresult/img-merge-resnet101/' --image-min-side 400 --image-max-side 400 csv data4eval/test_merge.csv data4eval/classes.txt models/resnet101/infer-resnet101.h5"

os.system(cmd_resnet50)