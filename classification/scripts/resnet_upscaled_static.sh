#!/usr/bin/env bash
subset=$1
subset=${subset:-0}
echo "subset: $subset"
python main/imagenet_resnet_upscaled_static.py -j 16 --gpu-id 0,1,2,3 --arch ResNetDCT_Upscaled_Static --subset $subset --resume pretrained/resnet50dct_upscaled_static_$subset/model_best.pth.tar --data ./data --evaluate