import torch
from collections import OrderedDict

file_path = '../../pretrained/model_best.pth.tar'
checkpoint = torch.load(file_path)['state_dict']
new_checkpoint = OrderedDict()

prefix = ['model.0', 'model.1', 'model.2', 'model.3']
for k, v in checkpoint.items():
    if any(word in k for word in prefix):
        k = k[7:]
        new_k = k.replace('model.' + k.split('.')[1], 'layer' + str(int(k.split('.')[1]) + 1))
        new_checkpoint[new_k] = v

torch.save(new_checkpoint, '../../pretrained/resnet50_192.pth.tar')

