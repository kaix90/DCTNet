# Learning in the Frequency Domain
Classification on ImageNet with PyTorch.

## Prerequisites
* PyTorch compatible GPU
* Python 3.7
* PyTorch >= 1.2.0
* opencv-python 4.1.1
* libjpeg-turbo 2.0.3
* [jpeg2dct](https://github.com/uber-research/jpeg2dct)

## Install
* Install [PyTorch](http://pytorch.org/)

* Clone this repo recursively
  ```
  git clone --recursive https://github.com/calmevtime1990/supp
  ```
  
* Install required packages
  ```
  pip install -r requirements.txt
  ```
  
* Install [libjpeg-turbo](http://www.linuxfromscratch.org/blfs/view/svn/general/libjpeg.html)
  ```
  bash install_libjpegturbo.sh
  ```

* Download pretrained [models][1] and extract to [`pretrained`](pretrained). The folder structure should look like this:
```
pretrained
├── resnet50dct_upscaled_static_24
│   ├── log.txt
│   └── model_best.pth.tar
└── resnet50dct_upscaled_static_64
    ├── log.txt
    └── model_best.pth.tar
```
* Prepare datasets
It is recommended to symlink the dataset root to [`data`](data). The folder structure should look like this:
```
data
├── train
├── val
└── README.md
```

## Evaluation
Run [`resnet_upscaled_static.sh`](scripts/resnet_upscaled_static.sh) to start testing. Change the --data $imagenet_dir to the location of the ImageNet dataset.
### Testing the proposed model with 24 channels
```
bash scripts/resnet_upscaled_static.sh 24
```

### Testing the proposed model with 64 channels
```
bash scripts/resnet_upscaled_static.sh 64
```

## Results
### Performance of the proposed model - ResNet-50 
|    ResNet-50   | #Channels | Size Per Channel |  Top-1 |  Top-5 | Normalized Input Size |
|:--------------:|:---------:|:----------------:|:------:|:------:|:---------------------:|
|       RGB      |     3     |      224x224     | 75.780 | 92.650 |          1.0          |
| [DCT-24  (ours)][2] |     24    |       56x56      | 76.792 | 93.254 |          0.5          |
| [DCT-64  (ours)][3] |     64    |       56x56      | 77.160 | 93.474 |          1.3          |

### Performance of the proposed model - MobileNetV2
|  MobileNetV2  | #Channels | Size Per Channel |  Top-1 |  Top-5 |
|:-------------:|:---------:|:----------------:|:------:|:------:|
|      RGB      |     3     |      224x224     | 71.702 | 90.415 |
| [DCT-24 (ours)][4] |     24    |      112x112     | 72.364 | 90.606 |
| [DCT-32 (ours)][5] |     32    |      112x112     | 72.282 | 90.592 |

[1]: https://drive.google.com/open?id=1Wl9JDY1zJPtkcRBpVojQ9ERtxul7k9rt
[2]: https://drive.google.com/open?id=1eZosbUN1-MwdIOk6ANrca049kztV-Gv2
[3]: https://drive.google.com/open?id=1eluAU8gngxMFL7ti73qv7vyrTujhGZA9
[4]: https://drive.google.com/open?id=1wkAFUDDj6wUVKLuFjlPvvaHTOsROOaRk
[5]: https://drive.google.com/open?id=1QaDqumx9j4cGgcJuQVzYSIPiaClOKZj3
