import torch.nn as nn
# from .utils import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch
import numpy as np
from models.utils import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'ResNetDCT_Upscaled_Static', 'ResNet50DCT']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        # Change the stride from 2 to 1 to match the input size
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)


    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

    # model = nn.Sequential(*list(model.children())[:])

    return model

class ResNet50DCT(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50DCT, self).__init__()
        model = resnet50(pretrained=pretrained)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_ch, out_ch = 64, 96
        self.model = nn.Sequential(*list(model.children())[5:-1])
        self.fc = list(model.children())[-1]
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dct_y, dct_cb, dct_cr):
        dct_cb = self.deconv1(dct_cb)
        dct_cr = self.deconv2(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.model(x)
        x = x.reshape(x.size(0), -1)  # 2048
        x = self.fc(x)
        return x

        # x = self.layer1(x)  # 256x56x56
        # x = self.layer2(x)  # 512x28x28
        # x = self.layer3(x)  # 1024x14x14
        # x = self.layer4(x)  # 2048x7x7
        #
        # x = self.avgpool(x)  # 2048x1x1
        # x = x.reshape(x.size(0), -1)  # 2048
        # x = self.fc(x)  # 1000

        # return x

class ResNet50DCT_Upscaled(nn.Module):
    def __init__(self):
        super(ResNet50DCT_Upscaled, self).__init__()
        model = resnet50(pretrained=True)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.model = nn.Sequential(*list(model.children())[4:-1])
        self.fc = list(model.children())[-1]

        upscale_factor_y, upscale_factor_cb, upscale_factor_cr = 1, 2, 2
        self.upconv_y = nn.Conv2d(in_channels=64, out_channels=22*upscale_factor_y*upscale_factor_y,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.upconv_cb = nn.Conv2d(in_channels=64, out_channels=21*upscale_factor_cb*upscale_factor_cb,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.upconv_cr = nn.Conv2d(in_channels=64, out_channels=21*upscale_factor_cr*upscale_factor_cr,
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_y = nn.BatchNorm2d(22)
        self.bn_cb = nn.BatchNorm2d(21)
        self.bn_cr = nn.BatchNorm2d(21)
        # self.pixelshuffle_y = nn.PixelShuffle(1)
        self.pixelshuffle_cb = nn.PixelShuffle(upscale_factor_cb)
        self.pixelshuffle_cr = nn.PixelShuffle(upscale_factor_cr)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        # initialize input layers
        for name, m in self.named_modules():
            if any(s in name for s in ['upconv_y', 'upconv_cb', 'upconv_cr', 'bn_y', 'bn_cb', 'bn_cr']):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, dct_y, dct_cb, dct_cr):
        # dct_y = self.relu(self.bn_y(self.pixelshuffle_y(self.upconv_y(dct_y))))
        dct_y = self.relu(self.bn_y(self.upconv_y(dct_y)))
        dct_cb = self.relu(self.bn_cb(self.pixelshuffle_cb(self.upconv_cb(dct_cb))))
        dct_cr = self.relu(self.bn_cr(self.pixelshuffle_cr(self.upconv_cr(dct_cr))))
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.model(x)
        x = x.reshape(x.size(0), -1)  # 2048
        x = self.fc(x)
        return x

class ResNetDCT_Upscaled_Static(nn.Module):
    def __init__(self, channels=0, pretrained=True, input_gate=False):
        super(ResNetDCT_Upscaled_Static, self).__init__()

        self.input_gate = input_gate

        model = resnet50(pretrained=pretrained)

        self.model = nn.Sequential(*list(model.children())[4:-1])
        self.fc = list(model.children())[-1]
        self.relu = nn.ReLU(inplace=True)

        if channels == 0 or channels == 192:
            out_ch = self.model[0][0].conv1.out_channels
            self.model[0][0].conv1 = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(self.model[0][0].conv1)

            out_ch = self.model[0][0].downsample[0].out_channels
            self.model[0][0].downsample[0] = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(self.model[0][0].downsample[0])

            # temp_layer = conv3x3(channels, out_ch, 1)
            # temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            # temp_layer.weight.data = self.model[0][0].conv1.weight.data.repeat(1, 3, 1, 1)
            # self.model[0][0].conv1 = temp_layer

            # out_ch = self.model[0][0].downsample[0].out_channels
            # temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            # temp_layer.weight.data = self.model[0][0].downsample[0].weight.data.repeat(1, 3, 1, 1)
            # self.model[0][0].downsample[0] = temp_layer
        elif channels < 64:
            out_ch = self.model[0][0].conv1.out_channels
            # temp_layer = conv3x3(channels, out_ch, 1)
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=3, stride=1, bias=False)
            temp_layer.weight.data = self.model[0][0].conv1.weight.data[:, :channels]
            self.model[0][0].conv1 = temp_layer

            out_ch = self.model[0][0].downsample[0].out_channels
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            temp_layer.weight.data = self.model[0][0].downsample[0].weight.data[:, :channels]
            self.model[0][0].downsample[0] = temp_layer

        if input_gate:
            self.inp_GM = GateModule192()
            self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if 'inp_gate_l' in str(name):
                m.weight.data.normal_(0, 0.001)
                m.bias.data[::2].fill_(0.1)
                m.bias.data[1::2].fill_(2)
            elif 'inp_gate' in str(name):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        if self.input_gate:
            x, inp_atten = self.inp_GM(x)

        x = self.model(x)
        x = x.reshape(x.size(0), -1)  # 2048
        x = self.fc(x)
        if self.input_gate:
            return x, inp_atten
        else:
            return x

def main():
    import numpy as np

    channels = 192
    model_resnet50dct = ResNetDCT_Upscaled_Static(channels=channels)
    input = torch.from_numpy(np.random.randn(16, channels, 56, 56)).float()
    x = model_resnet50dct(input)
    print(x.shape)

    model_resnet50dct = resnet50(pretrained=True)
    input = torch.from_numpy(np.random.randn(16, 3, 224, 224)).float()
    x = model_resnet50dct(input)
    print(x.shape)

    # ResNet50DCT
    model_resnet50dct = ResNet50DCT()

    dct_y  = torch.from_numpy(np.random.randn(16, 64, 28, 28)).float()
    dct_cb = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()
    dct_cr = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()

    x = model_resnet50dct(dct_y, dct_cb, dct_cr)
    print(x.shape)

    # SE-ResNet50DCT
    model_seresnet50dct = SE_ResNet50DCT()
    x = model_seresnet50dct(dct_y, dct_cb, dct_cr)
    print(x.shape)

if __name__ == '__main__':
    main()
