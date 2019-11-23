"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import math
from models.utils import *
from main import subset_channel_index

__all__ = ['mobilenetv2dct', 'mobilenetv2dct_deconv_subset',
           'mobilenetv2dct_subpixel', 'mobilenetv2dct_subpixel_subset',
           'mobilenetv2dct_upscaled', 'mobilenetv2dct_upscaled_subset',
           'mobilenetv2dct_subset_woinp', 'mobilenetv2dct_subset_woinp_from_scratch']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1., upscale=False):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        if not upscale:
            self.cfgs = [
                # t, c, n, s
                [1,  16, 1, 1],
                [6,  24, 2, 2],
                [6,  32, 3, 1],
                [6,  64, 4, 2],
                [6,  96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        else:
            self.cfgs = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(pretrained=True, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model =  MobileNetV2(**kwargs)
    if pretrained:
        state_dict = torch.load('./pretrained/mobilenetv2_1.0-0c6065bc.pth')
        model.load_state_dict(state_dict)

    return model

class MobileNetV2DCT(nn.Module):
    def __init__(self, upscale_ratio=1, channels=0):
        super(MobileNetV2DCT, self).__init__()

        self.upscale_ratio = upscale_ratio
        in_ch, out_ch = 64, 64

        if upscale_ratio == 1:
            model = mobilenetv2(pretrained=True, upscale=False)
            self.deconv_cb = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )
            self.deconv_cr = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )
        elif upscale_ratio == 2:
            model = mobilenetv2(pretrained=True, upscale=True)
            self.deconv_y = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )
            self.deconv_cb = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
            )
            self.deconv_cr = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )

        self.input_layer = nn.Sequential(
            # pw
            nn.Conv2d(3*in_ch, 3*out_ch, 3, 1, 1, bias=False, groups=3*in_ch),
            nn.BatchNorm2d(3*out_ch),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(3*out_ch, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24),
        )

        self._initialize_weights()

        if upscale_ratio == 1:
            self.features = nn.Sequential(*list(model.children())[0][4:])
        elif upscale_ratio == 2:
            self.features = nn.Sequential(*list(model.children())[0][3:])
        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, dct_y, dct_cb, dct_cr):
        if self.upscale_ratio == 1:
            dct_cb = self.deconv_cb(dct_cb)
            dct_cr = self.deconv_cr(dct_cr)
        elif self.upscale_ratio == 2:
            dct_y = self.deconv_y(dct_y)
            dct_cb = self.deconv_cb(dct_cb)
            dct_cr = self.deconv_cr(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.input_layer(x)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenetv2dct(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT(**kwargs)

    return model

class MobileNetV2DCT_Deconv_Subset(nn.Module):
    def __init__(self, upscale_ratio=1, channels=0):
        super(MobileNetV2DCT_Deconv_Subset, self).__init__()

        self.upscale_ratio = upscale_ratio
        if channels != 0:
            self.subset_channel_index = subset_channel_index

            y_in_ch = self.subset_channel_index[str(channels)][0]
            cb_in_ch = self.subset_channel_index[str(channels)][1]
            cr_in_ch = self.subset_channel_index[str(channels)][2]
        else:
            y_in_ch, cb_in_ch, cr_in_ch = 64, 64, 64
        y_out_ch, cb_out_ch, cr_out_ch = 64, 64, 64
        in_ch = y_out_ch + cb_out_ch + cr_out_ch
        out_ch = 192

        if upscale_ratio == 1:
            model = mobilenetv2(pretrained=True, upscale=False)
            self.conv_y = nn.Sequential(
                nn.Conv2d(in_channels=y_in_ch, out_channels=y_out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(y_out_ch),
                nn.ReLU6(inplace=True)
            )
            self.deconv_cb = nn.Sequential(
                nn.ConvTranspose2d(in_channels=cb_in_ch, out_channels=cb_out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cb_out_ch),
                nn.ReLU6(inplace=True)
            )
            self.deconv_cr = nn.Sequential(
                nn.ConvTranspose2d(in_channels=cr_in_ch, out_channels=cr_out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cr_out_ch),
                nn.ReLU6(inplace=True)
            )
        elif upscale_ratio == 2:
            model = mobilenetv2(pretrained=True, upscale=True)
            self.deconv_y = nn.Sequential(
                nn.ConvTranspose2d(in_channels=y_in_ch, out_channels=y_out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(y_out_ch),
                nn.ReLU6(inplace=True)
            )
            self.deconv_cb = nn.Sequential(
                nn.ConvTranspose2d(in_channels=cb_in_ch, out_channels=cb_out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cb_out_ch),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(in_channels=cb_out_ch, out_channels=cb_out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cb_out_ch),
                nn.ReLU6(inplace=True),
            )
            self.deconv_cr = nn.Sequential(
                nn.ConvTranspose2d(in_channels=cr_in_ch, out_channels=cr_out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cr_out_ch),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(in_channels=cr_out_ch, out_channels=cr_out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cr_out_ch),
                nn.ReLU6(inplace=True)
            )

        self.input_layer = nn.Sequential(
            # pw
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False, groups=in_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(out_ch, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24),
        )

        self._initialize_weights()

        if upscale_ratio == 1:
            self.features = nn.Sequential(*list(model.children())[0][4:])
        elif upscale_ratio == 2:
            self.features = nn.Sequential(*list(model.children())[0][3:])
        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, dct_y, dct_cb, dct_cr):
        if self.upscale_ratio == 1:
            dct_y = self.conv_y(dct_y)
            dct_cb = self.deconv_cb(dct_cb)
            dct_cr = self.deconv_cr(dct_cr)
        elif self.upscale_ratio == 2:
            dct_y = self.deconv_y(dct_y)
            dct_cb = self.deconv_cb(dct_cb)
            dct_cr = self.deconv_cr(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.input_layer(x)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenetv2dct_deconv_subset(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_Deconv_Subset(**kwargs)

    return model


class MobileNetV2DCT_Upscaled(nn.Module):
    def __init__(self):
        super(MobileNetV2DCT_Upscaled, self).__init__()

        model = mobilenetv2(pretrained=True, upscale=True)
        self.features = nn.Sequential(*list(model.children())[0][1:])
        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]
        upscale_factor_y, upscale_factor_cb, upscale_factor_cr = 1, 2, 2

        self.upconv_y = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=12 * upscale_factor_y * upscale_factor_y,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU6(inplace=True)
        )

        self.upconv_cb = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10 * upscale_factor_cb * upscale_factor_cb,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor_cb),
            nn.BatchNorm2d(10),
            nn.ReLU6(inplace=True)
        )

        self.upconv_cr = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10 * upscale_factor_cr * upscale_factor_cr,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor_cr),
            nn.BatchNorm2d(10),
            nn.ReLU6(inplace=True)
        )

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
        dct_y = self.upconv_y(dct_y)
        dct_cb = self.upconv_cb(dct_cb)
        dct_cr = self.upconv_cr(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenetv2dct_upscaled(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_Upscaled(**kwargs)

    return model


class MobileNetV2DCT_Upscaled_Subset(nn.Module):
    def __init__(self, upscale_ratio=False, channels=0):
        super(MobileNetV2DCT_Upscaled_Subset, self).__init__()

        model = mobilenetv2(pretrained=True, upscale=True)
        self.features = nn.Sequential(*list(model.children())[0][1:])
        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]
        upscale_factor_y, upscale_factor_cb, upscale_factor_cr = 1, 2, 2

        if channels != 0:
            self.subset_channel_index = subset_channel_index

            y_in_ch = self.subset_channel_index[str(channels)][0]
            cb_in_ch = self.subset_channel_index[str(channels)][1]
            cr_in_ch = self.subset_channel_index[str(channels)][2]
        else:
            y_in_ch, cb_in_ch, cr_in_ch = 64, 64, 64

        self.upconv_y = nn.Sequential(
            nn.Conv2d(in_channels=y_in_ch, out_channels=20 * upscale_factor_y * upscale_factor_y,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU6(inplace=True)
        )

        self.upconv_cb = nn.Sequential(
            nn.Conv2d(in_channels=cb_in_ch, out_channels=6 * upscale_factor_cb * upscale_factor_cb,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor_cb),
            nn.BatchNorm2d(6),
            nn.ReLU6(inplace=True)
        )

        self.upconv_cr = nn.Sequential(
            nn.Conv2d(in_channels=cr_in_ch, out_channels=6 * upscale_factor_cr * upscale_factor_cr,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor_cr),
            nn.BatchNorm2d(6),
            nn.ReLU6(inplace=True)
        )

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
        dct_y = self.upconv_y(dct_y)
        dct_cb = self.upconv_cb(dct_cb)
        dct_cr = self.upconv_cr(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenetv2dct_upscaled_subset(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_Upscaled_Subset(**kwargs)

    return model

class MobileNetV2DCT_Subpixel(nn.Module):
    def __init__(self, upscale_ratio=1, input_gate=False):
        super(MobileNetV2DCT_Subpixel, self).__init__()

        self.input_gate = input_gate

        if upscale_ratio == 1:
            model = mobilenetv2(pretrained=True, upscale=False)
            self.features = nn.Sequential(*list(model.children())[0][4:])
        elif upscale_ratio == 2:
            model = mobilenetv2(pretrained=True, upscale=True)
            self.features = nn.Sequential(*list(model.children())[0][3:])

        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

        upscale_factor_y, upscale_factor_cb, upscale_factor_cr = 1*upscale_ratio, 2*upscale_ratio, 2*upscale_ratio

        if upscale_ratio == 1:
            self.upconv_y = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=8 * upscale_factor_y * upscale_factor_y,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU6(inplace=True)
            )
        elif upscale_ratio == 2:
            self.upconv_y = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=8 * upscale_factor_y * upscale_factor_y,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor_y),
                nn.BatchNorm2d(8),
                nn.ReLU6(inplace=True)
            )

        self.upconv_cb = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8 * upscale_factor_cb * upscale_factor_cb,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor_cb),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True)
        )

        self.upconv_cr = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8 * upscale_factor_cr * upscale_factor_cr,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor_cr),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True)
        )

        if input_gate:
            self.inp_GM = GateModule(192, 28, False, False)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if any(s in name for s in ['upconv_y', 'upconv_cb', 'upconv_cr', 'bn_y', 'bn_cb', 'bn_cr']):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

            elif 'gate_l' in name:
                # Initialize last layer of gate with low variance
                m.weight.data.normal_(0, 0.001)
                m.bias.data[::2].fill_(0.1)
                m.bias.data[1::2].fill_(2)

    def forward(self, dct_y, dct_cb, dct_cr):
        if self.input_gate:
            dct_y, dct_cb, dct_cr, inp_atten = self.inp_GM(dct_y, dct_cb, dct_cr)

        dct_y = self.upconv_y(dct_y)
        dct_cb = self.upconv_cb(dct_cb)
        dct_cr = self.upconv_cr(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.input_gate:
            return x, inp_atten
        else:
            return x

def mobilenetv2dct_subpixel(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_Subpixel(**kwargs)

    return model

class MobileNetV2DCT_Subpixel_Subset(nn.Module):
    def __init__(self, upscale_ratio=1, channels=0):
        super(MobileNetV2DCT_Subpixel_Subset, self).__init__()

        self.upscale_ratio = upscale_ratio
        if channels != 0:
            self.subset_channel_index = subset_channel_index

            y_in_ch = self.subset_channel_index[str(channels)][0]
            cb_in_ch = self.subset_channel_index[str(channels)][1]
            cr_in_ch = self.subset_channel_index[str(channels)][2]
        else:
            y_in_ch, cb_in_ch, cr_in_ch = 64, 64, 64

        if upscale_ratio == 1:
            model = mobilenetv2(pretrained=True, upscale=False)
            self.features = nn.Sequential(*list(model.children())[0][4:])
        elif upscale_ratio == 2:
            model = mobilenetv2(pretrained=True, upscale=True)
            self.features = nn.Sequential(*list(model.children())[0][3:])

        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

        upscale_factor_y, upscale_factor_cb, upscale_factor_cr = 1*upscale_ratio, 2*upscale_ratio, 2*upscale_ratio

        if upscale_ratio == 1:
            self.upconv_y = nn.Sequential(
                nn.Conv2d(in_channels=y_in_ch, out_channels=8 * upscale_factor_y * upscale_factor_y,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU6(inplace=True)
            )
        elif upscale_ratio == 2:
            self.upconv_y = nn.Sequential(
                nn.Conv2d(in_channels=y_in_ch, out_channels=8 * upscale_factor_y * upscale_factor_y,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor_y),
                nn.BatchNorm2d(8),
                nn.ReLU6(inplace=True)
            )

        self.upconv_cb = nn.Sequential(
            nn.Conv2d(in_channels=cb_in_ch, out_channels=8 * upscale_factor_cb * upscale_factor_cb,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor_cb),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True)
        )

        self.upconv_cr = nn.Sequential(
            nn.Conv2d(in_channels=cr_in_ch, out_channels=8 * upscale_factor_cr * upscale_factor_cr,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor_cr),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if any(s in name for s in ['upconv_y', 'upconv_cb', 'upconv_cr', 'bn_y', 'bn_cb', 'bn_cr']):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, dct_y, dct_cb, dct_cr):
        dct_y = self.upconv_y(dct_y)
        dct_cb = self.upconv_cb(dct_cb)
        dct_cr = self.upconv_cr(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenetv2dct_subpixel_subset(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_Subpixel_Subset(**kwargs)

    return model

class MobileNetV2DCT_Subset_woinp(nn.Module):
    def __init__(self, channels=0, input_gate=False):
        super(MobileNetV2DCT_Subset_woinp, self).__init__()

        self.input_gate = input_gate

        model = mobilenetv2(pretrained=True, upscale=True)
        self.features = nn.Sequential(*list(model.children())[0][1:])
        if channels < 32:
            temp_layer = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            temp_layer.weight.data = self.features[0].conv[0].weight.data[:channels]
            self.features[0].conv[0] = temp_layer

            temp_layer = nn.BatchNorm2d(channels)
            temp_layer.weight.data = self.features[0].conv[1].weight.data[:channels]
            temp_layer.bias.data = self.features[0].conv[1].bias.data[:channels]
            temp_layer.running_mean.data = self.features[0].conv[1].running_mean.data[:channels]
            temp_layer.running_var.data = self.features[0].conv[1].running_var.data[:channels]
            self.features[0].conv[1] = temp_layer

            temp_layer = nn.Conv2d(channels, self.features[0].conv[3].out_channels, 1, 1, 0, bias=False)
            temp_layer.weight.data = self.features[0].conv[3].weight.data[:, :channels]
            self.features[0].conv[3] = temp_layer
        elif channels == 192:
            out_ch = self.features[0].conv[0].out_channels
            temp_layer = nn.Conv2d(channels, out_ch, 3, 1, 1, groups=out_ch, bias=False)
            temp_layer.weight.data = self.features[0].conv[0].weight.data.repeat(1, 6, 1, 1)
            self.features[0].conv[0] = temp_layer

        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

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

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.input_gate:
            return x, inp_atten
        else:
            return x

def mobilenetv2dct_subset_woinp(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_Subset_woinp(**kwargs)

    return model

class MobileNetV2DCT_Subset_woinp_from_scratch(nn.Module):
    def __init__(self, channels=0, input_gate=False):
        super(MobileNetV2DCT_Subset_woinp_from_scratch, self).__init__()

        self.input_gate = input_gate

        model = mobilenetv2(pretrained=False, upscale=True)
        self.features = nn.Sequential(*list(model.children())[0][1:])

        if channels == 192:
            out_ch = self.features[0].conv[0].out_channels
            self.features[0].conv[0] = nn.Conv2d(channels, out_ch, 3, 1, 1, groups=out_ch, bias=False)
            kaiming_init(self.features[0].conv[0])
        elif channels < 32:
            temp_layer = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            temp_layer.weight.data = self.features[0].conv[0].weight.data[:channels]
            self.features[0].conv[0] = temp_layer

            temp_layer = nn.BatchNorm2d(channels)
            temp_layer.weight.data = self.features[0].conv[1].weight.data[:channels]
            temp_layer.bias.data = self.features[0].conv[1].bias.data[:channels]
            temp_layer.running_mean.data = self.features[0].conv[1].running_mean.data[:channels]
            temp_layer.running_var.data = self.features[0].conv[1].running_var.data[:channels]
            self.features[0].conv[1] = temp_layer

            temp_layer = nn.Conv2d(channels, self.features[0].conv[3].out_channels, 1, 1, 0, bias=False)
            temp_layer.weight.data = self.features[0].conv[3].weight.data[:, :channels]
            self.features[0].conv[3] = temp_layer

        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

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

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.input_gate:
            return x, inp_atten
        else:
            return x

def mobilenetv2dct_subset_woinp_from_scratch(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_Subset_woinp_from_scratch(**kwargs)

    return model

if __name__ == '__main__':
    import numpy as np

    channels = 192
    img  = torch.from_numpy(np.random.randn(16, channels, 112, 112)).float()
    model = mobilenetv2dct_subset_woinp(channels=channels)
    x = model(img)
    print(x.shape)

    dct_y  = torch.from_numpy(np.random.randn(16, 64, 28, 28)).float()
    dct_cb = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()
    dct_cr = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()
    model_resnet50dct = mobilenetv2dct_subpixel(upscale_ratio=2)
    x = model_resnet50dct(dct_y, dct_cb, dct_cr)
    print(x.shape)

    dct_y  = torch.from_numpy(np.random.randn(16, 64, 28, 28)).float()
    dct_cb = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()
    dct_cr = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()

    # ResNet50DCT
    model_resnet50dct = mobilenetv2dct()

    x = model_resnet50dct(dct_y, dct_cb, dct_cr)
    print(x.shape)

    model = mobilenetv2dct()
    x = model(dct_y, dct_cb, dct_cr)
    print(x.shape)

    dct_y  = torch.from_numpy(np.random.randn(16, 8, 28, 28)).float()
    dct_cb = torch.from_numpy(np.random.randn(16, 8, 14, 14)).float()
    dct_cr = torch.from_numpy(np.random.randn(16, 8, 14, 14)).float()

    x = model(dct_y, dct_cb, dct_cr)
    print(x.shape)

    import csv
    with open("input_raw.csv", "w") as f:
        w = csv.writer(f)
        for key, val in list(model.named_parameters()):
            w.writerow([key, val])

