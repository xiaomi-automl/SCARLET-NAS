import math

from models.basic_ops import *


class ScarletB(nn.Module):
    def __init__(self, n_class=1000, input_size=224):
        super(ScarletB, self).__init__()
        assert input_size % 32 == 0
        mb_config = [
            # expansion, out_channel, kernel_size, stride, se
            [3, 32, 3, 2, True],
            [3, 32, 5, 1, True],
            [3, 40, 3, 2, True],
            [6, 40, 7, 1, True],
            [3, 40, 3, 1, False],
            [3, 40, 5, 1, False],
            [6, 80, 7, 2, True],
            [3, 80, 3, 1, True],
            "identity",
            [3, 80, 5, 1, False],
            [3, 96, 3, 1, True],
            [3, 96, 3, 1, True],
            [6, 96, 7, 1, True],
            [3, 96, 3, 1, True],
            [6, 192, 5, 2, True],
            [6, 192, 5, 1, True],
            "identity",
            [6, 192, 7, 1, True],
            [6, 320, 5, 1, True],
        ]
        input_channel = 16
        last_channel = 1280

        self.last_channel = last_channel
        self.stem = stem(3, 32, 2)
        self.separable_conv = separable_conv(32, 16)
        self.mb_module = list()
        for each_config in mb_config:
            if each_config == "identity":
                self.mb_module.append(Identity())
                continue
            t, c, k, s, e = each_config
            output_channel = c
            self.mb_module.append(InvertedResidual(input_channel, output_channel, k, s, expand_ratio=t, is_use_se=e))
            input_channel = output_channel
        self.mb_module = nn.Sequential(*self.mb_module)
        self.conv_before_pooling = conv_before_pooling(input_channel, self.last_channel)
        self.classifier = nn.Linear(self.last_channel, n_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        x = self.mb_module(x)
        x = self.conv_before_pooling(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()
