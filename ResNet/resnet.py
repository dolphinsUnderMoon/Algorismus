import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


affine_par = True


def conv3x3(num_in_kernels, num_out_kernels, stride=1):
    return nn.Conv2d(
        in_channels=num_in_kernels,
        out_channels=num_out_kernels,
        stride=stride,
        padding=1,
        bias=False
    )


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, num_in_kernels, num_out_kernels, stride=1, down_sample=None):
        super(ResidualBlock, self).__init__()

        self.conv_1 = conv3x3(num_in_kernels, num_out_kernels, stride)
        self.bn_1 = nn.BatchNorm2d(num_out_kernels, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(num_out_kernels, num_out_kernels)
        self.bn_2 = nn.BatchNorm2d(num_out_kernels, affine=affine_par)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        shortcut = x if self.down_sample is None else self.down_sample(x)

        residual = self.conv_1(x)
        residual = self.bn_1(residual)
        residual = self.relu(residual)

        residual = self.conv_2(residual)
        residual = self.bn_2(residual)

        output = shortcut + residual
        output = self.relu(output)

        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_in_kernels, num_neck_kernels, stride=1,
                 dilation_=1, down_sample=None):
        super(Bottleneck, self).__init__()

        self.conv_1 = nn.Conv2d(num_in_kernels, num_neck_kernels, kernel_size=1,
                                stride=stride, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_neck_kernels, affine=affine_par)
        for i in self.bn_1.parameters():
            i.requires_grad = False

        padding = dilation_

        self.conv_2 = nn.Conv2d(num_neck_kernels, num_neck_kernels, kernel_size=3,
                                stride=1, padding=padding, bias=False, dilation=dilation_)
        self.bn_2 = nn.BatchNorm2d(num_neck_kernels, affine=affine_par)
        for i in self.bn_2.parameters():
            i.requires_grad = False

        num_post_neck_kernels = num_neck_kernels * 4
        self.conv_3 = nn.Conv2d(num_neck_kernels, num_post_neck_kernels, kernel_size=1,
                                bias=False)
        self.bn_3 = nn.BatchNorm2d(num_post_neck_kernels, affine=affine_par)
        for i in self.bn_3.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.down_sample = down_sample

    def forward(self, x):
        shortcut = x if self.down_sample is None else self.down_sample(x)

        residual = self.conv_1(x)
        residual = self.bn_1(residual)
        residual = self.relu(residual)

        residual = self.conv_2(residual)
        residual = self.bn_2(residual)
        residual = self.relu(residual)

        residual = self.conv_3(residual)
        residual = self.bn_3(residual)

        output = shortcut + residual
        output = self.relu(output)

        return output


class ClassifierModule(nn.Module):
    def __init__(self, dilation_series, padding_series, no_labels):
        super(ClassifierModule, self).__init__()

        self.conv2d_list = nn.ModuleList()

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(
                in_channels=2048,
                out_channels=no_labels,
                kernel_size=(3, 3),
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=True
            ))

        for layer in self.conv2d_list:
            layer.weight.data.normal_(0, 0.01)

    def forward(self, x):
        output = self.conv2d_list[0](x)

        for i in range(len(self.conv2d_list) - 1):
            output += self.conv2d_list[i + 1](x)

        return output


class ResNet(nn.Module):
    def __init__(self, block, layers, no_labels):
        super(ResNet, self).__init__()

        self.num_input_channels = 64

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_features=64, affine=affine_par)
        for i in self.bn_1.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer_1 = self._create_layer(block, 64, layers[0])
        self.layer_2 = self._create_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._create_layer(block, 256, layers[2], stride=1, dilation_=2)
        self.layer_4 = self._create_layer(block, 512, layers[3], stride=1, dilation_=4)

        classifier_dilation_series = [6, 12, 18, 24]
        classifier_padding_series = [6, 12, 18, 24]
        self.layer_5 = self._make_pred_layer(block=ClassifierModule,
                                             dilation_series=classifier_dilation_series,
                                             padding_series=classifier_padding_series,
                                             no_labels=no_labels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _create_layer(self, block, planes, blocks, stride=1, dilation_=1):
        down_sample = None

        if stride != 1 or self.num_input_channels != planes * block.expansion or dilation_ == 2 or dilation_ == 4:
            down_sample = nn.Sequential(
                nn.Conv2d(self.num_input_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )

        for i in down_sample._modules['1'].parameters():
            i.requires_grad = False

        layers = [block(self.num_input_channels, planes, stride, dilation_=dilation_, down_sample=down_sample)]
        self.num_input_channels = block.expansion * planes
        for i in range(1, blocks):
            layers.append(block(self.num_input_channels, planes, dilation_=dilation_))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, no_labels):
        return block(dilation_series, padding_series, no_labels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer_5(
            self.layer_4(
                self.layer_3(
                    self.layer_2(
                        self.layer_1(x)
                    )
                )
            ))

        return x
