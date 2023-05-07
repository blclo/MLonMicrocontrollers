###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
The CIFAR network found by NAS.
"""
from torch import nn

import ai8x
import ai8x_blocks

class HomemadeSimpleNet(nn.Module):
    """
    SimpleNet v1 Model with BatchNorm
    """
    def __init__(
            self,
            num_classes=100,
            num_channels=3,
            dimensions=(32, 32),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()
       

        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 64, 3, stride=1, padding=1,
                                              bias=bias, batchnorm='Affine', **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(64, 128, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.conv3 = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.conv4 = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.maxpool1 = ai8x.MaxPool2d(kernel_size=2, stride=2,
                                dilation=1, padding=0)
        
        self.dropout1 = nn.Dropout(0.1)

        self.conv5 = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.conv6 = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.conv7 =  ai8x.FusedConv2dBNReLU(128, 256, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.maxpool2 = ai8x.MaxPool2d(kernel_size=2, stride=2,
                                dilation=1, padding=0)
        
        self.dropout2 = nn.Dropout(0.1)

        self.conv8 = ai8x.FusedConv2dBNReLU(256, 256, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.conv9 = ai8x.FusedConv2dBNReLU(256, 256, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.maxpool3 = ai8x.MaxPool2d(kernel_size=2, stride=2,
                                dilation=1, padding=0)
        
        self.dropout3 = nn.Dropout(0.1)
        self.conv10 = ai8x.FusedConv2dBNReLU(256, 512, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.maxpool4 = ai8x.MaxPool2d(kernel_size=2, stride=2,
                                dilation=1, padding=0)
        
        self.dropout4 = nn.Dropout(0.1)

        self.conv11 = ai8x.FusedConv2dBNReLU(512, 2048, 1, stride=1, padding=0, bias=bias, batchnorm='Affine', **kwargs)
        self.conv12 = ai8x.FusedConv2dBNReLU(2048, 256, 1, stride=1, padding=0, bias=bias, batchnorm='Affine', **kwargs)

        self.maxpool5 = ai8x.MaxPool2d(kernel_size=2, stride=2,
                                dilation=1, padding=0)
        
        self.dropout5 = nn.Dropout(0.1)
        self.conv13 = ai8x.FusedConv2dBNReLU(256, 256, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.classifier = nn.Linear(256, num_classes)
        self.drp = nn.Dropout(0.1)



    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.conv10(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool5(x)
        x = self.dropout5(x)
        x = self.conv13(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def homemadesimplenet(pretrained=False, **kwargs):
    """
    Constructs a NAS v1 model.
    """
    assert not pretrained
    return HomemadeSimpleNet(**kwargs)


models = [
    {
        'name': 'homemadesimplenet',
        'min_input': 1,
        'dim': 2,
    },
]
