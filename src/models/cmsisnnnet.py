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

class CmsisnNet(nn.Module):
    """
    SimpleNet v1 Model with BatchNorm
    """
    def __init__(
            self,
            num_classes=10,
            num_channels=3,
            dimensions=(32, 32),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()
       

        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 32, 3, stride=1, padding=1,
                                              bias=bias, batchnorm='Affine', **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(32, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(32, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1, bias=bias, batchnorm='Affine', **kwargs)

        self.maxpool1 = ai8x.MaxPool2d(kernel_size=2, stride=2,
                                dilation=1, padding=0)
        
        
        self.classifier = ai8x.Linear(1024, num_classes,  bias=bias,  wide=True, **kwargs)




    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def cmsisnnet(pretrained=False, **kwargs):
    """
    Constructs a NAS v1 model.
    """
    assert not pretrained
    return CmsisnNet(**kwargs)


models = [
    {
        'name': 'cmsisnnet',
        'min_input': 1,
        'dim': 2,
    },
]
