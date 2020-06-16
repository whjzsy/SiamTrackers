# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

from pysot.utils.conv2d_util import conv2d_psvf, conv2d_svf

def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

def multixcorr_depthwise(x, kernel):
    """
    multi depthwise correlation
    :param x:
    :param kernel:
    :return:
    """
    batch = kernel.size(0)
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    # Square variance formula depthwise correlation x^2 - y^2
    out_1 = conv2d_svf(x, kernel, groups=batch * channel)
    out_1 = out_1.view(batch, channel, out_1.size(2), out_1.size(3))
    # Perfect Square variance formula depthwise correlation (x - y)^2
    out_2 = conv2d_psvf(x, kernel, groups=batch * channel)
    out_2 = out_2.view(batch, channel, out_2.size(2), out_2.size(3))
    # normal depthwise correlation x * y
    out_3 = F.conv2d(x, kernel, groups=batch * channel)
    out_3 = out_3.view(batch, channel, out_3.size(2), out_3.size(3))
    out = torch.cat((out_1, out_2, out_3), 1)
    return out

if __name__ == '__main__':
    input = torch.ones(4, 16, 31, 31)
    kernel = torch.ones(4, 16, 7, 7)
    # out = F.conv2d(input, kernel, groups= input.size(1))
    out = multixcorr_depthwise(input, kernel)
    # print(out)
    print(out.shape)