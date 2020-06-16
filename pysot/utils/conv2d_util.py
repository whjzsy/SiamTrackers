from numbers import Number

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import _pair
try:
    import pyinn as P

    has_pyinn = True
except ImportError:
    P = None
    has_pyinn = False
    pass

def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False,
           use_pyinn_if_possible=False):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.'
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [(k - 1) * d - p for (k, d, p) in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1)
        padding = _pair(0)

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up
    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output

def conv2d_psvf(input, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Forward computation
    (x - y)^2
    :param input: A tensor with shape [1, batch * channel, h, w]
        representation search features
    :param kernel: A tensor with shape [batch * channel, 1, k, k]
        representation template feature
    :param bias:
    :param stride:
    :param padding:
    :param dilation: 空洞卷积拓展感受野
    :param groups: 确定卷积的层次
    :return: outputs: [1, batch * channel, (h - k)/stride + 1, (w - k)/stride + 1]
    """
    kernel_size = tuple(kernel.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel = kernel.view(1, groups, kernel_size[0], kernel_size[1])
    in_sz = tuple(input.shape[-2:])
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    output = torch.zeros([1, groups, out_sz[0], out_sz[1]], device='cuda')
    # main compute batch channel [1,256, 31, 31] [1, 256, 7, 7]
    for i in range(out_sz[0]):
        for j in range(out_sz[1]):
            temp = input[:, :, i:i+kernel_size[0], j:j+kernel_size[1]] - kernel
            output[:,:,i,j] = torch.einsum('ijkl->ij', temp)
    output = torch.pow(output, 2)
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    return output

def conv2d_svf(input, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Forward computation
    x^2 - y^2
    :param input: A tensor with shape [1, batch * channel, h, w]
         representation search features
    :param kernel: A tensor with shape [batch * channel, 1, k, k]
         representation template feature
    :param bias:
    :param stride:
    :param padding:
    :param dilation: 空洞卷积拓展感受野
    :param groups: 确定卷积的层次
    :return: outputs: [1, batch * channel, (h - k)/stride + 1, (w - k)/stride + 1]
    """
    kernel_size = tuple(kernel.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    # main computation
    kernel = kernel.view(1, groups, kernel_size[0], kernel_size[1])
    kernel = torch.pow(kernel, 2)
    input = torch.pow(input, 2)
    in_sz = tuple(input.shape[-2:])
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    output = torch.zeros([1, groups, out_sz[0], out_sz[1]], device='cuda')
    # main compute batch channel [1,256, 31, 31] [1, 256, 7, 7]
    for i in range(out_sz[0]):
        for j in range(out_sz[1]):
            temp = input[:, :, i:i + kernel_size[0], j:j + kernel_size[1]] - kernel
            output[:, :, i, j] = torch.einsum('ijkl->ij', temp)
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    return output

if __name__ == '__main__':
    input = torch.ones(1, 16, 31, 31)
    kernel = torch.randn(16, 1, 7, 7)
    # out = F.conv2d(input, kernel, groups= input.size(1))
    out = conv2d_svf(input, kernel, groups=input.size(1))
    print(out)
    out_1 = conv2d_psvf(input, kernel, groups=input.size(1))
    print(out_1.shape)
    print(out_1)

