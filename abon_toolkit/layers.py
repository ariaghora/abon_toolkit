import torch.nn as nn

class BatchConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BatchConv1d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels*groups, out_channels*groups,
                              kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x):
        assert len(x.shape) == 4, "batchconv1d expects a 4d [{}] tensor".format(x.shape)
        b_i, b_j, c, h = x.shape
        out = self.conv(x.permute([1, 0, 2, 3]).contiguous().view(b_j, b_i * c, h))
        return out.view(b_j, b_i, self.out_channels,
                        out.shape[-1]).permute([1, 0, 2, 3])
