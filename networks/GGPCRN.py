import torch
import torch.nn as nn
import torch.nn.functional as F

class PyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):

    def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1, 4, 8, 16]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class PyResBlock(nn.Module):
    def __init__(self, inFe, outFe):
        super(PyResBlock, self).__init__()
        self.conv1 = get_pyconv(inFe,outFe,[3, 5, 7, 9])
        self.relu = nn.ReLU()
        self.conv2 = get_pyconv(inFe,outFe,[3, 5, 7, 9])

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x


class PyResGroup(nn.Module):
    def __init__(self, numFe, num_block):
        super(PyResGroup, self).__init__()
        body = []
        for _ in range(num_block):
            body.append(PyResBlock(numFe, numFe))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = self.body(x)
        return out


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        numFe = opt['numFe']
        in_channels = opt['in_channels']
        num_block = opt['num_block']
        num_group = opt['num_group']

        self.fea_conv = nn.Sequential(
            nn.Conv2d(5, numFe, 3, 1, 1),
            nn.ReLU()
        )

        self.conv1=PyResGroup(numFe,num_block)
        self.conv2=nn.Sequential(
            nn.Conv2d(2 * numFe, numFe, 3, 1, 1),
            nn.ReLU(),
            PyResGroup(numFe, num_block)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(2 * numFe, numFe, 3, 1, 1),
            nn.ReLU(),
            PyResGroup(numFe, num_block)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(2 * numFe, numFe, 3, 1, 1),
            nn.ReLU(),
            PyResGroup(numFe, num_block)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(2 * numFe, numFe, 3, 1, 1),
            nn.ReLU(),
            PyResGroup(numFe, num_block)
        )
        self.fuse=nn.Sequential(
            nn.Conv2d(2*numFe,numFe,3,1,1),
            nn.ReLU()
        )

        self.re = nn.Conv2d(numFe, in_channels, 3, 1, 1)

        self.get_grad=Get_gradient_nopadding()
        self.g_fea=nn.Sequential(
            nn.Conv2d(5, numFe, 3, 1, 1),
            nn.ReLU()
        )

        self.g_block1 = nn.Sequential(
            nn.Conv2d(2 * numFe, numFe, 3, 1, 1),
            nn.ReLU(),
            PyResBlock(numFe, numFe)
        )

        self.g_block2 = nn.Sequential(
            nn.Conv2d(2 * numFe, numFe, 3, 1, 1),
            nn.ReLU(),
            PyResBlock(numFe, numFe)
        )

        self.g_block3 = nn.Sequential(
            nn.Conv2d(2 * numFe, numFe, 3, 1, 1),
            nn.ReLU(),
            PyResBlock(numFe, numFe)
        )

        self.g_block4 = nn.Sequential(
            nn.Conv2d(2 * numFe, numFe, 3, 1, 1),
            nn.ReLU(),
            PyResBlock(numFe, numFe)
        )

        self.g_conv1=nn.Sequential(
            nn.Conv2d(numFe,numFe,3,1,1),
            nn.ReLU()
        )

        self.g_conv2=nn.Conv2d(numFe,numFe,3,1,1)






    def forward(self, ms, pan):
        res = F.interpolate(ms, scale_factor=4, mode='bicubic', align_corners=False)
        fea = self.fea_conv(torch.cat([pan, res], dim=1))
        fea1 = self.conv1(fea)

        x_grad=self.get_grad(ms)
        x_grad=F.interpolate(x_grad, scale_factor=4, mode='bicubic', align_corners=False)
        pan_grad = self.get_grad(pan)
        g_fea=self.g_fea(torch.cat([x_grad,pan_grad],dim=1))
        g_fea1=self.g_block1(torch.cat([g_fea,fea1],dim=1))
        fea2 = self.conv2(torch.cat([fea1, g_fea1], dim=1))
        g_fea2=self.g_block2(torch.cat([g_fea1,fea2],dim=1))
        fea3 = self.conv3(torch.cat([fea2, g_fea2], dim=1))
        g_fea3=self.g_block3(torch.cat([g_fea2,fea3],dim=1))
        fea4 = self.conv4(torch.cat([fea3, g_fea3], dim=1))
        g_fea4 = self.g_block4(torch.cat([g_fea3, fea4], dim=1))
        fea5=self.conv5(torch.cat([fea4, g_fea4], dim=1))
        g_res=self.g_conv1(g_fea4)
        g_out=g_fea+g_res
        g_fuse=self.g_conv2(g_out)


        out=self.fuse(torch.cat([fea5,g_fuse],dim=1))
        out = self.re(out)
        out = out + res
        return out


class myloss(nn.Module):
    def __init__(self, opt):
        super(myloss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.get_grad = Get_gradient_nopadding()
        self.lmd2 = opt['lmd2']

    def forward(self, ms, HR):
        global_loss = self.l1_loss(ms, HR)
        hr_grad=self.get_grad(HR)
        sr_grad=self.get_grad(ms)
        grad_sr_loss=self.l1_loss(sr_grad,hr_grad)
        return global_loss+self.lmd2*grad_sr_loss