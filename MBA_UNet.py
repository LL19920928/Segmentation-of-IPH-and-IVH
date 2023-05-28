"""
    Implementation of MBA-UNet
"""
import torch.nn as nn
import torch


class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(Double_Conv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down(nn.Module):
    def __init__(self, kernel_size=(2, 2, 2)):
        super(Down, self).__init__()
        self.down = nn.MaxPool3d(kernel_size)

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                                nn.BatchNorm3d(out_channels),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class MBA_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(MBA_UNet, self).__init__()
        # Encoder
        self.conv3d_1 = Double_Conv(in_channels=in_channels, out_channels=16)
        self.down3d_1 = Down(kernel_size=(2, 2, 2))
        self.conv3d_2 = Double_Conv(in_channels=16, out_channels=32)
        self.down3d_2 = Down(kernel_size=(1, 2, 2))
        self.conv3d_3 = Double_Conv(in_channels=32, out_channels=64)
        self.down3d_3 = Down(kernel_size=(1, 2, 2))
        self.conv3d_4 = Double_Conv(in_channels=64, out_channels=128)
        self.down3d_4 = Down(kernel_size=(1, 2, 2))

        # MBAM
        self.assp1_d1 = nn.Conv3d(128, 128, kernel_size=1, stride=1, padding=0)
        self.assp1_d2 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.assp1_d3 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.assp1_d4 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3)

        # Decoder
        self.up3d_1 = Up(in_channels=512, out_channels=128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3d_6 = Double_Conv(in_channels=128 + 128, out_channels=128)
        self.up3d_2 = Up(in_channels=128, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3d_7 = Double_Conv(in_channels=64 + 64, out_channels=64)
        self.up3d_3 = Up(in_channels=64, out_channels=32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3d_8 = Double_Conv(in_channels=32 + 32, out_channels=32)
        self.up3d_4 = Up(in_channels=32, out_channels=16)
        self.conv3d_9 = Double_Conv(in_channels=16 + 16, out_channels=16)
        self.conv_seg = nn.Conv3d(in_channels=16, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x: input( BxCxDxHxW )
            B: batch size
            C: channel
            D: slice number (depth)
            H: height
            W: weight
        :return x: output( Bx3CxDxHxW )
            1st C: IPH
            2nd C: IVH
            3rd C: ICH
        """
        # Encoder
        skip1 = self.conv3d_1(x)
        x = self.down3d_1(skip1)
        skip2 = self.conv3d_2(x)
        x = self.down3d_2(skip2)
        skip3 = self.conv3d_3(x)
        x = self.down3d_3(skip3)
        skip4 = self.conv3d_4(x)
        x = self.down3d_4(skip4)

        # MBAM
        x1 = self.assp1_d1(x)
        x2 = self.assp1_d2(x)
        x3 = self.assp1_d3(x)
        x4 = self.assp1_d4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        # Decoder
        x = self.up3d_1(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.conv3d_6(x)
        x = self.up3d_2(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv3d_7(x)
        x = self.up3d_3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3d_8(x)
        x = self.up3d_4(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv3d_9(x)
        x = self.conv_seg(x)
        return x

