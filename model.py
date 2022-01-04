import torch
import torch.nn as nn

class conv2d_block(nn.Module):
    """
    2D Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv2d_block, self).__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv2d(x)
        return x


class conv3d_block(nn.Module):
    """
    3D Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv3d_block, self).__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x


class down(nn.Module):
    """
    Downsampling
    """

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d = conv2d_block(in_ch, out_ch)

    def forward(self, x):
        x = self.Maxpool(x)
        x = self.conv2d(x)

        return x


class up(nn.Module):
    """
    Upsampling
    """

    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.conv3d = conv3d_block(in_ch, out_ch)

    def forward(self, f_e, d_e):
        f_e = f_e.unsqueeze(2)
        f_e = f_e.repeat(1, 1, d_e.shape[2], 1, 1)
        x = torch.cat((f_e, d_e), dim=1)
        x = self.conv3d(x)

        return x


class U_Net_Generator(nn.Module):
    """
    3D UNet implementation
    """

    def __init__(self, inch=1, out_ch=1):
        super(U_Net, self).__init__()

        self.conv0 = conv2d_block(1, 64)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

        self.up1 = up(768, 512)
        self.up2 = up(768, 256)
        self.up3 = up(384, 128)
        self.up4 = up(192, 64)

        self.upsample = nn.Upsample(scale_factor=2)

        self.final_1 = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1))
        self.final_2 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1))

        self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e0 = self.conv0(x)

        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)

        gen_volume = e4.view(-1, 256, 4, 8, 8)

        d4 = self.upsample(gen_volume)
        d4 = self.up1(e3, d4)

        d3 = self.upsample(d4)
        d3 = self.up2(e2, d3)

        d2 = self.upsample(d3)
        d2 = self.up3(e1, d2)

        d1 = self.upsample(d2)
        d1 = self.up4(e0, d1)

        out = self.final_1(d1)
        out = self.final_2(out)

        out = self.active(out)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(  # [-1, 1, 60, 128, 128]
            nn.Conv3d(1, 32, kernel_size=(2,4,4), stride=2, padding=(2,1,1), bias=True), # [-1, 32, 32, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=(4, 4, 4), stride=2, padding=(1, 1, 1), bias=True),# [-1, 64, 16, 32, 32]
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=(4, 4, 4), stride=2, padding=(1, 1, 1), bias=True),  # [-1, 128, 8, 16, 16]
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=(4, 4, 4), stride=2, padding=(1, 1, 1), bias=True),  # [-1, 256, 4, 8, 8]
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, kernel_size=(4, 4, 4), stride=2, padding=(1, 1, 1), bias=True),  # [-1, 512, 2, 4, 4]
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 2, kernel_size=(2, 4, 4), stride=1, padding=(0, 0, 0), bias=True)  # [-1, 2, 1, 1, 1]
        )
        self.mymodules = nn.ModuleList([nn.Sequential(nn.Linear(2,1), nn.Sigmoid())])

    def forward(self, x):
        out = self.model(x).squeeze()
        out = self.mymodules[0](out).squeeze()
        return out










































