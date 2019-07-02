import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            nn.Conv2d(channel_num, channel_num, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel_num)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class SpecGenerator(nn.Module):
    def __init__(self, noise_dim, pix_num=1024*1024):
        super(SpecGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 4*4*512),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),   # output 8*8

            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),  # output 16*16

            nn.ConvTranspose2d(256, 256, 4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),  # output 64*64

            nn.ConvTranspose2d(256, 128, 4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # output 256*256

            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # output 512*512

            nn.ConvTranspose2d(128, 3, 1, stride=1, padding=0, bias=False),
            nn.Tanh()   # [3, 512, 512]
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        out = out.view(out.size(0), 512, 4, 4)
        out = self.conv(out)
        return out

class SpecDiscriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(SpecDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),     # 256*256
            nn.Conv2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),     # 128*128
            nn.Conv2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),     # 64*64
            nn.Conv2d(256, 128, 4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),     # 16*16
            nn.Conv2d(128, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),     # 8*8
            nn.Conv2d(128, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),      # 4*4
        )
        self.block_layer1 = ResidualBlock(128)
        self.block_layer2 = ResidualBlock(128)
        self.block_layer3 = ResidualBlock(128)
        self.block_layer4 = ResidualBlock(128)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Sequential(
            # nn.AvgPool2d(4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        out = self.conv(x)
        out = self.block_layer1(out)
        out = self.block_layer2(out)
        out = self.block_layer3(out)
        out = self.block_layer4(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out

# noise = torch.rand(4, 100)
# G = SpecGenerator(noise_dim=100)
# D = SpecDiscriminator()
# #
# fake_img = G(noise)
# print(fake_img.shape)    # torch.Size([4, 3, 1024, 1024])
# out_scale = D(fake_img)
# print(out_scale.shape)   # torch.Size([4, 1])

