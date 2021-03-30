"""
model file
some class is not test yet ...
"""

from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class PathSRGAN(nn.Module):
    """
    idea: calculate mount,receptive,unet, densenet
    """
    def __init__(self, input_num, dense_beta=0.5):
        super(PathSRGAN, self).__init__()
        self.stage1_eye = nn.Sequential(
            nn.Conv2d(input_num*3, 64, 9, 1, padding=4),
            nn.PReLU()
        )
        self.stage1_down1 = nn.Sequential(
            DenseBlock_fixed(64, beta=dense_beta),
            DenseBlock_fixed(64, beta=dense_beta)
        )
        self.stage1_down2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            DenseBlock_fixed(128, beta=dense_beta),
            DenseBlock_fixed(128, beta=dense_beta)
        )
        self.stage1_down3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            DenseBlock_fixed(256, beta=dense_beta),
            DenseBlock_fixed(256, beta=dense_beta)
        )
        self.stage1_bottom = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.PReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(512, 256, 3, 1, padding=1),
            nn.PReLU()
        )
        self.stage1_up1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock_fixed(256, beta=dense_beta) for _ in range(2)]),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.PReLU()
        )
        self.stage1_up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock_fixed(128, beta=dense_beta) for _ in range(2)]),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.PReLU()
        )
        self.stage1_up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock_fixed(64, beta=dense_beta) for _ in range(2)]),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU()
        )
        self.stage1_upsample = UpsampleBLock(64, 2)
        self.stage1_out = nn.Conv2d(64,3,7,1,padding=3)
        self.stage2_down1 = nn.Sequential(
            DenseBlock_fixed(64, beta=dense_beta),
            nn.PReLU()
        )
        self.stage2_down2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,padding=1),
            nn.PReLU(),
            DenseBlock_fixed(128, beta=dense_beta),
            nn.PReLU()
        )
        self.stage2_upsample = UpsampleBLock(128,2)
        self.stage2_out = nn.Conv2d(128,3,9,1,padding=4)

    def forward(self, x , stage1 = False):
        eye = self.stage1_eye(x)
        # first stage
        s1_down1 = self.stage1_down1(eye)
        downsample1 = F.avg_pool2d(s1_down1, kernel_size=2, stride=2)
        s1_down2 = self.stage1_down2(downsample1)
        downsample2 = F.avg_pool2d(s1_down2, kernel_size=2, stride=2)
        s1_down3 = self.stage1_down3(downsample2)
        downsample3 = F.avg_pool2d(s1_down3, kernel_size=2, stride=2)
        s1_bottom = self.stage1_bottom(downsample3)
        up1 = F.interpolate(s1_bottom, scale_factor=2)
        s1_up1 = self.stage1_up1(torch.cat([s1_down3, up1], dim=1))
        up2 = F.interpolate(s1_up1, scale_factor=2)
        s1_up2 = self.stage1_up2(torch.cat([s1_down2, up2], dim=1))
        up3 = F.interpolate(s1_up2, scale_factor=2)
        s1_up3 = self.stage1_up3(torch.cat([s1_down1, up3], dim=1))
        sr_up = self.stage1_upsample(s1_up3)
        out1 = self.stage1_out(sr_up)
        if stage1:
            return out1
        # second stage
        s2_down1 = self.stage2_down1(sr_up)
        s2_down2 = self.stage2_down2(s2_down1)
        sr_up2 = self.stage2_upsample(s2_down2)
        conv2 = self.stage2_out(sr_up2)
        ## cautions ,cause stage1 no sigmoid, this make out1 darker
        #return [torch.sigmoid(out1), torch.sigmoid(conv2)]
        return [out1, torch.sigmoid(conv2)]

class UpsampleBLock(nn.Module):
    """
    上采样模块，pixelshuffle
    """
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self,x):
        x=self.conv(x)
        x=self.pixel_shuffle(x)
        x=self.prelu(x)
        
        return x

class Generator(nn.Module):
    '''
    单个生成器,模块包括9x9的卷积(提高感受野)，5个残差模块，残差跳跃连接，上采样模块，9x9的卷积
    '''
    def __init__(self,stage1_num = 5 ):
        super(Generator, self).__init__()
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        stage1_block = [ResidualBlock(64) for _ in range(stage1_num)]
        self.stage1_block = nn.Sequential(*stage1_block)
        self.stage1_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.stage1_up = UpsampleBLock(64, 2)
        self.stage1_conv3 = nn.Conv2d(64,3,kernel_size = 9,padding= 4)

         
    def forward(self,x):
        stage1_conv1 = self.stage1_conv1(x)
        stage1_block = self.stage1_block(stage1_conv1)
        stage1_conv2 = self.stage1_conv2(stage1_block)
        stage1_up = self.stage1_up(stage1_conv1+stage1_conv2)
        stage1_conv3 = self.stage1_conv3(stage1_up)
        gen10 = (torch.tanh(stage1_conv3)+1)/2
        
        return gen10

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024,1, kernel_size=3,padding=1)
        )

    def forward(self, x):
        x = self.net(x)
        return torch.sigmoid(x)

class Classifier(nn.Module):
    '''
    output: 
            resize 256 to 512
            classifer_labels: float type in the range[0,1] list with batch_size
            feature map: in the last relu layer before avgpool and output_layer
    '''
    def __init__(self):
        super(Classifier, self).__init__()
        classifer=resnet50(pretrained=None)
        self.resnet_layer=nn.Sequential(*list(classifer.children())[:-2])
        self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1,padding=0)
        self.GlobalMaxPooling = nn.AdaptiveAvgPool2d(1)
        self.Linear_layer0 = nn.Linear(in_features=2048, out_features=64)
        self.Activation_layer0 = nn.ReLU()
        self.Linear_layer1 = nn.Linear(in_features=64, out_features=1)

    def forward(self,x):
        out1 = self.resnet_layer(x)
        out2 = self.avgpool(out1)
        out2 = self.GlobalMaxPooling(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.Linear_layer0(out2)
        out2 = self.Activation_layer0(out2)

        y = self.Linear_layer1(out2)
        classifer_labels=torch.sigmoid(y)
        return classifer_labels,out1

class DenseBlock_fixed(nn.Module):
    """
    DenseNet密连接
    !!!module4_out do not add to this module
    !!!DO NOT ADD THIS, CAUSE LOTS OF MODULE RELY ON THIS
    """
    def __init__(self,channels,beta = 0.5):
        super(DenseBlock_fixed,self).__init__()
        self.beta = beta
        self.conv_module1 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module2 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module3 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module4 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.last_conv = nn.Conv2d(channels,channels,3,1,padding = 1)
    def forward(self,x):
        module1_out = self.conv_module1(x)
        module1_out_temp = x+module1_out
        module2_out = self.conv_module2(module1_out_temp)
        module2_out_temp = x+module1_out_temp+module2_out
        module3_out = self.conv_module3(module2_out_temp)
        module3_out_temp = x+module1_out_temp+module2_out_temp+module3_out
        module4_out = self.conv_module4(module3_out_temp)
        module4_out_temp = x+module1_out_temp+module2_out_temp+module4_out
        last_conv = self.last_conv(module4_out_temp)
        out = x + last_conv*self.beta
        return out

class ResidualBlock(nn.Module):
    """
    使用BN层的残差块
    """
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        #need to be identified in every layer
        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(channels)
        self.prelu=nn.PReLU()
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(channels)
        
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.prelu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        
        return x+out
