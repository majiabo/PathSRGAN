import torch
import torch.nn as nn

input_dim = 3
filters=[64, 128, 256, 512, 1024]


class encoder(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_flag):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.dropout_flag = dropout_flag
        if dropout_flag:
        	self.dropout = nn.Dropout2d(p=0.5) 

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        if self.dropout_flag:
            y = self.dropout(y) 
        return y

class decoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='nearest')
        self.up_conv = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=2, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=output_channel*2, out_channels=output_channel, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU()

    def forward(self, x , skip):
        '''
        x为之前decoder的输出，skip为通过连接得到的特征图
        '''
        y = self.upsample(x)
        y = self.up_conv(y)
        y = y[:,:,1:,1:]
        y = self.relu1(y)
        y = torch.cat([y, skip], dim=1)
        y = self.conv1(y)
        y = self.relu2(y)
        y = self.conv2(y)
        y = self.relu3(y)
        return y


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        
        self.encoder1 = encoder(input_channel=input_dim, output_channel=filters[0], dropout_flag=False)
        self.encoder2 = encoder(input_channel=filters[0], output_channel=filters[1], dropout_flag=False)
        self.encoder3 = encoder(input_channel=filters[1], output_channel=filters[2], dropout_flag=False)
        self.encoder4 = encoder(input_channel=filters[2], output_channel=filters[3], dropout_flag=True)
        
        self.center = encoder(input_channel=filters[3], output_channel=filters[4], dropout_flag=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2,2))

        self.decoder1 = decoder(input_channel=filters[4], output_channel=filters[3])
        self.decoder2 = decoder(input_channel=filters[3], output_channel=filters[2])
        self.decoder3 = decoder(input_channel=filters[2], output_channel=filters[1])
        self.decoder4 = decoder(input_channel=filters[1], output_channel=filters[0])
        
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=filters[0], out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.maxpool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.maxpool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.maxpool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.maxpool4(e4)

        center = self.center(p4)

        d1 = self.decoder1(center, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e1)

        y = self.final(d4)
        
        return y
