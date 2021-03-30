import torch
from torch import nn
from torchvision.models.vgg import vgg16
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
from torchvision.models import resnet50

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self,out_labels, out_images, target_images):
        # Adversarial Loss
        valid = Variable(torch.cuda.FloatTensor(out_images.size(0),1).fill_(1.0),requires_grad = False)
        adversarial_loss = nn.BCELoss()(out_labels,valid)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return [image_loss,adversarial_loss,perception_loss,tv_loss]

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class G_loss_vgg(nn.Module):
    """
    5:第一次maxpool后
    10：第二次maxpool后
    17：第三次maxpool后
    24：第四次maxpool后
    31：第五次maxpool后
    only for single GPU
    ps:2019.1.15
    损失函数加权,只对image mse加权

    """
    def __init__(self,floor):
        super(G_loss_vgg, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:floor]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        #self.tv_loss = TVLoss()

    def forward(self,out_labels, out_images, target_images,weight_map):
        # Adversarial Loss
        valid = Variable(torch.cuda.FloatTensor(out_images.size(0),1).fill_(1.0),requires_grad = False)
        adversarial_loss = nn.BCELoss()(out_labels,valid)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        out_images = torch.mul(out_images,weight_map)
        target_images = torch.mul(target_images,weight_map)
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        #tv_loss = self.tv_loss(out_images)
        #return [image_loss,adversarial_loss,perception_loss,tv_loss]    
        return [image_loss,adversarial_loss,perception_loss]  

class PerceptionLoss(nn.Module):
    """
    5:第一次maxpool后
    10：第二次maxpool后
    17：第三次maxpool后
    24：第四次maxpool后
    31：第五次maxpool后
    """
    def __init__(self,floor):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:floor]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self,out_images, target_images):

        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        return perception_loss


class PLossResnet(nn.Module):
    '''
    output: 
            resize 256 to 512
            classifer_labels: float type in the range[0,1] list with batch_size
            feature map: in the last relu layer before avgpool and output_layer
    '''
    def __init__(self):
        super(PLossResnet, self).__init__()
        classifer=resnet50(pretrained=None)
        resnet_layer=nn.Sequential(*list(classifer.children())[:-2])
        for param in resnet_layer.parameters():
            param.requires_grad = False
        
        self.loss_network = resnet_layer
        self.mse_loss = nn.MSELoss()
    def forward(self,x,y):
        x = self.loss_network(x)
        y = self.loss_network(y)
        out = self.mse_loss(x,y)
        return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
