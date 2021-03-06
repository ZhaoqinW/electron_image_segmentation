# source https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
# Unet for biomedical image: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))   

def dice_loss(pred, target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

class diceloss(torch.nn.Module):
    def init(self):
        super(diceLoss, self).init()
    def forward(self,pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class Unet(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)

        return out
    
    def loss(self, predicted, target,device):
        #predicted = torch.softmax(predicted,dim=1)[:,0]
        #target = torch.nn.functional.one_hot(target, 2).transpose(1, 4).squeeze(-1)
        target = torch.reshape(target, [-1,960,960])
        weights = [0.2,1]
        class_weights = torch.FloatTensor(weights)
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
        #return dice_loss(predicted, target)
        return criterion(predicted, target)
