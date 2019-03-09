import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ShuffleNetV2 import shufflenetv2, shufflenetFeature
from model.MobileNetV2 import mobileNetFeature
from model.resnet import resnet50AvgPooling

class Model(nn.Module): 
  def __init__(self, net, path_to_predefined='', pretrained=True, last_conv_stride=1):
    super(Model, self).__init__()
    if net == 'shuffelnetV2':
      model = shufflenetv2(path_to_predefined_model=path_to_predefined, pretrained=pretrained)
      self.base = shufflenetFeature(model)
    elif net == 'mobilenetV2':
      model = mobileNetFeature(path_to_predefined_model=path_to_predefined, pretrained=pretrained)
      self.base = model
    else:
      model = resnet50AvgPooling(pretrained=True, last_conv_stride=last_conv_stride)
      self.base = model

  def forward(self, x):
    # shape [N, C, H, W]
    x = self.base(x)
    #
    # shape [N, C]
    x = x.view(x.size(0), -1)

    return x


'''
from .resnet import resnet50


class Model(nn.Module):
  def __init__(self, last_conv_stride=2):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)

  def forward(self, x):
    # shape [N, C, H, W]
    x = self.base(x)
    x = F.avg_pool2d(x, x.size()[2:])
    # shape [N, C]
    x = x.view(x.size(0), -1)

    return x
'''