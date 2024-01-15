import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
#import util.util as util

class InnerCos(nn.Module):
    def __init__(self):
        super(InnerCos, self).__init__()
        self.criterion = nn.L1Loss()
        self.target = None
        self.down_model = nn.Sequential(
          #  nn.Conv2d(3, 3, kernel_size=1,stride=1, padding=0),
            nn.Tanh()
        )

    def set_target(self, targetst):
        self.targetst = F.interpolate(targetst, size=(32, 32), mode='bilinear')
     #   self.targetde = F.interpolate(targetde, size=(32, 32), mode='bilinear')

    def get_target(self):
        return self.target

    def forward(self, in_data):
        loss_co = in_data[1]
        self.Out = self.down_model(loss_co[0])
        self.loss = self.criterion(self.Out, self.targetst)
        self.output = in_data[0]
        return self.output

    def backward(self, retain_graph=True):

        self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def __repr__(self):

        return self.__class__.__name__