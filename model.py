import torch.nn as nn
import torch
from torchvision import models


class ResNet(nn.Module):

    def __init__(self, class_num=2, architecture="resnet18", pretrained=True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        model = models.resnet18(pretrained=pretrained)
        fc_input_dim = model.fc.in_features
        # change the dimension of output
        model.fc = nn.Linear(fc_input_dim, class_num)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, inp):
        """predict digit for input"""
        self.eval()
        with torch.no_grad():
            raw_output = self.model(inp)
            _, pred = torch.max(raw_output, 1)
            return pred
