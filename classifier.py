import torch
import torch.nn as nn
from torchvision import models


def Convblock(in_channel, out_channel, norm_layer=True):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True))


class Custom_vgg(nn.Module):
    def __init__(self, in_channel, out_dim, device=torch.device('cpu')):
        super(Custom_vgg, self).__init__()
        self.device = device
        self.convs1 = nn.Sequential(Convblock(in_channel, 32),
                                    Convblock(32, 32)).to(self.device)

        self.pool1 = nn.MaxPool2d(2, stride=2).to(self.device)
        self.convs2 = nn.Sequential(Convblock(32, 64),
                                    Convblock(64, 64)).to(self.device)

        self.pool2 = nn.MaxPool2d(2, stride=2).to(self.device)
        self.convs3 = nn.Sequential(Convblock(64, 128),
                                    Convblock(128, 128),
                                    nn.Dropout(0.33),
                                    Convblock(128, 128)).to(self.device)

        self.pool3 = nn.MaxPool2d(2, stride=2).to(self.device)
        self.flat = torch.nn.Flatten().to(self.device)
        self.FC512 = nn.Sequential(nn.Linear(4608, 512), nn.ReLU(True), nn.Dropout(0.33)).to(self.device)
        self.FC256 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True)).to(self.device)
        self.FCOUT = nn.Sequential(nn.Linear(256, out_dim)).to(self.device)

    def forward(self, x):
        x = self.convs1(x)
        x = self.pool1(x)
        x = self.convs2(x)
        x = self.pool2(x)
        x = self.convs3(x)
        x = self.pool3(x)
        x = self.flat(x)
        x = self.FC512(x)
        x = self.FC256(x)
        x = self.FCOUT(x)
        return x

    def readable_output(self, x, cats):
        softmax = nn.Softmax(dim=1).to(self.device)
        y = softmax(self.forward(x))[0]
        for i, cat in enumerate(cats):
            print("Le visage appartient à la categorie {} à {}%".format(cat, round(float(100 * y[i]), 2)))

    def predict_single(self, x):
        softmax = nn.Softmax(dim=1).to(self.device)
        y = softmax(self.forward(x))[0]
        return [round(float(100 * proba), 2) for proba in y]
