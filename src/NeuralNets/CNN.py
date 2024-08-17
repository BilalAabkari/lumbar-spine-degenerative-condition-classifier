from turtle import Turtle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

class CustomResNet34(nn.Module):
    def __init__(self, num_conditions, num_disc_levels):
        super(CustomResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        
        # Replace the final fully connected layer to match your output requirements
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_conditions * num_disc_levels * 3)
        
        self.num_conditions = num_conditions
        self.num_disc_levels = num_disc_levels
    
    def forward(self, x):
        x = self.resnet(x)
        x = F.softmax(x.view(-1, self.num_disc_levels, self.num_conditions, 3), dim=3)
        return x



class ConvNetV1(nn.Module):
    def __init__(self, size):
        super(ConvNetV1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.linear1 = nn.Linear(512*8*8, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, 75)

    def forward(self, x):
        show_shapes = False

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        if show_shapes:
            print(x.shape)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        if show_shapes:
            print(x.shape)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        if show_shapes:
            print(x.shape)

        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        if show_shapes:
            print(x.shape)

        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        if show_shapes:
            print(x.shape)

        x = F.relu(self.bn6(self.conv6(x)))
        if show_shapes:
            print(x.shape)

        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        if show_shapes:
            print(x.shape)

    

        x = self.linear2(x)
        if show_shapes:
            print(x.shape)

        x = F.softmax(x.view(-1, 5, 5, 3), dim=3)
        return x
    


class ResNetV1(nn.Module):
    def __init__(self, size):
        super(ResNetV1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.linear1 = nn.Linear(512*8*8, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, 75)

    def forward(self, x):
        show_shapes = False



        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        if show_shapes:
            print(x.shape)
        
        residual1 = x

        x = self.bn2(self.conv2(x))
        if show_shapes:
            print(x.shape)

        x += residual1
        x = self.pool2(F.relu(x))

        residual2 = x

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        if show_shapes:
            print(x.shape)

        residual2 = self.pool3(self.conv3(residual2))

        x = self.bn4(self.conv4(x))
        x += self.conv4(residual2)

        x = self.pool4(F.relu(x))

        if show_shapes:
            print(x.shape)

        residual3 = x

        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        if show_shapes:
            print(x.shape)

        x = self.bn6(self.conv6(x))
        if show_shapes:
            print(x.shape)

        x += self.conv6(self.pool5(self.conv5(residual3)))
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        if show_shapes:
            print(x.shape)

    

        x = self.linear2(x)
        if show_shapes:
            print(x.shape)

        x = F.softmax(x.view(-1, 5, 5, 3), dim=3)
        return x