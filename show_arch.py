import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torchvision import models

import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets
import cifar10.wide_resnets as wide_resnets
import mnist.model
import Utils
from dataloader import dataloader
from torchsummary import summary
from vgg import Vgg16
import WGAN_GP_ADV_vgg
import Utils
# g = WGAN_GP_ADV_vgg.generator(input_dim=62, output_dim=3, input_size=32)
# g.cuda()
# d = WGAN_GP_ADV_vgg.discriminator(input_dim=3, output_dim=1, input_size=32)
# d.cuda()
# vgg = Vgg16().cuda()
# model1 = cifar_resnets.resnet20().cuda()
# model2 = cifar_resnets.resnet20().cuda()
# model3 = wide_resnets.Wide_ResNet(28, 10, 0, 10).cuda()
# summary(model1,(3*32*32))
# summary(model2,(3*32*32))
# summary(model3,(3*32*32))

# summary(d,(3,32,32))
lenet = mnist.model.LeNet5()
lenet.cuda()
# Utils.print_network(lenet)
# Utils.print_network(vgg)
summary(lenet,(1,28,28))