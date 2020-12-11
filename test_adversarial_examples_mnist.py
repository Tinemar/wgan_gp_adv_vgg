import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import numpy as np
import torch.nn.functional as F
import utils.image_utils as img_utils

import os
import WGAN_GP
import resnet
import Utils
import cifar10.cifar_resnets as cifar_resnets
import cifar10.wide_resnets as wide_resnets
import cifar10.cifar_loader as cifar_loader
import mnist.model
use_cuda = True
image_nc = 1
batch_size = 100


def visualize_results(G, batch_size,adv_images,i):
    # G.eval()
    tot_num_samples = 100
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    # print(image_frame_dim)
    # # sample_z_ = torch.rand((64, 2352))
    # sample_z_ = torch.rand((batch_size, 62))
    # sample_z_ = sample_z_.cuda()
    # samples = G(sample_z_)

    samples = adv_images.cpu().data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2
    Utils.save_images(samples[:(image_frame_dim * image_frame_dim)+i*64, :, :, :], [image_frame_dim, image_frame_dim],
                      './results/cifar10/test/WGAN_GP'+str(i)+'.png')


# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (
    use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
# net = resnet.ResNet18()
# net = net.cuda()
# net = torch.nn.DataParallel(net)
# checkpoint = torch.load("H:/pytorch-cifar/checkpoint/DataPackpt.pth")
# net.load_state_dict(checkpoint['net'])
net = mnist.model.LeNet5().cuda()
net.load_state_dict(torch.load('./target_models/best.pth'))
target_model = net
target_model.eval()

#cifar10 resnet
# target_model= cifar_loader.load_pretrained_cifar_resnet(flavor=32)
# target_model = cifar_loader.load_pretrained_cifar_wide_resnet()
# target_model = cifar_loader.load_pretrained_cifar_resnet(flavor=20)
# target_model = wide_resnets.Wide_ResNet(28, 10, 0, 10)


#mnist
# from mnist import model, dataset
# target_model = model.mnist(pretrained=os.path.join(os.path.expanduser('~/.torch/models'), 'mnist.pth'))

#advtrain model
# target_model.load_state_dict(torch.load('./target_models\PGDadv_trained_resnet32_197.pkl'))
# target_model.load_state_dict(torch.load('./target_models\PGDadv_trained_wideresnet_68.pkl'))
# target_model.load_state_dict(torch.load('./target_models\FGSMtrain.resnet20.000100.path.tar'))
# target_model.load_state_dict(torch.load('./target_models\FGSMtrain.resnet32.000100.path.tar'))
# target_model.load_state_dict(torch.load('./target_models\PGDadv_trained_resnet20_149.pkl'))
# target_model = mnist.model.LeNet5()
# target_model.load_state_dict(torch.load('adv_trained_lenet5.pkl'))

target_model = target_model.cuda()
target_model.eval()
# load the generator of adversarial examples
#图3.1
# pretrained_generator_path = './models/WGAN_GP_resnet32/cifar10/WGAN_GP_ADV/WGAN_GP_ADV1450_G.pkl'

# pretrained_generator_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_resnet32_vgg_from_begin/cifar10/WGAN_GP_ADV_vgg/WGAN_GP_ADV_vgg_G_best.pkl'
# pretrained_generator_path = './models/WGAN_GP_wideresnet/cifar10/WGAN_GP/WGAN_GP_G_best.pkl'
# pretrained_generator_path = './models/WGAN_GP_resnet32/cifar10/WGAN_GP/WGAN_GP300_G.pkl'
# pretrained_generator_path = './models/cifar10/WGAN_GP_resnet20/WGAN_GP_G_best.pkl'

pretrained_generator_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_vgg_mnist_2/2_0_0.6/mnist/WGAN_GP_ADV_vgg/WGAN_GP_ADV_vgg_G_best.pkl'
# pretrained_generator_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_mnist_2/mnist/WGAN_GP_ADV/WGAN_GP_ADV20_G.pkl'

# pretrained_G = WGAN_GP.generator(input_dim=62, output_dim=3, input_size=32).to(device)
pretrained_G = WGAN_GP.generator(input_dim=62, output_dim=1, input_size=28).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in cifar10 training dataset
# transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

num_correct = 0
num_correct_ori = 0
label4 = [0]*batch_size
# target_label = torch.LongTensor(64).zero_()
target_label = torch.LongTensor(label4)
target_label = target_label.cuda()


# test adversarial examples in cifar10 testing dataset
# test_dataset = torchvision.datasets.CIFAR10(
#     './data/cifar10', train=False, transform=transform, download=True)
# test_dataloader = DataLoader(
#     test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# test adversarial examples in mnist testing dataset
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset,
    batch_size=batch_size, shuffle=True)

num_correct = 0
num_correct_ori = 0
for i, data in enumerate(test_dataloader):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    z_ = torch.rand((batch_size, 62))
    z_ = z_.cuda()

    adv_img = pretrained_G(z_)
    pred_lab = torch.argmax(target_model(adv_img), 1)
    pred_lab_ori = torch.argmax(target_model(test_img),1)
    num_correct += torch.sum(pred_lab == target_label, 0)
    num_correct_ori += torch.sum(pred_lab_ori == test_label, 0)
    visualize_results(pretrained_G, batch_size,adv_img,i)
    # except:
    #     break
print('len test dataset',len(test_dataset))
print('num_correct: ', num_correct.item())
print('num_correct_ori: ', num_correct_ori.item())
acc_adv = num_correct.item()/len(test_dataset)
acc_ori = num_correct_ori.item()/len(test_dataset)
print('accuracy of adv imgs in testing set: %f\n' % acc_adv)
print('accuracy of ori imgs in testing set: %f\n' % acc_ori)
