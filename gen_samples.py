import os

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as transforms

import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets
import cifar10.wide_resnets as wide_resnets
import mnist.model
import pytorch_ssim
import resnet
import Utils
import utils.image_utils as img_utils
import WGAN_GP
from inception_score import inception_score

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
    if not os.path.exists('./results/cifar10/'+ str(G)+'/'):
            os.makedirs('./results/cifar10/'+ str(G)+'/')
    Utils.save_images(samples[:(image_frame_dim * image_frame_dim)+i*64, :, :, :], [image_frame_dim, image_frame_dim],
                      './results/cifar10/'+ str(G)+'/' + str(G)+str(i)+'.png')


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
# target_model = net
# target_model.eval()

#cifar10 resnet
# target_model= cifar_loader.load_pretrained_cifar_resnet(flavor=32)
# target_model = cifar_loader.load_pretrained_cifar_wide_resnet()
# target_model = cifar_loader.load_pretrained_cifar_resnet(flavor=20)
# target_model = wide_resnets.Wide_ResNet(28, 10, 0, 10)


#mnist
# from mnist import model, dataset
# target_model = model.mnist(pretrained=os.path.join(os.path.expanduser('~/.torch/models'), 'mnist.pth'))
net = mnist.model.LeNet5().cuda()
net.load_state_dict(torch.load('./target_models/best.pth'))
target_model = net
target_model.eval()

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

#cifar10
ori_gen_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_resnet32/cifar10/WGAN_GP/oriWGAN_GP3000_G.pkl'
adv_gen_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_resnet32_new_perloss/cifar10/WGAN_GP_ADV/WGAN_GP_ADV450_G.pkl'
imadv_gen_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_resnet32_vgg/cifar10/WGAN_GP_ADV_vgg/WGAN_GP_ADV_vgg_G_best.pkl'
#62,3,32
ori_G = WGAN_GP.generator(input_dim=62, output_dim=3, input_size=32).to(device)
adv_G = WGAN_GP.generator(input_dim=62, output_dim=3, input_size=32).to(device)
imadv_G = WGAN_GP.generator(input_dim=62, output_dim=3, input_size=32).to(device)
#mnist
# ori_gen_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_mnist/mnist/WGAN_GP/oriWGAN_GP850_G.pkl'
# adv_gen_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_mnist_2/mnist/WGAN_GP_ADV/WGAN_GP_ADV20_G.pkl'
# imadv_gen_path = '/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_vgg_mnist_2/2_0_0.6/mnist/WGAN_GP_ADV_vgg/WGAN_GP_ADV_vgg_G_best.pkl'
# ori_G = WGAN_GP.generator(input_dim=62, output_dim=1, input_size=28).to(device)
# adv_G = WGAN_GP.generator(input_dim=62, output_dim=1, input_size=28).to(device)
# imadv_G = WGAN_GP.generator(input_dim=62, output_dim=1, input_size=28).to(device)

ori_G.load_state_dict(torch.load(ori_gen_path))
adv_G.load_state_dict(torch.load(adv_gen_path))
imadv_G.load_state_dict(torch.load(imadv_gen_path))

ori_G.eval()
adv_G.eval()
imadv_G.eval()

# test adversarial examples in cifar10 training dataset
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, 4),
#     transforms.ToTensor(),
#     normalize,
# ])


# test adversarial examples in mnist training dataset
# transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
# train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
# train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)


num_correct_ori = 0
num_correct_adv = 0
num_correct_imadv = 0
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
ori_img_a,adv_image_a,imadv_image_a = 0,0,0
ori_img_s,adv_image_s,imadv_image_s = 0,0,0

for i, data in enumerate(test_dataloader):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    z_ = torch.rand((batch_size, 62))
    z_ = z_.cuda()

    ori_img = ori_G(z_)
    adv_image = adv_G(z_)
    imadv_image = imadv_G(z_)

    
    # pred_lab_ori = torch.argmax(target_model(test_img),1)
    # pred_lab_adv = torch.argmax(target_model(adv_image), 1)
    # pred_lab_imadv = torch.argmax(target_model(imadv_image),1)

    # num_correct_ori += torch.sum(pred_lab_ori == test_label, 0)
    # num_correct_adv += torch.sum(pred_lab_adv == target_label, 0)
    # num_correct_imadv += torch.sum(pred_lab_imadv == target_label, 0)

    visualize_results('ori_G', batch_size,ori_img,i)
    visualize_results('adv_G', batch_size,adv_image,i)
    visualize_results('imadv_G', batch_size,imadv_image,i)
# print('len test dataset',len(test_dataset))
# print('num_correct_imadv: ', num_correct_imadv.item())
# print('num_correct_adv: ', num_correct_adv.item())
# print('num_correct_ori: ', num_correct_ori.item())
# acc_adv = num_correct_adv.item()/len(test_dataset)
# acc_ori = num_correct_ori.item()/len(test_dataset)
# acc_imadv = num_correct_imadv.item()/len(test_dataset)
# print('accuracy of adv imgs in testing set: %f\n' % acc_adv)
# print('accuracy of ori imgs in testing set: %f\n' % acc_ori)
# print('accuracy of imadv imgs in testing set: %f\n' % acc_imadv)
#IS
# z_ = torch.rand(10000,62).cuda()
# ori_img = ori_G(z_)
# adv_image = adv_G(z_)
# imadv_image = imadv_G(z_)
# ori_img_a,ori_img_s = inception_score(ori_img, cuda=True, batch_size=32, resize=True, splits=10)
# adv_image_a,adv_image_s = inception_score(adv_image, cuda=True, batch_size=32, resize=True, splits=10)
# imadv_image_a,imadv_image_s = inception_score(imadv_image, cuda=True, batch_size=32, resize=True, splits=10)
# print('ori_img_a:%f\n',ori_img_a)
# print('adv_image_a:%f\n',adv_image_a)
# print('imadv_image_a:%f\n',imadv_image_a)
# print('ori_img_s:%f\n',ori_img_s)
# print('adv_image_s:%f\n',adv_image_s)
# print('imadv_image_s:%f\n',imadv_image_s)


#FID
os.system("python ./pytorch-fid/fid_score.py /data1 /data2 --gpu 0")

#SSIM
# z_ = torch.rand(20000,62).cuda()
# ori_img = ori_G(z_)
# adv_image = adv_G(z_)
# imadv_image = imadv_G(z_)
# print(pytorch_ssim.ssim(ori_img, adv_image))
# print(pytorch_ssim.ssim(ori_img,imadv_image))
