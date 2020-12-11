import copy
import logging
import os
import pickle
import time

import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets
import mnist.model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Utils
from dataloader import dataloader
from torch.autograd import Variable, grad
from vgg import Vgg16

dtype = torch.cuda.FloatTensor
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        Utils.initialize_weights(self)

    def forward(self, input):
        # print('ginput',input.size())
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            # nn.Sigmoid(),
        )
        Utils.initialize_weights(self)

    def forward(self, input):
        # print('dinput',input.size())
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class WGAN_GP_ADV_vgg(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.epoch_now = 0
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.lambda_ = 10
        # the number of iterations of the critic per generator iteration
        self.n_critic = 5           
        self.checkpoint = args.checkpoint

        self.loss_adv_avg = 10
        self.loss_content_avg = 10
        self.loss_perturb_avg = 10
        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]
        # vgg net init

        # self.vgg_class = Vgg16()
        # self.vgg_class.cuda()
        # self.vgg_class.eval()

        self.vgg = Vgg16().type(dtype)
        self.vgg.eval()
        # networks init
        self.G_ori = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        # load checkpoint
        if self.checkpoint != '':
            print(self.checkpoint+'G.pkl')
            print(self.checkpoint+'G_optimizer.pkl')
            try:
                op = torch.load(self.checkpoint+'all.tar')
                # print(op)
                # self.G_optimizer.load_state_dict(op['G_optimizer'])
                # self.D_optimizer.load_state_dict(op['D_optimizer'])
                self.epoch_now = op['epoch']
            except:
                print('1')
            # self.G_optimizer.load_state_dict(torch.load(self.checkpoint+'G_optimizer.pkl', map_location='cpu'))
            # self.D_optimizer.load_state_dict(torch.load(self.checkpoint+'D_optimizer.pkl', map_location='cpu'))
            # self.G_ori.load_state_dict(torch.load('/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_resnet32/cifar10/WGAN_GP/oriWGAN_GP3000_G.pkl'))
            self.G_ori.load_state_dict(torch.load('/mnt/ty/pytorch-generative-model-collections/models/WGAN_GP_mnist/mnist/WGAN_GP/oriWGAN_GP850_G.pkl'))
            self.G.load_state_dict(torch.load(self.checkpoint+'G.pkl'))
            self.D.load_state_dict(torch.load(self.checkpoint+'D.pkl'))
        # self.G_ori = copy.deepcopy(self.G)
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.G_ori.cuda()
            self.G_ori.eval()

        print('---------- Networks architecture -------------')
        Utils.print_network(self.G)
        Utils.print_network(self.D)
        print('-----------------------------------------------')
        #cifar10 targeted model
        self.target_model = args.target_model
        if self.target_model=="resnet20":
            self.model= cifar_loader.load_pretrained_cifar_resnet(flavor=20)
        elif self.target_model=="resnet32":
            self.model = cifar_loader.load_pretrained_cifar_resnet(flavor=32)
        elif self.target_model=="wideresnet":
            self.model = cifar_loader.load_pretrained_cifar_wide_resnet()
        elif self.target_model=="mnist_1":
            self.model = mnist.model.mnist()
            self.model.load_state_dict(torch.load('./mnist/mnist.pth'))
        elif self.target_model=="mnist_2":
            self.model = mnist.model.LeNet5()
            self.model.load_state_dict(torch.load('./target_models/best.pth'))
        #adv train
        # model = cifar_resnets.resnet32()
        # model.load_state_dict(torch.load('./advtrain.resnet32.000100.path.tar'))
        #mnist
        # from mnist import model, dataset
        # self.model = model.mnist(pretrained=os.path.join(os.path.expanduser('~/.torch/models'), 'mnist.pth'))


        self.model = self.model.cuda()
        self.model.eval()
        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()
        if not os.path.exists(os.path.join(self.save_dir, self.dataset, self.model_name)):
            print('make dir and log:',os.path.join(self.save_dir, self.dataset, self.model_name))
            os.makedirs(os.path.join(self.save_dir, self.dataset, self.model_name))
            os.mknod(os.path.join(self.save_dir, self.dataset, self.model_name,self.model_name+'log.txt'))
        logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(self.save_dir, self.dataset, self.model_name,self.model_name+'log.txt'),
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        epoch_now = self.epoch_now
        for epoch in range(epoch_now,self.epoch):
            self.G.train()
            loss_adv_sum = 0
            loss_content_sum = 0
            loss_perturb_sum = 0

            adv_lambda = 2
            pert_lambda = 0
            content_lambda = 0.6

            epoch_start_time = time.time()
            for iter, (x_, labels) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()
                
                D_real = self.D(x_)
                D_real_loss = -torch.mean(D_real)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(D_fake)

                # gradient penalty
                alpha = torch.rand((self.batch_size, 1, 1, 1))
                if self.gpu_mode:
                    alpha = alpha.cuda()

                x_hat = alpha * x_.data + (1 - alpha) * G_.data
                x_hat.requires_grad = True

                pred_hat = self.D(x_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = D_real_loss + D_fake_loss + gradient_penalty

                D_loss.backward()
                self.D_optimizer.step()

                if ((iter+1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.item())

                    # G_loss.backward()
                    
                    # adv part
                    # cw loss
                    # loss_content = torch.mean(torch.norm(
                    #     G_.view(G_.shape[0], -1), 2, dim=1))
                    ori = self.G_ori(z_)
                    loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)
                    
                    loss_perturb = loss_mse(ori,G_) 

                    if self.target_model=="mnist_2":
                        ori_ex = ori
                        ori_ex = ori_ex.repeat(1, 3, 1, 1)
                        G_ex = G_
                        G_ex = G_ex.repeat(1, 3, 1, 1)
                        ori_ex = Variable(ori_ex).type(dtype)
                        ori_cont_features = self.vgg(ori_ex)
                        G_cont_features = self.vgg(G_ex)
                    else:
                        ori = Variable(ori).type(dtype)
                        ori_cont_features = self.vgg(ori)
                        G_cont_features = self.vgg(G_)
                    loss_content = loss_mse(ori_cont_features[1],G_cont_features[1]) * 100

                    # loss_class = torch.mean(self.vgg_class(G_))

                    # C = 0.1
                    # loss_content = torch.max(
                    #     loss_content - C, torch.zeros(1, device='cuda'))
                    # adv loss
                    logits_model = self.model(G_)
                    probs_model = F.softmax(logits_model, dim=1)
                    label4 = [0]*self.batch_size
                    target_label = torch.LongTensor(label4)
                    target_label = target_label.cuda()
                    onehot_labels = torch.eye(10, device='cuda')[target_label]
                    
                    real = torch.sum(onehot_labels * probs_model, dim=1)
                    other, _ = torch.max(
                        (1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
                    zeros = torch.zeros_like(other)
                    loss_adv = torch.max(other - real, zeros)
                    loss_adv = torch.sum(loss_adv)*4
                    # loss_adv = -F.mse_loss(logits_model, onehot_labels)
                    
                    # loss_G_adv = adv_lambda * loss_adv + pert_lambda * loss_content
                    loss_G_adv = adv_lambda * loss_adv + content_lambda * loss_content + pert_lambda * loss_perturb
                    # G_loss_total = 0.4*loss_G_adv + G_loss
                    G_loss_total = loss_G_adv + G_loss
                    
                    loss_adv_sum += loss_adv.item()
                    loss_perturb_sum += loss_perturb.item()
                    loss_content_sum += loss_content.item()

                    G_loss_total.backward()
                    
                    self.G_optimizer.step()
                    self.train_hist['D_loss'].append(D_loss.item())

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
                    logging.info("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
            if (epoch) % 5 == 0:
                self.save(epoch=epoch+1)
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))
            
            loss_adv_avg_temp = loss_adv_sum/self.data_loader.dataset.__len__()
            loss_content_avg_temp = loss_content_sum/self.data_loader.dataset.__len__()
            loss_perturb_avg_temp = loss_perturb_sum/self.data_loader.dataset.__len__()
            print(loss_adv_avg_temp, loss_content_avg_temp,loss_perturb_sum)
            print(self.loss_adv_avg, self.loss_content_avg,self.loss_perturb_avg)
            logging.info("loss_adv_avg_temp:%.8f,loss_content_avg_temp:%.8f"%(loss_adv_avg_temp,loss_content_avg_temp))
            if self.loss_adv_avg >= loss_adv_avg_temp and self.loss_content_avg >= loss_content_avg_temp:
                self.loss_adv_avg = loss_adv_avg_temp
                self.loss_content_avg = loss_content_avg_temp
                print('save best:',epoch)
                logging.info("loss_adv_avg_temp:%.8f,loss_content_avg_temp:%.8f"%(loss_adv_avg_temp,loss_content_avg_temp))
                self.save(best=True, epoch=epoch+1)
               

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        Utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        Utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        Utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self, best=False, epoch=0):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save({
            'epoch':epoch,
            'G':self.G.state_dict(),
            'D':self.D.state_dict(),
            'G_optimizer':self.G_optimizer.state_dict(),
            'D_optimizer':self.D_optimizer.state_dict()
        },os.path.join(
                save_dir, self.model_name + str(epoch) + '_all.tar'))
        torch.save(self.G_optimizer.state_dict(),os.path.join(
                save_dir, self.model_name + str(epoch) + '_G_optimizer.pkl'))
        torch.save(self.D_optimizer.state_dict(),os.path.join(
                save_dir, self.model_name + str(epoch) + '_D_optimizer.pkl'))
        if best == True:
            # with open(os.path.join(save_dir, self.model_name + 'best.txt'), 'w') as f:
            #     f.write(str(self.loss_adv_avg)+'\n'+str(self.loss_content_avg))
            torch.save(self.G.state_dict(), os.path.join(
                save_dir, self.model_name + '_G_best.pkl'))
            torch.save(self.D.state_dict(), os.path.join(
                save_dir, self.model_name + '_D_best.pkl'))
        else:
            torch.save(self.G.state_dict(), os.path.join(
                save_dir, self.model_name + str(epoch) + '_G.pkl'))
            torch.save(self.D.state_dict(), os.path.join(
                save_dir, self.model_name + str(epoch) + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
