
# from vgg import Vgg16

# import cv2
# import numpy as np
# import torch
# from torch.autograd import Variable
# from torchvision import models

# #创建30个文件夹
# def mkdir(path):  # 判断是否存在指定文件夹，不存在则创建
#     # 引入模块
#     import os

#     # 去除首位空格
#     path = path.strip()
#     # 去除尾部 \ 符号
#     path = path.rstrip("\\")

#     # 判断路径是否存在
#     # 存在     True
#     # 不存在   False
#     isExists = os.path.exists(path)

#     # 判断结果
#     if not isExists:
#         # 如果不存在则创建目录
#         # 创建目录操作函数
#         os.makedirs(path)
#         return True
#     else:

#         return False


# def preprocess_image(cv2im, resize_im=True):
#     """
#         Processes image for CNNs

#     Args:
#         PIL_img (PIL_img): Image to process
#         resize_im (bool): Resize to 224 or not
#     returns:
#         im_as_var (Pytorch variable): Variable that contains processed float tensor
#     """
#     # mean and std list for channels (Imagenet)
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     # Resize image
#     if resize_im:
#         cv2im = cv2.resize(cv2im, (224, 224))
#     im_as_arr = np.float32(cv2im)
#     im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
#     im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
#     # Normalize the channels
#     for channel, _ in enumerate(im_as_arr):
#         im_as_arr[channel] /= 255
#         im_as_arr[channel] -= mean[channel]
#         im_as_arr[channel] /= std[channel]
#     # Convert to float tensor
#     im_as_ten = torch.from_numpy(im_as_arr).float()
#     # Add one more channel to the beginning. Tensor shape = 1,3,224,224
#     im_as_ten.unsqueeze_(0)
#     # Convert to Pytorch variable
#     im_as_var = Variable(im_as_ten, requires_grad=True)
#     return im_as_var


# class FeatureVisualization():
#     def __init__(self,img_path,selected_layer):
#         self.img_path=img_path
#         self.selected_layer=selected_layer
#         self.pretrained_model = models.vgg16(pretrained=True).features
#         #print( self.pretrained_model)
#     def process_image(self):
#         img=cv2.imread(self.img_path)
#         img=preprocess_image(img)
#         return img

#     def get_feature(self):
#         # input = Variable(torch.randn(1, 3, 224, 224))
#         input=self.process_image()
#         print("input shape",input.shape)
#         x=input
#         for index,layer in enumerate(self.pretrained_model):
#             #print(index)
#             #print(layer)
#             x=layer(x)
#             if (index == self.selected_layer):
#                 return x

#     def get_single_feature(self):
#         features=self.get_feature()
#         print("features.shape",features.shape)
#         feature=features[:,0,:,:]
#         print(feature.shape)
#         feature=feature.view(feature.shape[1],feature.shape[2])
#         print(feature.shape)
#         return features

#     def save_feature_to_img(self):
#         #to numpy
#         features=self.get_single_feature()
#         for i in range(features.shape[1]):
#             feature = features[:, i, :, :]
#             feature = feature.view(feature.shape[1], feature.shape[2])
#             feature = feature.data.numpy()
#             # use sigmod to [0,1]
#             feature = 1.0 / (1 + np.exp(-1 * feature))
#             # to [0,255]
#             feature = np.round(feature * 255)
#             print(feature[0])
#             mkdir('./feature/' + str(self.selected_layer))
#             cv2.imwrite('./feature/'+ str( self.selected_layer)+'/' +str(i)+'.jpg', feature)
# if __name__=='__main__':
#     # get class
#     for  k in range(30):
#         myClass=FeatureVisualization('/mnt/ty/pytorch-generative-model-collections/feature/20190102161337613.jpg',k)
#         # print (myClass.pretrained_model)
#         myClass.save_feature_to_img()

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
 
def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var
 
 
class FeatureVisualization():
    def __init__(self,img_path,selected_layer):
        self.img_path=img_path
        self.selected_layer=selected_layer
        self.pretrained_model = models.vgg16(pretrained=True).features
 
    def process_image(self):
        img=cv2.imread(self.img_path)
        img=preprocess_image(img)
        return img
 
    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()
        print(input.shape)
        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            if (index == self.selected_layer):
                return x
 
    def get_single_feature(self):
        features=self.get_feature()
        print(features.shape)
 
        feature=features[:,0,:,:]
        print(feature.shape)
 
        feature=feature.view(feature.shape[1],feature.shape[2])
        print(feature.shape)
 
        return feature
 
    def save_feature_to_img(self):
        #to numpy
        feature=self.get_single_feature()
        feature=feature.data.numpy()
 
        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))
 
        # to [0,255]
        feature=np.round(feature*255)
        print('./feature/'+str(self.selected_layer)+'img.jpg')
 
        cv2.imwrite('./feature/'+str(self.selected_layer)+'img.jpg',feature)
 
 
 
 
if __name__=='__main__':
    # get class
    for i in [2,8,15,22]:
        print(i)
        myClass=FeatureVisualization('/mnt/ty/pytorch-generative-model-collections/feature/feature.jpg',i)
        print (myClass.pretrained_model)
        myClass.save_feature_to_img()
    
 
    
