import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
import torchvision.datasets as dset
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import cv2
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from math import ceil,floor
import random
from scipy.io import loadmat
from model.aspp import *
from model.resnet import *
from model.deeplabv3 import *
import matplotlib.pyplot as plt

#visualization
model = torch.load('checkpoint/models/model4.pth') #pretrained model
model.eval()
model = model.to('cuda:0')

imgs = os.listdir('data_reduced_test')
imgs.sort(key=lambda x:x[4:-4])
img = Image.open('data_reduced_test/'+imgs[10]).convert('RGB')  #change the index to any image

trans = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
img = trans(img)
pre = model(img.unsqueeze(0).to('cuda:0'))

array = pre.detach().cpu().numpy()[0,:,:,:].argmax(0)
for i in range(1,14):
    array[array == i] = i*19
plt.imshow(array,cmap='tab20')
# plt.savefig('example.jpg')

#learning curve
with open('log3.txt','r') as f:
    data = f.readlines()
for d in data:
    d = d.split(' ')
    train_loss.append(float(d[0]))
    test_loss.append(float(d[1]))
    train_acc.append(float(d[2]))
    test_acc.append(float(d[3][:-1]))

plt.figure()
plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(['training loss','test loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.figure()
plt.plot(train_acc)
plt.plot(test_acc)
plt.legend(['training accuracy','test accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
