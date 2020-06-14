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


#dataset for training
class dataset(Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.transform = transform
        imgs,mats = os.listdir(path+'data_reduced_train'),os.listdir(path+'labels_reduced_train2')
        imgs,mats = [path+'data_reduced_train/'+img for img in imgs],[path+'labels_reduced_train2/'+mat for mat in mats]
        imgs.sort(key=lambda x:x[4:-4])
        mats.sort(key=lambda x:x[12:-4])
        self.mats =  mats
        self.images = imgs
        

    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image=  self.transform(image)     
        mat_ori = cv2.imread(self.mats[idx])[:,:,0].astype(np.long) #hxw        
        return (image,mat_ori)
#dataset for validation
class dataset2(Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.transform = transform
        imgs,mats = os.listdir('data_reduced_test'),os.listdir('labels_reduced_test')
        imgs,mats = ['data_reduced_test/'+img for img in imgs],['labels_reduced_test/'+mat for mat in mats]
        imgs.sort(key=lambda x:x[4:-4])
        mats.sort(key=lambda x:x[12:-4])
        self.mats =  mats
        self.images = imgs
        

    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image = Image.open(self.images[idx]).convert('RGB')
#         name = self.mats[idx].split('/')[-1]
        if self.transform:
            image=  self.transform(image)
#         mat = np.load(self.mats[idx]).astype(np.float32)      
        mat_ori = cv2.imread(self.mats[idx])[:,:,0].astype(np.long) #hxw
#         mat = np.moveaxis(mat,2,0)
        
        return (image,mat_ori)
#compute accuracy
def check_acc(model,loader,criterion,batch_size=1):
    model.eval()
    total_p = 0
    wrong_p = 0
    loss_all = 0
    total_num = 0
    for (imgs,mats_ori) in loader:
        imgs= imgs.to(device)
#         mats =mats.to(device)
        mats_ori = mats_ori.to(device)
        height,width = imgs.size(2),imgs.size(3)
        y_pre = model(imgs)
        loss =  criterion(y_pre,mats_ori).item()
        loss_all += loss * imgs.size(0)
        y_pre = y_pre.detach().cpu().numpy()
        y_pre = np.argmax(y_pre,axis=1)
        wrong_p += np.sum(y_pre != mats_ori.cpu().numpy())
        total_p += height*width*imgs.size(0)
        total_num += imgs.size(0)
    return (1-wrong_p/total_p,loss_all/total_num)

def count_wrong(input,target):
    input = input.clone().detach().cpu().numpy()
    target = target.clone().detach().cpu().numpy() #hxw
    input = np.argmax(input,axis=1)
    return np.sum(input != target)

#training
def train(model,loader_train,loader_val,optimizer,criterion,scheduler=None,epochs=1,batch_size=32):
    model = model.to(device)
    losses_train,losses_val,acces_train,acces_val = [],[],[],[]
    for e in range(epochs):
        print('begin epoch :',e+1)
        model.train()
        wrong = 0
        total = 0
        running_loss = 0
        total_num = 0
        for (imgs,mats_ori) in tqdm(loader_train):
            imgs = imgs.to(device)
#             mats = mats.to(device)
            mats_ori = mats_ori.to(device)
            y_pre = model(imgs) # bx38xhxw
            loss = criterion(y_pre,mats_ori)
            running_loss += loss.item()*batch_size
            wrong += count_wrong(y_pre,mats_ori)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += y_pre.size(0)*y_pre.size(2) * y_pre.size(3)            
            total_num += imgs.size(0)
        running_loss = running_loss/ total_num
        (acc_val,loss_val) = check_acc(model,loader_val,criterion,batch_size)
        losses_val.append(loss_val)
        losses_train.append(running_loss)
        acc_train = 1-wrong/total
        acces_train.append(acc_train)
        acces_val.append(acc_val)
        torch.save(model,'checkpoint/models/'+'model3.pth')
        print(running_loss)
        print(acc_train)
        print(acc_val)
        with open('checkpoint/log3.txt','a') as f:
            f.write(str(running_loss)+' ')
            f.write(str(loss_val)+' ')
            f.write(str(acc_train) + ' ')
            f.write(str(acc_val)+'\n')

#         if e in [10]:
#             if e == 8:
#                 optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5
#             else :
#                 optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            
        if scheduler:
            scheduler.step(running_loss)
        print('end epoch :', e+1)
        print('  '+'loss = ',running_loss)
    with open('checkpoint/losses3.txt','a') as f:
        for i in range(len(acces_train)):
            f.write(str(losses_train[i])+' ')
            f.write(str(losses_val[i])+' '+'\n')
    with open('checkpoint/acc3.txt','a') as f:
        for i in range(len(acces_train)):
            f.write(str(acces_train[i])+' ')
            f.write(str(acces_val[i])+'\n')
    
if __name__ == '__main__':
    #main function
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size=6
    transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    # dset_train = dataset('data/training',transform)
    dset_val = dataset2('val',transform)
    dset_train = dataset('',transform)
    model = DeepLabV3(0,'')
#     model = torch.load('checkpoint/models/model3.pth')
    loader_train,loader_val = DataLoader(dset_train,shuffle=True,batch_size=batch_size),DataLoader(dset_val,shuffle=False,batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(),lr = 1e-2,weight_decay = 0.0001)
#     criterion = custom_loss()
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,verbose=True,patience=1,min_lr=1e-6)
    train(model,loader_train,loader_val,optimizer,criterion,scheduler=scheduler,epochs=100,batch_size=batch_size)
    
        
