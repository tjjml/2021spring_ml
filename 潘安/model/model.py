import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import cv2
from PIL import Image
classes = []

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.conv3 = nn.Conv2d(16,30,4)
        self.fnc1 = nn.Linear(144,100)
        self.fnc2 = nn.Linear(100,80)
        self.fnc3 = nn.Linear(80,65)
    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def forward(self,x):
        x = F.relu(self.conv1(x)) #size (20-5+0)/1+1=16
        x = F.max_pool2d(x,(2,2)) 
        x = F.relu(self.conv2(x)) #size (28-5+0)/1+1 = 24
        x = F.max_pool2d(x,(2,2)) 
        #x = F.relu(self.conv3(x)) #size (24-5+0)/1+1 = 20
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fnc1(x))
        x = F.relu(self.fnc2(x))
        x = self.fnc3(x)
        return x



class DanTongDao(object):
    def __call__(self, pics):
        pics = np.array(pics)
        #pics = cv2.resize(pics, (20, 20), interpolation=cv2.INTER_AREA)
        pics = cv2.cvtColor(pics,cv2.COLOR_BGR2GRAY)
        pics = cv2.copyMakeBorder(pics, 0, 0, 5, 5, cv2.BORDER_CONSTANT, value = [0,0,0])
        PIL_image = Image.fromarray(pics) 
        return PIL_image
    def __repr__(self):
        return self.__class__.__name__ + '()'


#数据加载
data_transform = transforms.Compose([
        DanTongDao(),
        transforms.CenterCrop((20,20)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,],
                             std=[0.5,])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='./DataSet/cnn_char_train/',
                                           transform=data_transform)
print(len(hymenoptera_dataset))
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=8, shuffle=True,
                                             num_workers=0)



#模型
net = Net()
#损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#训练
if __name__ == "__main__":
    for epoch in range(20):
        running_loss = 0.0
        for i ,data in enumerate(dataset_loader):
            inputs,labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            #print statistics
            running_loss+=loss.item()
            #print every 2000 mini-batch
            if i%200 ==199:
                print('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss/200))
                running_loss=0.0
    print('Finished Training')
    # 保存
    torch.save(net, './model/net2.pth')
    # 加载
    # model = torch.load('\model.pth')


