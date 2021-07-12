import cv2
import torch
import os
from model.model import Net
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image



img = cv2.imread("./DataSet/cnn_char_train/0/4-3.jpg")
model = torch.load("./model/net2.pth")
classes = os.listdir("./DataSet/cnn_char_train")

#数据加载
test_data_transform = transforms.Compose([
        transforms.Resize(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
test_hymenoptera_dataset = datasets.ImageFolder(root='./DataSet/cnn_char_train/',
                                           transform=test_data_transform)
test_dataset_loader = torch.utils.data.DataLoader(test_hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=0)
dataiter = iter(test_dataset_loader)
images, labels = dataiter.next()

class DanTongDao(object):
    def __call__(self, pics):
        pics = np.array(pics)
        pics = cv2.cvtColor(pics,cv2.COLOR_BGR2GRAY)
        PIL_image = Image.fromarray(pics) 
        return PIL_image
    def __repr__(self):
        return self.__class__.__name__ + '()'

if __name__ == "__main__":

    data_transform = transforms.Compose([
            DanTongDao(),
            transforms.Resize((20,20)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ],
                                std=[0.5])
        ])

    image = cv2.imread("./DataSet/cnn_char_train/9/3-1.jpg")
    print(image.shape)
    PIL_image = Image.fromarray(image) 
    image = data_transform(PIL_image)
    image = image.reshape(1,1,20,20)
    with torch.no_grad():
        out = model(image)
        _,predict = torch.max(out,1)
        print(predict)
        print(classes[predict])
        # _,predict = torch.max(out,1)
        # print('Predicted: ', ' '.join('%5s' % classes[predict[j]] for j in range(4)))
        # print('labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))