# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:01:35 2021

@author: haoyuan
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from dataloader import read_bci_data
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import torch.optim as optim
import pandas as pd
from torchsummary import summary
import os

def testing(x_test,y_test,model,device,filepath):

    # model.load_state_dict(torch.load(filepath))
    model.eval()
    with torch.no_grad():
        model.cuda(0)
        n = x_test.shape[0]

        x_test = x_test.astype("float32")
        y_test = y_test.astype("float32").reshape(y_test.shape[0],)

        x_test, y_test = Variable(torch.from_numpy(x_test)),Variable(torch.from_numpy(y_test))
        
        x_test,y_test = x_test.to(device),y_test.to(device)
        y_pred_test = model(x_test)

        correct_test = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
        test_accuracy = correct_test/n
        # print("testing accuracy:",correct/n)
        
    return test_accuracy

torch.manual_seed(1)    # reproducible
epochs = 750
lr = 1e-3

filepath = os.path.abspath(os.path.dirname(__file__))+"\checkpoint\EEGNet_checkpoint_ReLU.rar"
filepath_csv = os.path.abspath(os.path.dirname(__file__))+"\history_csv\EEGNet_ReLU.csv"

min_loss=1
max_accuracy = 0
device = torch.device("cuda:0")

train_data, train_label, test_data, test_label = read_bci_data()

n = train_data.shape[0]

train_data = train_data.astype("float32")
train_label = train_label.astype("float32").reshape(train_label.shape[0],)

# train_data.shape = (1080,1,2,750)
# train_label.shape = (1080,)

# loader = DataLoader(TensorDataset(train_data,train_label),batch_size=8)
x, y = Variable(torch.from_numpy(train_data)),Variable(torch.from_numpy(train_label))
y=torch.tensor(y, dtype=torch.long) 

class EEGNet_ReLU(torch.nn.Module):
    def __init__(self, n_output):
        super(EEGNet_ReLU, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=8,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
            nn.Dropout(p=0.35)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=0),
            nn.Dropout(p=0.35)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(736,n_output,bias=True)
        )

    def forward(self, x):
        out = self.firstConv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out) 
        out = self.classify(out)
        return out

model = EEGNet_ReLU(n_output=2)
print(model)
criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(),lr = lr)
optimizer = optim.RMSprop(model.parameters(),lr = lr, momentum = 0.2)
# optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.5, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,300,500], gamma=0.1)

model.cuda(0)
summary(model.cuda(),(1,2,750))

loss_history = []
train_accuracy_history = []
test_accuracy_history = []

for epoch in range(epochs):
    # for idx,(data,target) in enumerate(loader):
    model.train()
    x,y = x.to(device),y.to(device)
    y_pred = model(x)

    # print(y_pred.shape)
    # print(y.shape)

    # loss  =  F.mse_loss(y_pred, y)

    loss = criterion(y_pred, y)
    train_loss = loss.item()
    loss_history.append(train_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

    if epoch%1==0:

        # correct= (y_pred.ge(0.5) == y).sum().item()
        n = y.shape[0]
        correct = (torch.max(y_pred,1)[1]==y).sum().item()
        train_accuracy = correct / n
        train_accuracy_history.append(train_accuracy)

        # print("epochs:",epoch,"loss:",loss.item(),"Accuracy:",(correct / n),"Learning rate:",scheduler.get_last_lr()[0])
        test_accuracy = testing(test_data,test_label,model,device,filepath)
        test_accuracy_history.append(test_accuracy)

        print("epochs:",epoch,"loss:",train_loss,"Training Accuracy:",train_accuracy,"Testing Accuracy:",test_accuracy,"Learning rate:",scheduler.get_last_lr()[0])

        if train_loss<min_loss:
            min_loss = train_loss
            # torch.save(model.state_dict(), filepath)
        
        if train_accuracy>max_accuracy:
            max_accuracy = train_accuracy
            torch.save(model.state_dict(), filepath)

print("最大的Accuracy為:",max_accuracy,"最小的Loss值為:",min_loss)
df = pd.DataFrame({"loss":loss_history,"train_accuracy_history":train_accuracy_history,"test_accuracy_history":test_accuracy_history})
# print(df)
df.to_csv(filepath_csv,encoding="utf-8-sig")
    