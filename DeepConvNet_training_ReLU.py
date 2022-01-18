import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules import transformer
from dataloader import read_bci_data
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms
import pandas as pd
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
        correct = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
        # print("testing accuracy:",correct/n)
    return correct/n


def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.uniform(m.weight)
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.08)

train_data, train_label, test_data, test_label = read_bci_data()

filepath = os.path.abspath(os.path.dirname(__file__))+"\checkpoint\DeepConvNet_checkpoint_ReLU.rar"
filepath_csv = os.path.abspath(os.path.dirname(__file__))+"\history_csv\DeepConvNet_ReLU.csv"

n = train_data.shape[0]
epochs = 3000
lr = 1e-3

min_loss = 1
max_accuracy = 0
device = torch.device("cuda:0")

train_data = train_data.astype("float32")
train_label = train_label.astype("float32").reshape(train_label.shape[0],)

# train_data.shape = (1080,1,2,750)
# train_label.shape = (1080,)

x, y = Variable(torch.from_numpy(train_data)),Variable(torch.from_numpy(train_label))
y=torch.tensor(y, dtype=torch.long) 

class DeepConvNet_ReLU(torch.nn.Module):
    def __init__(self, n_output):
        super(DeepConvNet_ReLU, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5),bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Flatten(),
            nn.Linear(8600,n_output,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out

model = DeepConvNet_ReLU(n_output=2)
# model.apply(init_weights)
criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(),lr = 1e-3)
optimizer = optim.RMSprop(model.parameters(),lr = lr, momentum = 0.9, weight_decay=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600], gamma=5e-1)

model.to(device)
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