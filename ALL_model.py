import torch.nn as nn
import torch


class DeepConvNet_ELU(torch.nn.Module):
    def __init__(self, n_output):
        super(DeepConvNet_ELU, self).__init__()
        self.model = nn.Sequential(
            # Conv2d(1, 25, kernel_size=(1,5),padding='VALID',bias=False),
            # Conv2d(25, 25, kernel_size=(2,1), padding='VALID',bias=False),
            nn.Conv2d(1, 25, kernel_size=(1,5),bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            # Conv2d(25, 50, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            # Conv2d(50, 100, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            # Conv2d(100, 200, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Flatten(),
            nn.Linear(8600,n_output,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out

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

class DeepConvNet_LeakyReLU(torch.nn.Module):
    def __init__(self, n_output):
        super(DeepConvNet_LeakyReLU, self).__init__()
        self.model = nn.Sequential(
            # Conv2d(1, 25, kernel_size=(1,5),padding='VALID',bias=False),
            # Conv2d(25, 25, kernel_size=(2,1), padding='VALID',bias=False),
            nn.Conv2d(1, 25, kernel_size=(1,5),bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            # Conv2d(25, 50, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.09),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            # Conv2d(50, 100, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            # Conv2d(100, 200, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.09),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Flatten(),
            nn.Linear(8600,n_output,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out

class EEGNet_ELU(torch.nn.Module):
    def __init__(self, n_output):
        super(EEGNet_ELU, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=8,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=0.1),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
            nn.Dropout(p=0.35)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=0.1),
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

class EEGNet_LeakyReLU(torch.nn.Module):
    def __init__(self, n_output):
        super(EEGNet_LeakyReLU, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=8,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.06),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
            nn.Dropout(p=0.35)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.06),
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