# EEG-classification (Summer Course homework 2)
It is the task to classify BCI competition datasets (EEG signals) using EEGNet and DeepConvNet with different activation functions.  

You can get some detailed introduction and experimental results in the link below.  
https://github.com/secondlevel/EEG-classification/blob/main/Experiment%20Report.pdf  

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/149881111-c70eccd7-a0f4-4b6f-aae4-bef73b4814a4.png" width="338" title="training curve" hspace="20"/>
  <img src="https://user-images.githubusercontent.com/44439517/149881402-0e22521c-4d97-4717-b4ec-8ffa9d7ebae4.png" width="500" title="testing result"  hspace="40"/>
</p>

<p float="center">

</p>

## Hardware
Operating System: Windows 10  

CPU Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  

GPU 0 NVIDIA GeForce GTX TITAN X  

## Requirement

In this work, I use Anaconda and Pip to manage my environment.

```bash=
$ conda create --name eegenv python=3.8
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install numpy
$ conda install matplotlib -y 
$ conda install pandas -y
$ pip install torchsummary
```

## Model Architecture

- ### EEGNet  

<p float="center">
   <img src="https://user-images.githubusercontent.com/44439517/149887574-2e972ab3-77d4-4b2d-8329-f57b709e2e97.png" width="700" title="DeepConvNet architecture"/>
</p>

- ### DeepConvNet

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/149887186-927398ba-f909-4ce3-988a-68efd5f53036.png" width="700" title="DeepConvNet architecture"/>
</p>


##  Reference
- https://arxiv.org/abs/1611.08024
- https://arxiv.org/pdf/1703.05051.pdf
- https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf  
- https://reurl.cc/QjLZnM  
- https://reurl.cc/k71a5L  
- https://zhuanlan.zhihu.com/p/35709485
