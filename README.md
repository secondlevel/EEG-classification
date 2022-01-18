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

The model architecture that combines with different activation function was in the [**ALL_model.py**](https://github.com/secondlevel/EEG-classification/blob/main/ALL_model.py) file.

- ### EEGNet  

<p float="center">
   <img src="https://user-images.githubusercontent.com/44439517/149887574-2e972ab3-77d4-4b2d-8329-f57b709e2e97.png" width="500" title="DeepConvNet architecture"/>
</p>

- ### DeepConvNet

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/149887186-927398ba-f909-4ce3-988a-68efd5f53036.png" width="500" title="DeepConvNet architecture"/>
</p>

## Data Description

In this project, the training and testing data were provided by [**BCI Competition III – IIIb**](http://www.bbci.de/competition/iii/desc_IIIb.pdf) and stored in the **[S4b_test.npz](https://github.com/secondlevel/EEG-classification/blob/main/S4b_test.npz), [S4b_train.npz](https://github.com/secondlevel/EEG-classification/blob/main/S4b_train.npz), [X11b_test.npz](https://github.com/secondlevel/EEG-classification/blob/main/X11b_test.npz) and [X11b_train.npz](https://github.com/secondlevel/EEG-classification/blob/main/X11b_train.npz)** file.  

```bash=
Input: [Batch Size, 1, 2, 750]
Output: [Batch Size, 2]
```

You can use the read_bci_data function in the [**dataloader.py**](https://github.com/secondlevel/EEG-classification/blob/main/dataloader.py) file to obtain the training data, training label, testing data and testing label.

```python=
train_data, train_label, test_data, test_label = read_bci_data()
```

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/149894100-82685e56-ee95-4f69-a9e2-961bdaaa0a1e.png" width="800" title="training curve"/>
</p>

## Training

<p float="center">
   <img src="https://user-images.githubusercontent.com/44439517/149902713-e4d82b0e-4d45-4a80-8100-d61090c24a32.png" width="800" title="EEGNet training curve"/>
</p>

<p float="center">
   <img src="https://user-images.githubusercontent.com/44439517/149901329-097d1238-a1c4-4bf5-a078-bd42ea201a51.png" width="800" title="DeepConvNet training curve"/>
</p>


## Testing

You can display the testing results in different models by using the following commands in combination with different activation functions.
The model checkpoint were in the [**checkpoint**](https://github.com/secondlevel/EEG-classification/tree/main/checkpoint) directory. 

The detailed experimental result are in the following link.  
https://github.com/secondlevel/EEG-classification/blob/main/Experiment%20Report.pdf

```bash=
python model_testing.py
```

Then you will get the best result like this, each of the values were accuracy.

|             | ReLU      |LeakyReLU |ELU       |
|-------------|-----------|----------|----------|
| EEGNet      | 87.1296 %  | 88.2407 % | 87.2222 % |
| DeepConvNet | 85.4630 %  | 84.0741 % | 83.7963 % |


##  Reference
- https://arxiv.org/abs/1611.08024
- https://arxiv.org/pdf/1703.05051.pdf
- https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf  
- https://reurl.cc/QjLZnM  
- https://reurl.cc/k71a5L  
- https://zhuanlan.zhihu.com/p/35709485
