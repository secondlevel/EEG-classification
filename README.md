# EEG-classification (Summer Course homework 2)
The task is to classify BCI competition datasets(EEG signals) by using EEGNet and DeepConvNet with different activation functions. I have built EEGNet and DeepConvNet by using **pytorch**.

You can get some detailed introduction and experimental results in the link below.  
https://github.com/secondlevel/EEG-classification/blob/main/Experiment%20Report.pdf  

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/149881111-c70eccd7-a0f4-4b6f-aae4-bef73b4814a4.png" width="338" title="training curve" hspace="20"/>
  <img src="https://user-images.githubusercontent.com/44439517/149881402-0e22521c-4d97-4717-b4ec-8ffa9d7ebae4.png" width="500" title="testing result"  hspace="40"/>
</p>

## Hardware
Operating System: Windows 10  

CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  

GPU: NVIDIA GeForce GTX TITAN X  

## Requirement

In this work, I use Anaconda and Pip to manage my environment.

```bash=
$ conda create --name eegenv python=3.8 -y
$ conda activate eegenv
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
Data: [Batch Size, 1, 2, 750]
Label: [Batch Size, 2]
```

You can use the read_bci_data function in the [**dataloader.py**](https://github.com/secondlevel/EEG-classification/blob/main/dataloader.py) file to obtain the training data, training label, testing data and testing label.

```python=
train_data, train_label, test_data, test_label = read_bci_data()
```

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/149894100-82685e56-ee95-4f69-a9e2-961bdaaa0a1e.png" width="800" title="training curve"/>
</p>

## Training

In the training step, there provided six file to train different model.

Each file contains a different model architecture with a different activation function. In addition, you can config the training parameters through the following argparse, and use the following instructions to train different method.  

Finally, you will get such training result. The first picture is about DeepConvNet, and the second picture is about EEGNet.  

You can get some detailed introduction and experimental results in the link below.  
https://github.com/secondlevel/EEG-classification/blob/main/Experiment%20Report.pdf 

```bash=
parser.add_argument('--epochs', type=int, default='700', help='training epochs')
parser.add_argument('--learning_rate', type=float, default='1e-3', help='learning rate')
parser.add_argument('--save_model', action='store_true', help='check if you want to save the model.')
parser.add_argument('--save_csv', action='store_true', help='check if you want to save the training history.')
```

- #### DeepConvNet with ELU

```bash=
python DeepConvNet_training_ELU.py --epochs 3000 --learning_rate 1e-3 --save_model --save_csv
```

- #### DeepConvNet with LeakyReLU

```bash=
python DeepConvNet_training_LeakyReLU.py --epochs 3000 --learning_rate 1e-3 --save_model --save_csv
```

- #### DeepConvNet with ReLU

```bash=
python DeepConvNet_training_ReLU.py --epochs 3000 --learning_rate 1e-3 --save_model --save_csv
```

- #### EEGNet with ELU

```bash=
python EEGNet_training_ELU.py --epochs 700 --learning_rate 1e-3 --save_model --save_csv
```

- #### EEGNet with LeakyReLU

```bash=
python EEGNet_training_LeakyReLU.py --epochs 700 --learning_rate 1e-3 --save_model --save_csv
```

- #### EEGNet with ReLU

```bash=
python EEGNet_training_ReLU.py --epochs 700 --learning_rate 1e-3 --save_model --save_csv
```

- #### DeepConvNet training curve

<p float="center">
   <img src="https://user-images.githubusercontent.com/44439517/149902713-e4d82b0e-4d45-4a80-8100-d61090c24a32.png" width="800" title="EEGNet training curve"/>
</p>

- #### EEGNet training curve

<p float="center">
   <img src="https://user-images.githubusercontent.com/44439517/149901329-097d1238-a1c4-4bf5-a078-bd42ea201a51.png" width="800" hspace="20" title="DeepConvNet training curve"/>
</p>

## Testing

You can display the testing results in different models by using the following commands in combination with different activation functions.
The model checkpoint were in the [**checkpoint**](https://github.com/secondlevel/EEG-classification/tree/main/checkpoint) directory. 

The detailed experimental result are in the following link.  
https://github.com/secondlevel/EEG-classification/blob/main/Experiment%20Report.pdf

```bash=
python model_testing.py
```

Then you will get the best result like this, each of the values were the testing accuracy.

|             | ReLU      |LeakyReLU |ELU       |
|-------------|-----------|----------|----------|
| EEGNet      | 87.1296 %  | 88.2407 % | 87.2222 % |
| DeepConvNet | 85.4630 %  | 84.0741 % | 83.7963 % |

## Performance Metrics

In this project, **Mean Squared Error** is the loss function, and **Accuracy** is the classification metrics. 

### Mean Squared Error(MSE)

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/151695030-61f1f71e-6fb6-498d-bee4-ef85a0b5d959.gif" title="Mean Squared Error(MSE)" width="200" />
</p>  

- **y_j:** ground-truth value
- **y_hat:** predicted value from the regression model
- **N:** number of datums

---

### Crossentropy

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/151697769-2a41096b-5af1-484a-baab-23ced39c2acb.png" title="Mean Squared Error(MSE)" width="180" />
</p>  

- M: number of classes
- log: the natural log
- y: binary indicator (0 or 1) if class label c is the correct classification for observation o
- p: predicted probability observation o is of class c

---

### Accuracy

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/151693558-d2ea220b-607b-41c3-9d03-e01d0682aaed.gif" title="Accuracy" width="300" />
</p>    

- **True Positive(TP)** signifies how many positive class samples your model predicted correctly.
- **True Negative(TN)** signifies how many negative class samples your model predicted correctly.
- **False Positive(FP)** signifies how many negative class samples your model predicted incorrectly. This factor represents Type-I error in statistical nomenclature. This error positioning in the confusion matrix depends on the choice of the null hypothesis.
- **False Negative(FN)** signifies how many positive class samples your model predicted incorrectly. This factor represents Type-II error in statistical nomenclature. This error positioning in the confusion matrix also depends on the choice of the null hypothesis. 

##  Reference
- https://arxiv.org/abs/1611.08024
- https://arxiv.org/pdf/1703.05051.pdf
- https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf  
- https://reurl.cc/QjLZnM  
- https://reurl.cc/k71a5L  
- https://zhuanlan.zhihu.com/p/35709485
