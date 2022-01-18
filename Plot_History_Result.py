import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_loss_curve(DeepConvNet_ELU,DeepConvNet_ReLU,DeepConvNet_LeakyReLU,EEGNet_ELU,EEGNet_ReLU,EEGNet_LeakyReLU):

    plt.plot(DeepConvNet_ELU['loss'], '-b', label='DeepConvNet_ELU')
    plt.plot(DeepConvNet_ReLU['loss'], '-g', label='DeepConvNet_ReLU')
    plt.plot(DeepConvNet_LeakyReLU['loss'], '-r', label='DeepConvNet_LeakyReLU')
    plt.plot(EEGNet_ELU['loss'], '-c', label='EEGNet_ELU')
    plt.plot(EEGNet_ReLU['loss'], '-m', label='EEGNet_ReLU')
    plt.plot(EEGNet_LeakyReLU['loss'], '-y', label='EEGNet_LeakyReLU')

    plt.xlabel("Epoch",fontsize=13)
    plt.legend(loc='best')

    plt.ylabel("Loss Value",fontsize=13)
    plt.title("(Loss Curve)Activation function comparision(All)",fontsize=18)

    plt.show()
    return "loss圖繪製成功"

def plot_EEGNet_accuracy_curve(DeepConvNet_ELU,DeepConvNet_ReLU,DeepConvNet_LeakyReLU,EEGNet_ELU,EEGNet_ReLU,EEGNet_LeakyReLU):

    plt.plot(np.array(EEGNet_ELU['train_accuracy_history'])*100, '-b', label='ELU_train')
    plt.plot(np.array(EEGNet_ReLU['train_accuracy_history'])*100, '-g', label='ReLU_train')
    plt.plot(np.array(EEGNet_LeakyReLU['train_accuracy_history'])*100, '-r', label='LeakyReLU_train')

    plt.plot(np.array(EEGNet_ELU['test_accuracy_history'])*100, '-c', label='ELU_test')
    plt.plot(np.array(EEGNet_ReLU['test_accuracy_history'])*100, '-m', label='ReLU_test')
    plt.plot(np.array(EEGNet_LeakyReLU['test_accuracy_history'])*100, '-y', label='LeakyReLU_test')

    plt.xlabel("Epoch",fontsize=13)
    plt.legend(loc='best')

    plt.ylabel("Accuracy(%)",fontsize=13)
    plt.title("Activation function comparision(EGGNet)",fontsize=18)

    plt.show()
    return "EEGNet Accuracy圖繪製成功"

def plot_DeepConvNet_accuracy_curve(DeepConvNet_ELU,DeepConvNet_ReLU,DeepConvNet_LeakyReLU,EEGNet_ELU,EEGNet_ReLU,EEGNet_LeakyReLU):

    plt.plot(np.array(DeepConvNet_ELU['train_accuracy_history'])*100, '-b', label='ELU_train')
    plt.plot(np.array(DeepConvNet_ReLU['train_accuracy_history'])*100, '-g', label='ReLU_train')
    plt.plot(np.array(DeepConvNet_LeakyReLU['train_accuracy_history'])*100, '-r', label='LeakyReLU_train')

    plt.plot(np.array(DeepConvNet_ELU['test_accuracy_history'])*100, '-c', label='ELU_test')
    plt.plot(np.array(DeepConvNet_ReLU['test_accuracy_history'])*100, '-m', label='ReLU_test')
    plt.plot(np.array(DeepConvNet_LeakyReLU['test_accuracy_history'])*100, '-y', label='LeakyReLU_test')

    plt.xlabel("Epoch",fontsize=13)
    plt.legend(loc='best')

    plt.ylabel("Accuracy(%)",fontsize=13)
    plt.title("Activation function comparision(DeepConvNet)",fontsize=18)

    plt.show()
    return "DeepConvNet Accuracy圖繪製成功"

if __name__ == "__main__":
    
    path = os.path.abspath(os.path.dirname(__file__))+"/history_csv/"

    DeepConvNet_ELU = pd.DataFrame(pd.read_csv(path+"DeepConvNet_ELU.csv",encoding="utf-8-sig"))
    DeepConvNet_ReLU = pd.DataFrame(pd.read_csv(path+"DeepConvNet_ReLU.csv",encoding="utf-8-sig"))
    DeepConvNet_LeakyReLU = pd.DataFrame(pd.read_csv(path+"DeepConvNet_LeakyReLU.csv",encoding="utf-8-sig"))

    EEGNet_ELU = pd.DataFrame(pd.read_csv(path+"EEGNet_ELU.csv",encoding="utf-8-sig"))
    EEGNet_ReLU = pd.DataFrame(pd.read_csv(path+"EEGNet_ReLU.csv",encoding="utf-8-sig"))
    EEGNet_LeakyReLU = pd.DataFrame(pd.read_csv(path+"EEGNet_LeakyReLU.csv",encoding="utf-8-sig"))

    # plot_loss_curve(DeepConvNet_ELU,DeepConvNet_ReLU,DeepConvNet_LeakyReLU,EEGNet_ELU,EEGNet_ReLU,EEGNet_LeakyReLU)
    plot_EEGNet_accuracy_curve(DeepConvNet_ELU,DeepConvNet_ReLU,DeepConvNet_LeakyReLU,EEGNet_ELU,EEGNet_ReLU,EEGNet_LeakyReLU)
    # plot_DeepConvNet_accuracy_curve(DeepConvNet_ELU,DeepConvNet_ReLU,DeepConvNet_LeakyReLU,EEGNet_ELU,EEGNet_ReLU,EEGNet_LeakyReLU)

